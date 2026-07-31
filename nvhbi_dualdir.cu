// nvhbi_dualdir.cu -- drive ONE fabric direction with two payload sources.
//
//     die A SMs  write into die B L2   -> payload travels A -> B
//     die B SMs  read  from die A HBM  -> request B->A (small),
//                                         PAYLOAD returns A -> B
//
// Both payloads move A->B and are issued by different SMs on different dies, so
// neither source's own issue rate caps the total.
//
// WHAT IT MEASURED (B200, one session, buf_mult=64)
//     write alone, 70 SMs            2320 GB/s
//     read  alone, 76 SMs            1887
//     both  together            1063 + 1075 = 2138
//
// The combination is WORSE than writes alone, and the link splits almost
// exactly 50/50 between the two sources with under 0.1% run-to-run spread. So
// writes on their own already saturate this direction; reads do not add
// headroom, they take a share and cost about 8% in mixing. The hypothesis this
// program was written to test -- that the scattered store pattern was leaving
// fabric capacity unused -- is refuted. Other sessions put write-alone at
// 3.4-3.5 TB/s; the instance-to-instance spread is about 1.5x, so only compare
// numbers from one run.
//
// Design points that were settled by measurement, and why they look like this:
//
//  * ONE kernel, not two. As separate launches each asked for a full 148x32
//    grid, oversubscribing the GPU 2x, so they fought over SM slots instead of
//    the link: the write counter read exactly 0.00 on the first repeat of every
//    point. One grid gives every SM its blocks and leaves the fabric as the
//    only shared resource.
//
//  * Reads use a 128B stride, one 4B load per line, ONE instruction per 4KiB
//    chunk. A contiguous 16B/lane version that reads all 4096 B explicitly was
//    measured alongside it: 1909 vs 1826 GB/s, i.e. the same, with 8x the
//    instructions. Equality also settles the granularity question -- if a 4B
//    load pulled only a 32B sector, the strided count would have overstated
//    traffic 4x and read ~4x higher. It does not, so a 4B remote load really
//    drags a whole 128B line across.
//
//  * Reads must STREAM over a footprint far larger than L2. A remote read
//    leaves a replica on the reading die, so a chunk touched twice is served
//    locally and never crosses again. Read bandwidth fell 5960 -> 4820 -> 2390
//    -> 1920 GB/s as the sweep grew 0.5 -> 1 -> 2 -> 4 GB; only the last is
//    fabric traffic. Hence buf_mult defaults to 64.
//
//  * ld.global.cg. The .cs (evict-first) and .cv (re-fetch) variants were tried:
//    .cs moved read from 1076 to 1129 GB/s and left write unchanged, .cv the
//    same. Cache hints cannot protect the writers' working set when reads churn
//    the whole L2 every ~33 us.
//
//  * NVHBI_R_LOCAL=1 keeps the readers on their own die. It is the validity
//    control: it fills die B's L2 just as hard but crosses nothing, and it
//    separates the two ways reads hurt writes -- local reads cost writes 16%
//    (eviction), crossing reads cost 54% (eviction plus fabric).
//
// argv: [writer_partition]  0|1   (die A = writing die; readers sit on die B)
//
// Env: NVHBI_W_SMS      writer SM counts, 0=off.  default "0,70"
//      NVHBI_R_SMS      reader SM counts, 0=off.  default "0,8,16,32,64,78"
//      NVHBI_R_CHUNKS   read sweep in 4KiB chunks, 0 = whole source die (default)
//      NVHBI_R_LOCAL    1 = readers read their OWN die (crosses nothing), default 0
//      NVHBI_BUF_MULT   allocation as a multiple of L2, default 64 (~4 GB/die)
//      NVHBI_WINDOW_MS  default 200      NVHBI_REPEAT default 3
//      NVHBI_BLOCK / NVHBI_BLOCKS_PER_SM   default 64 / 32

#include "nvhbi_common.cuh"
#include <chrono>

// L1-bypassing 4B load. One per 128B line is all it takes to pull the line.
__device__ __forceinline__ unsigned int nvhbi_ld1(const unsigned int* addr) {
    unsigned int v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}

/* ---------------------------------------------------------------------------
   One launch, two roles, split by which die the SM sits on. Writers push A->B,
   readers pull A->B. Both count in 32B sectors so the counters simply add.
   --------------------------------------------------------------------------- */
__global__ void nvhbi_dual(unsigned int* __restrict__ data,
                           const unsigned int* __restrict__ far_idx,
                           const unsigned int* __restrict__ near_idx,
                           const unsigned int* __restrict__ sm_side,
                           unsigned int wp,              // die A: the writing die
                           unsigned int w_active_sm,
                           unsigned int r_active_sm,
                           unsigned int r_local,
                           unsigned int nbps,
                           unsigned int sm_count,
                           unsigned int r_count,         // chunks in the read sweep
                           unsigned long long deadline_cycles,
                           const unsigned int* __restrict__ stop_flag,
                           unsigned long long* __restrict__ w_prog,
                           unsigned long long* __restrict__ r_prog,
                           unsigned int* __restrict__ sink) {
    const unsigned int smid = nvhbi_smid();
    const unsigned int part = sm_side[smid] % 2u;
    const unsigned int rank = sm_side[smid] / 2u;

    const unsigned int wpb  = (blockDim.x + 31u) / 32u;
    const unsigned int wib  = threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x % 32u;
    const unsigned int q    = blockIdx.x / sm_count;
    const unsigned long long lanes =
        (unsigned long long)min(32u, blockDim.x - wib * 32u);

    const bool is_writer = (part == wp);
    if (rank >= (is_writer ? w_active_sm : r_active_sm)) return;

    const unsigned int slot = wib + wpb * (q + nbps * rank);
    unsigned long long done = 0ull;
    unsigned int acc = 0u;
    const unsigned long long t0 = clock64();

    if (is_writer) {
        // Writers on die A target die B: "far from SM0" when A is partition 0.
        const unsigned int* list = (wp == 0u) ? far_idx : near_idx;
        unsigned int val = smid * 1000003u + threadIdx.x + 1u;
#pragma unroll 1
        for (unsigned int it = 0; ; ++it) {
            nvhbi_store_group(data, list[slot], lane, val);
            ++val;
            done += 4ull;
            if ((it & NVHBI_POLL_MASK) == NVHBI_POLL_MASK) {
                if (w_prog && lane == 0u) { atomicAdd(w_prog, done * lanes); done = 0ull; }
                if ((unsigned long long)(clock64() - t0) > deadline_cycles) break;
                if (stop_flag && *(volatile const unsigned int*)stop_flag) break;
            }
        }
        if (w_prog && lane == 0u && done) atomicAdd(w_prog, done * lanes);
    } else {
        if (r_count == 0u) return;
        // Readers on die B pull from die A (or from die B for the control).
        // Each active warp starts on its own chunk and advances by the number
        // of active warps, so together they sweep the list linearly and revisit
        // a chunk only after a full pass -- by which time its replica is gone.
        const unsigned int* list = r_local ? ((wp == 0u) ? far_idx  : near_idx)
                                           : ((wp == 0u) ? near_idx : far_idx);
        const unsigned int nwarps = wpb * nbps * r_active_sm;
        unsigned int c = (slot >= r_count) ? (slot % r_count) : slot;
#pragma unroll 1
        for (unsigned int it = 0; ; ++it) {
            // byte 128*lane: 32 lanes cover the 32 lines of the 4KiB chunk,
            // one instruction, 4 sectors pulled per lane.
            acc += nvhbi_ld1(&data[list[c] + 32u * lane]);
            done += 4ull;
            c += nwarps;
            if (c >= r_count) c = (c >= 2u * r_count) ? (c % r_count) : (c - r_count);
            if ((it & NVHBI_POLL_MASK) == NVHBI_POLL_MASK) {
                if (r_prog && lane == 0u) { atomicAdd(r_prog, done * lanes); done = 0ull; }
                if ((unsigned long long)(clock64() - t0) > deadline_cycles) break;
                if (stop_flag && *(volatile const unsigned int*)stop_flag) break;
            }
        }
        if (r_prog && lane == 0u && done) atomicAdd(r_prog, done * lanes);
    }
    nvhbi_st(&sink[smid], acc + (unsigned int)done);
}

/* ------------------------------------------------------------------ host */

static unsigned int env_u(const char* k, unsigned int d) {
    const char* s = getenv(k);
    return (s && *s) ? (unsigned int)atoi(s) : d;
}
static int parse_list(const char* env, const char* dflt, unsigned int* out, int cap) {
    const char* s = getenv(env);
    if (!s || !*s) s = dflt;
    int n = 0; char buf[512];
    snprintf(buf, sizeof(buf), "%s", s);
    for (char* tok = strtok(buf, ","); tok && n < cap; tok = strtok(nullptr, ",")) {
        int v = atoi(tok);
        if (v >= 0) out[n++] = (unsigned int)v;
    }
    return n;
}
static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
static void spin_ms(double ms) { const double u = now_ms() + ms; while (now_ms() < u) {} }

int main(int argc, char** argv) {
    const unsigned int wp = (argc > 1) ? (unsigned int)atoi(argv[1]) : 0u;   // die A
    const unsigned int rp = 1u - wp;                                          // die B

    const unsigned int nbps      = env_u("NVHBI_BLOCKS_PER_SM", 32u);
    const unsigned int block     = env_u("NVHBI_BLOCK", 64u);
    const unsigned int window_ms = env_u("NVHBI_WINDOW_MS", 200u);
    const unsigned int repeat    = env_u("NVHBI_REPEAT", 3u);
    const unsigned int r_local   = env_u("NVHBI_R_LOCAL", 0u);
    const double       buf_mult  = (double)env_u("NVHBI_BUF_MULT", 64u);

    unsigned int w_list[16], r_list[16];
    const int w_n = parse_list("NVHBI_W_SMS", "0,70", w_list, 16);
    const int r_n = parse_list("NVHBI_R_SMS", "0,8,16,32,64,78", r_list, 16);

    NvhbiTopo t;
    nvhbi_probe(t, 0, buf_mult);

    const unsigned int w_max = (wp == 0u) ? t.sms_p0 : t.sms_p1;
    const unsigned int r_max = (rp == 0u) ? t.sms_p0 : t.sms_p1;
    for (int i = 0; i < w_n; ++i) if (w_list[i] > w_max) w_list[i] = w_max;
    for (int i = 0; i < r_n; ++i) if (r_list[i] > r_max) r_list[i] = r_max;

    const unsigned int w_target_die = rp;
    const unsigned int r_source_die = r_local ? rp : wp;
    const unsigned int* d_wlist = (w_target_die == 1u) ? t.d_far_idx : t.d_near_idx;
    const unsigned int  r_avail = (r_source_die == 1u) ? t.far_count : t.near_count;

    unsigned int r_chunks = env_u("NVHBI_R_CHUNKS", 0u);
    if (!r_chunks || r_chunks > r_avail) r_chunks = r_avail;

    unsigned int w_max_chunks = 0;
    for (int i = 0; i < w_n; ++i)
        w_max_chunks = max(w_max_chunks, nvhbi_chunks_used(w_list[i], nbps, block, 1u));

    printf("die%u -> die%u direction, two payload sources, one fused kernel\n", wp, rp);
    printf("  writers: die%u SMs -> die%u L2,  %u chunks (%.1f MB), warmed\n",
           wp, w_target_die, w_max_chunks, w_max_chunks * 4096.0 / 1048576.0);
    printf("  readers: die%u SMs <- die%u HBM, streaming %u chunks (%.1f MB = %.0fx per-die L2)%s\n",
           rp, r_source_die, r_chunks, r_chunks * 4096.0 / 1048576.0,
           r_chunks * 4096.0 / (t.l2_bytes / 2.0),
           r_local ? "   [CONTROL: crosses nothing]" : "");
    printf("# CFG,w_sms,r_sms,r_local,rep,write_GBps,read_GBps,total_GBps\n");

    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    unsigned long long *d_wprog = nullptr, *d_rprog = nullptr;
    CHECK_CUDA(cudaMalloc(&d_wprog, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_rprog, sizeof(unsigned long long)));
    NvhbiStopFlag stop;
    nvhbi_stop_flag_create(stop);

    for (int wi = 0; wi < w_n; ++wi) {
    for (int ri = 0; ri < r_n; ++ri) {
        const unsigned int w_sms = w_list[wi];
        const unsigned int r_sms = r_list[ri];
        if (!w_sms && !r_sms) continue;
        const unsigned int w_chunks = w_sms ? nvhbi_chunks_used(w_sms, nbps, block, 1u) : 0u;

        for (unsigned int rep = 0; rep < repeat; ++rep) {
            nvhbi_flush_l2(t);
            // Only the write targets are warmed. The read sweep is far larger
            // than L2 by construction, so it comes from the source die's HBM.
            if (w_chunks) {
                nvhbi_warm_chunks<<<t.sm_count * 8, 128>>>(
                    t.d_data, d_wlist, 0u, w_chunks, t.d_sm_side, w_target_die, t.d_sink);
                CHECK_CUDA(cudaGetLastError());
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemset(d_wprog, 0, sizeof(unsigned long long)));
            CHECK_CUDA(cudaMemset(d_rprog, 0, sizeof(unsigned long long)));
            nvhbi_stop_flag_reset(stop);
            const unsigned long long dl =
                (unsigned long long)(4u * window_ms + 2000u) * (unsigned long long)t.clock_khz;

            nvhbi_dual<<<t.sm_count * nbps, block, 0, s>>>(
                t.d_data, t.d_far_idx, t.d_near_idx, t.d_sm_side,
                wp, w_sms, r_sms, r_local, nbps, (unsigned int)t.sm_count,
                r_sms ? r_chunks : 0u, dl, stop.d, d_wprog, d_rprog, t.d_sink);
            CHECK_CUDA(cudaGetLastError());
            spin_ms(100.0);

            unsigned long long w0 = 0, r0 = 0, w1 = 0, r1 = 0;
            CHECK_CUDA(cudaMemcpy(&w0, d_wprog, sizeof(w0), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(&r0, d_rprog, sizeof(r0), cudaMemcpyDeviceToHost));
            const double t_start = now_ms();
            spin_ms((double)window_ms);
            const double wall = now_ms() - t_start;
            CHECK_CUDA(cudaMemcpy(&w1, d_wprog, sizeof(w1), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(&r1, d_rprog, sizeof(r1), cudaMemcpyDeviceToHost));

            nvhbi_stop_flag_set(stop);
            CHECK_CUDA(cudaStreamSynchronize(s));

            const double wg = (double)(w1 - w0) * 32.0 / (wall * 1e-3) / 1e9;
            const double rg = (double)(r1 - r0) * 32.0 / (wall * 1e-3) / 1e9;
            // Both payloads travel die A -> die B, so for a crossing read the
            // sum is what that one direction carried.
            printf("CFG,%u,%u,%u,%u,%.2f,%.2f,%.2f\n",
                   w_sms, r_sms, r_local, rep, wg, rg, wg + (r_local ? 0.0 : rg));
            fflush(stdout);
        }
    }}

    CHECK_CUDA(cudaStreamDestroy(s));
    CHECK_CUDA(cudaFree(d_wprog));
    CHECK_CUDA(cudaFree(d_rprog));
    nvhbi_stop_flag_destroy(stop);
    nvhbi_free(t);
    return 0;
}
