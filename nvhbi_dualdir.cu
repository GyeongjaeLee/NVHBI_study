// nvhbi_dualdir.cu -- load ONE fabric direction from two sources at once.
//
//     die A SMs  write into die B memory  -> payload travels A -> B
//     die B SMs  read  from die A memory  -> request B->A (small),
//                                            PAYLOAD returns A -> B
//
// Both payloads move A->B, and they are issued by different SMs on different
// dies, so neither mechanism's own issue-rate ceiling caps the total. Writes
// alone stop around 3.4 TB/s because the scattered store pattern costs 32
// transactions per instruction; the reads are here to find out whether that is
// the fabric's limit or only the write path's.
//
// ONE kernel, not two
// -------------------
// Writers and readers share a single launch and are separated by which die the
// SM sits on. As two separate kernels they each asked for a full 148x32 grid,
// oversubscribing the GPU 2x, so they contended for SM slots rather than for
// the link: the write counter read exactly 0.00 on the first repeat of every
// point and about half its solo rate afterwards. With one grid every SM gets
// its nbps blocks and the only shared resource left is the fabric.
//
// Read shape
// ----------
// Contiguous 16B per lane, eight instructions covering the 4KiB chunk. A
// 128B-stride variant (one 4B load per lane, one instruction per chunk) was
// tried and removed: an L2 line is 128B but fills per 32B sector, so a 4B load
// pulls 32B, not 128B. Counting it as 128B overstated read bandwidth 4x -- it
// reported 18.5 TB/s for a local streaming read, more than HBM can supply.
//
// Reads must STREAM. A remote read leaves a replica on the reading die, so a
// chunk touched twice is served locally the second time and never crosses
// again. Sweeping the whole source die (506 MB against a 63 MB L2) fixed it:
// read bandwidth fell 18948 -> 6857 -> 3319 -> 2985 GB/s as the footprint grew
// 8 -> 64 -> 256 -> 506 MB, converging once reuse became impossible. The 8 MB
// figure matched the local control exactly, which is what a pure replica hit
// looks like.
//
// argv: [writer_partition]  0|1   (die A = writing die; readers sit on die B)
//
// Env: NVHBI_W_SMS      writer SM counts, 0=off.  default "0,70"
//      NVHBI_R_SMS      reader SM counts, 0=off.  default "0,16,32,64,78"
//      NVHBI_R_CHUNKS   read sweep in 4KiB chunks, 0 = whole source die (default)
//      NVHBI_R_LOCAL    1 = readers read their OWN die (crosses nothing), default 0
//      NVHBI_WINDOW_MS  default 200      NVHBI_REPEAT default 3
//      NVHBI_BLOCK / NVHBI_BLOCKS_PER_SM   default 64 / 32
//      NVHBI_BUF_MULT   default 8
//
// Build switches:
//      -DNVHBI_READ_OP=0|1|2   .cg (default) | .cv | .cs
//          .cs marks the replica evict-first, protecting the writers' warmed
//          lines, but can also drop a line between the lanes that share it.
//          .cv forces a re-fetch. .cg is the plain L1-bypassing load.
//      -DNVHBI_STORE_MODE=0|3  scattered (default) | contiguous, see nvhbi_common.cuh

#include "nvhbi_common.cuh"
#include <chrono>

#ifndef NVHBI_READ_OP
#define NVHBI_READ_OP 0
#endif
// Read shape. Both cover the same 4KiB chunk and count the same 128 sectors,
// but they bet differently on the cross-die transfer granularity:
//   0 contiguous 16B/lane, 8 instructions. Reads all 4096 B explicitly, so the
//     count is right whatever the granularity -- but if a line really crosses
//     as 128 B, then the 8 lanes sharing a line make 7/8 of the loads redundant
//     for fabric purposes while still costing SM issue slots.
//   1 128B stride, 1 instruction: one 4B load per line, 32 lanes -> 32 lines.
//     Correct only if a 4B load really drags the whole 128 B across.
// Comparing them at a large NVHBI_BUF_MULT settles it:
//   equal read_GBps      -> 128 B granularity, and stride is 8x cheaper to issue
//   stride ~4x higher    -> 32 B sectors, stride overcounts by 4
#ifndef NVHBI_READ_PATTERN
#define NVHBI_READ_PATTERN 0
#endif
#if   NVHBI_READ_OP == 2
#define NVHBI_LDOP "cs"
#elif NVHBI_READ_OP == 1
#define NVHBI_LDOP "cv"
#else
#define NVHBI_LDOP "cg"
#endif

__device__ __forceinline__ void nvhbi_ld4(const unsigned int* addr,
                                          unsigned int& a, unsigned int& b,
                                          unsigned int& c, unsigned int& d) {
    asm volatile("ld.global." NVHBI_LDOP ".v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(addr));
}
__device__ __forceinline__ unsigned int nvhbi_ld1(const unsigned int* addr) {
    unsigned int v;
    asm volatile("ld.global." NVHBI_LDOP ".u32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}

// Pull one 4KiB chunk. Both shapes count 4 sectors per lane.
__device__ __forceinline__ unsigned int nvhbi_read_chunk(const unsigned int* data,
                                                         unsigned int cidx,
                                                         unsigned int lane) {
    unsigned int acc = 0u;
#if NVHBI_READ_PATTERN == 1
    acc = nvhbi_ld1(&data[cidx + 32u * lane]);      // byte 128*lane: one line per lane
#else
#pragma unroll
    for (unsigned int k = 0; k < 8u; ++k) {
        unsigned int a, b, c, d;
        nvhbi_ld4(&data[cidx + 128u * k + 4u * lane], a, b, c, d);
        acc += a + b + c + d;
    }
#endif
    return acc;
}

/* ---------------------------------------------------------------------------
   One launch, two roles. Writers push A->B, readers pull A->B. Both count in
   32B sectors, so the two counters are directly addable.
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
        // Writers on die A target die B: the "far from SM0" list when A is
        // partition 0, the "near" list otherwise.
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
        // Readers on die B pull from die A (or from die B itself for the
        // no-crossing control). Each active warp starts on its own chunk and
        // advances by the number of active warps, so together they sweep the
        // list linearly and revisit a chunk only after a full pass.
        const unsigned int* list = r_local ? ((wp == 0u) ? far_idx  : near_idx)
                                           : ((wp == 0u) ? near_idx : far_idx);
        const unsigned int nwarps = wpb * nbps * r_active_sm;
        unsigned int c = (slot >= r_count) ? (slot % r_count) : slot;
#pragma unroll 1
        for (unsigned int it = 0; ; ++it) {
            acc += nvhbi_read_chunk(data, list[c], lane);
            done += 4ull;                  // 4096 B per warp = 4 sectors per lane
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
    // Big by default. At buf_mult=8 the sweep was 506 MB, only 8x one die's L2,
// and that was not enough: the LOCAL control read 12.2 TB/s, about 3x what a
// single die's HBM can supply, so most of it was still L2 hits. 32 gives ~2 GB
// per die, ~32x the per-die L2.
    const double       buf_mult  = (double)env_u("NVHBI_BUF_MULT", 32u);

    unsigned int w_list[16], r_list[16];
    const int w_n = parse_list("NVHBI_W_SMS", "0,70", w_list, 16);
    const int r_n = parse_list("NVHBI_R_SMS", "0,16,32,64,78", r_list, 16);

    NvhbiTopo t;
    nvhbi_probe(t, 0, buf_mult);

    const unsigned int w_max = (wp == 0u) ? t.sms_p0 : t.sms_p1;
    const unsigned int r_max = (rp == 0u) ? t.sms_p0 : t.sms_p1;
    for (int i = 0; i < w_n; ++i) if (w_list[i] > w_max) w_list[i] = w_max;
    for (int i = 0; i < r_n; ++i) if (r_list[i] > r_max) r_list[i] = r_max;

    const unsigned int w_target_die = rp;                       // writers push to die B
    const unsigned int r_source_die = r_local ? rp : wp;        // readers pull from die A
    const unsigned int* d_wlist = (w_target_die == 1u) ? t.d_far_idx : t.d_near_idx;
    const unsigned int  r_avail = (r_source_die == 1u) ? t.far_count : t.near_count;

    unsigned int r_chunks = env_u("NVHBI_R_CHUNKS", 0u);
    if (!r_chunks || r_chunks > r_avail) r_chunks = r_avail;

    unsigned int w_max_chunks = 0;
    for (int i = 0; i < w_n; ++i)
        w_max_chunks = max(w_max_chunks, nvhbi_chunks_used(w_list[i], nbps, block, 1u));

    printf("dual-source load on the die%u -> die%u direction (single fused kernel)\n", wp, rp);
    printf("  writers: die%u SMs -> die%u memory, %u chunks (%.1f MB), warmed\n",
           wp, w_target_die, w_max_chunks, w_max_chunks * 4096.0 / 1048576.0);
    printf("  readers: die%u SMs <- die%u memory, streaming %u chunks (%.1f MB = %.1fx per-die L2)%s\n",
           rp, r_source_die, r_chunks, r_chunks * 4096.0 / 1048576.0,
           r_chunks * 4096.0 / (t.l2_bytes / 2.0),
           r_local ? "  [LOCAL CONTROL: crosses nothing]" : "");
    printf("  loads: ld.global.%s, %s\n", NVHBI_LDOP,
           NVHBI_READ_PATTERN == 1 ? "128B stride (1 instr/chunk)"
                                   : "contiguous 16B/lane (8 instr/chunk)");
    // A read served from one die's HBM cannot beat that die's HBM bandwidth.
    // Anything far above it is L2 reuse, not fabric traffic, so print the bar.
    printf("  NOTE: reads come from ONE die's HBM. Anything far above that die's\n"
           "        HBM bandwidth (~half the GPU's) is L2 reuse, not crossing --\n"
           "        raise NVHBI_BUF_MULT until read_GBps stops falling.\n");
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
            // than L2 by construction, so it comes from the source die anyway.
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
