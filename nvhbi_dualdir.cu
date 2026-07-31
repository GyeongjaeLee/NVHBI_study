// nvhbi_dualdir.cu -- TEMPORARY: load ONE fabric direction from two sources.
//
// Writes alone cannot saturate the A->B direction: the scattered store pattern
// costs 32 transactions per instruction and the SM issue path gives out around
// 2.9 TB/s. Reads let us push the same direction with a second, independent
// mechanism:
//
//     die A SMs  write into die B memory   -> data travels A -> B
//     die B SMs  read  from die A memory   -> request B->A (small),
//                                             DATA returns A -> B
//
// So both payload flows are A->B and they are issued by different SMs on
// different dies, which means neither one's issue-rate ceiling caps the total.
//
// Two hazards the reads have to dodge, both real:
//
//  1. Replica. A remote read was measured to leave a local copy, so re-reading
//     the same lines would hit on die B and never cross again. Loads therefore
//     use ld.global.cv ("re-fetch, treat as volatile"), which is meant to defeat
//     exactly that. Build with -DNVHBI_READ_CG=1 to fall back to .cg and compare:
//     if .cg reports much higher read bandwidth than .cv, .cg is being served by
//     replicas and only the .cv number is real fabric traffic.
//
//  2. Eviction. Read traffic must not push the warmed write targets out of die
//     B's L2, or the writes turn into misses and the experiment measures HBM.
//     Keep NVHBI_R_CHUNKS small and watch whether write bandwidth drops when
//     reads are switched on but the read footprint is grown.
//
// Sanity check built in: with NVHBI_R_LOCAL=1 the readers read their OWN die.
// That crosses nothing, so if remote and local read bandwidths are equal, the
// reads are not crossing and the numbers mean nothing.
//
// argv: [writer_partition]  0|1   (die A = the writing die; readers sit on B)
//
// Env: NVHBI_W_SMS      writer SM counts, 0=off.  default "0,70"
//      NVHBI_R_SMS      reader SM counts, 0=off.  default "0,16,32,64,78"
//      NVHBI_R_CHUNKS   read footprint in 4KiB chunks, default 2048 (8MB)
//      NVHBI_R_LOCAL    1 = readers read their own die (control), default 0
//      NVHBI_WINDOW_MS  default 200      NVHBI_REPEAT default 3
//      NVHBI_BLOCK / NVHBI_BLOCKS_PER_SM  default 64 / 32
//      NVHBI_BUF_MULT   default 8

#include "nvhbi_common.cuh"
#include <chrono>

// Read shape and cache policy, both switchable at build time.
//   NVHBI_READ_PATTERN 0 = 128B stride, ONE instruction per 4KiB chunk. Each
//                          lane's 4B load drags a whole 128B line across, so 32
//                          lanes pull all 32 lines of the chunk. Maximum fabric
//                          bytes per instruction.
//                      1 = contiguous 16B per lane, 8 instructions per chunk.
//                          Wasteful for reads: lanes 0-7 share one 128B line,
//                          so seven of them just re-read what already arrived.
//   NVHBI_READ_OP      0 = .cs streaming (evict-first, protects the write set)
//                      1 = .cv volatile (forced re-fetch)
//                      2 = .cg ordinary L1-bypassing
#ifndef NVHBI_READ_PATTERN
#define NVHBI_READ_PATTERN 0
#endif
#ifndef NVHBI_READ_OP
#define NVHBI_READ_OP 0
#endif

#if   NVHBI_READ_OP == 2
#define NVHBI_LDOP "cg"
#elif NVHBI_READ_OP == 1
#define NVHBI_LDOP "cv"
#else
#define NVHBI_LDOP "cs"
#endif

__device__ __forceinline__ unsigned int nvhbi_ld1(const unsigned int* addr) {
    unsigned int v;
    asm volatile("ld.global." NVHBI_LDOP ".u32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}
__device__ __forceinline__ void nvhbi_ld4(const unsigned int* addr,
                                          unsigned int& a, unsigned int& b,
                                          unsigned int& c, unsigned int& d) {
    asm volatile("ld.global." NVHBI_LDOP ".v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(addr));
}

// Pull one 4KiB chunk across the fabric. Both patterns move the same 4096 B and
// count the same 4 sectors per lane; they differ only in how many instructions
// it costs.
__device__ __forceinline__ unsigned int nvhbi_read_chunk(const unsigned int* data,
                                                         unsigned int cidx,
                                                         unsigned int lane) {
    unsigned int acc = 0u;
#if NVHBI_READ_PATTERN == 1
#pragma unroll
    for (unsigned int k = 0; k < 8u; ++k) {
        unsigned int a, b, c, d;
        nvhbi_ld4(&data[cidx + 128u * k + 4u * lane], a, b, c, d);
        acc += a + b + c + d;
    }
#else
    // byte offset 128*lane: 32 lanes -> 32 distinct 128B lines -> the whole chunk
    acc = nvhbi_ld1(&data[cidx + 32u * lane]);
#endif
    return acc;
}

// Readers on `reader_partition` pull from the OTHER die, so the returned data
// flows toward them -- the same direction the writers push.
//
// STREAMING is the whole point. A remote read leaves a 128B replica on the
// reading die, so any chunk touched twice is served locally the second time and
// never crosses again. The earlier version cycled a small footprint and measured
// pure local L2 (11 TB/s, 3x the fabric). Here the active warps sweep the entire
// source-die chunk list linearly -- warp w takes chunk w, w+N, w+2N, ... -- so a
// chunk is revisited only after a full ~500 MB sweep, by which time its replica
// has long been evicted from a ~63 MB L2.
//
// The source is far larger than any cache, so reads come from the source die's
// HBM. That is fine and on a separate die from the writes' destination.
__global__ void nvhbi_stream_read(const unsigned int* __restrict__ data,
                                  const unsigned int* __restrict__ idx_list,
                                  unsigned int count,
                                  const unsigned int* __restrict__ sm_side,
                                  unsigned int reader_partition,
                                  unsigned int num_active_sm,
                                  unsigned int num_blocks_per_sm,
                                  unsigned int sm_count,
                                  unsigned long long deadline_cycles,
                                  const unsigned int* __restrict__ stop_flag,
                                  unsigned long long* __restrict__ progress,
                                  unsigned int* __restrict__ sink) {
    const unsigned int smid = nvhbi_smid();
    if ((sm_side[smid] % 2u) != reader_partition) return;
    const unsigned int sm_rank = sm_side[smid] / 2u;
    if (sm_rank >= num_active_sm || count == 0u) return;

    const unsigned int wpb  = (blockDim.x + 31u) / 32u;
    const unsigned int wib  = threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x % 32u;
    const unsigned int q    = blockIdx.x / sm_count;
    const unsigned long long lanes =
        (unsigned long long)min(32u, blockDim.x - wib * 32u);

    // Position in the sweep, and the stride that keeps every active warp on a
    // distinct chunk.
    const unsigned int nwarps = wpb * num_blocks_per_sm * num_active_sm;
    unsigned int c = wib + wpb * (q + num_blocks_per_sm * sm_rank);
    if (c >= count) c %= count;

    unsigned int acc = 0u;
    unsigned long long done = 0ull;
    const unsigned long long t0 = clock64();

#pragma unroll 1
    for (unsigned int it = 0; ; ++it) {
        acc += nvhbi_read_chunk(data, idx_list[c], lane);
        done += 4ull;                       // 4 sectors per lane, as on the write side
        c += nwarps;
        // One subtraction suffices while nwarps < count (the normal case); the
        // modulo covers a deliberately tiny NVHBI_R_CHUNKS.
        if (c >= count) c = (c >= 2u * count) ? (c % count) : (c - count);
        if ((it & NVHBI_POLL_MASK) == NVHBI_POLL_MASK) {
            if (progress && lane == 0u) { atomicAdd(progress, done * lanes); done = 0ull; }
            if ((unsigned long long)(clock64() - t0) > deadline_cycles) break;
            if (stop_flag && *(volatile const unsigned int*)stop_flag) break;
        }
    }
    if (progress && lane == 0u && done) atomicAdd(progress, done * lanes);
    nvhbi_st(&sink[smid], acc);
}

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
    const unsigned int r_chunks_req = env_u("NVHBI_R_CHUNKS", 0u);  // 0 = all
    const unsigned int r_local   = env_u("NVHBI_R_LOCAL", 0u);
    const double       buf_mult  = (double)env_u("NVHBI_BUF_MULT", 8u);

    unsigned int w_list[16], r_list[16];
    const int w_n = parse_list("NVHBI_W_SMS", "0,70", w_list, 16);
    const int r_n = parse_list("NVHBI_R_SMS", "0,16,32,64,78", r_list, 16);

    NvhbiTopo t;
    nvhbi_probe(t, 0, buf_mult);

    const unsigned int w_max = (wp == 0u) ? t.sms_p0 : t.sms_p1;
    const unsigned int r_max = (rp == 0u) ? t.sms_p0 : t.sms_p1;
    for (int i = 0; i < w_n; ++i) if (w_list[i] > w_max) w_list[i] = w_max;
    for (int i = 0; i < r_n; ++i) if (r_list[i] > r_max) r_list[i] = r_max;

    // Writers on die A push into die B. Readers on die B pull from die A
    // (or from die B itself when NVHBI_R_LOCAL=1, the no-crossing control).
    const unsigned int w_target_die = rp;
    const unsigned int r_source_die = r_local ? rp : wp;
    const unsigned int* d_wlist = (w_target_die == 1u) ? t.d_far_idx : t.d_near_idx;
    const unsigned int* d_rlist = (r_source_die == 1u) ? t.d_far_idx : t.d_near_idx;

    unsigned int w_max_chunks = 0;
    for (int i = 0; i < w_n; ++i)
        w_max_chunks = max(w_max_chunks, nvhbi_chunks_used(w_list[i], nbps, block, 1u));
    const unsigned int r_avail = (r_source_die == 1u) ? t.far_count : t.near_count;
    // Default to the entire source die. The sweep must be far larger than the
    // reading die's L2 or the replicas get reused and nothing crosses.
    unsigned int r_chunks = r_chunks_req ? r_chunks_req : r_avail;
    if (r_chunks > r_avail) r_chunks = r_avail;

    printf("dual-direction load on the die%u -> die%u direction\n", wp, rp);
    printf("  writers: die%u SMs -> die%u memory (%u chunks, %.1f MB)\n",
           wp, w_target_die, w_max_chunks, w_max_chunks * 4096.0 / 1048576.0);
    printf("  readers: die%u SMs <- die%u memory, STREAMING %u chunks (%.1f MB)%s\n",
           rp, r_source_die, r_chunks, r_chunks * 4096.0 / 1048576.0,
           r_local ? "  [LOCAL CONTROL: crosses nothing]" : "");
    printf("           sweep is %.0fx the per-die L2, so a chunk's replica is gone\n"
           "           before the sweep returns to it\n",
           r_chunks * 4096.0 / (t.l2_bytes / 2.0));
    printf("  loads: ld.global.%s, pattern=%s\n", NVHBI_LDOP,
           NVHBI_READ_PATTERN == 1 ? "contiguous 16B/lane (8 instr/chunk)"
                                   : "128B stride (1 instr/chunk)");
    printf("# CFG,w_sms,r_sms,r_local,rep,write_GBps,read_GBps,total_GBps\n");

    cudaStream_t sw, sr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&sw, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&sr, cudaStreamNonBlocking));
    unsigned long long *d_wprog = nullptr, *d_rprog = nullptr;
    CHECK_CUDA(cudaMalloc(&d_wprog, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_rprog, sizeof(unsigned long long)));
    NvhbiStopFlag stop;
    nvhbi_stop_flag_create(stop);

    // The reader grid needs one warp per chunk it will cycle through, so pick
    // lines_mult from the footprint instead of forcing the writer's layout.
    for (int wi = 0; wi < w_n; ++wi) {
    for (int ri = 0; ri < r_n; ++ri) {
        const unsigned int w_sms = w_list[wi];
        const unsigned int r_sms = r_list[ri];
        if (!w_sms && !r_sms) continue;

        const unsigned int w_chunks = w_sms ? nvhbi_chunks_used(w_sms, nbps, block, 1u) : 0u;

        for (unsigned int rep = 0; rep < repeat; ++rep) {
            nvhbi_flush_l2(t);
            if (w_chunks)
                nvhbi_warm_chunks<<<t.sm_count * 8, 128>>>(
                    t.d_data, d_wlist, 0u, w_chunks, t.d_sm_side, w_target_die, t.d_sink);
            // No read warm-up: the sweep is far bigger than L2, so the source
            // is served from the source die's HBM by construction.
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemset(d_wprog, 0, sizeof(unsigned long long)));
            CHECK_CUDA(cudaMemset(d_rprog, 0, sizeof(unsigned long long)));
            nvhbi_stop_flag_reset(stop);
            const unsigned long long dl =
                (unsigned long long)(4u * window_ms + 2000u) * (unsigned long long)t.clock_khz;

            if (w_sms)
                nvhbi_stress_write<<<t.sm_count * nbps, block, 0, sw>>>(
                    t.d_data, t.d_far_idx, t.d_near_idx, t.d_sm_side,
                    wp, /*own_die=*/0u, w_sms, nbps, (unsigned int)t.sm_count,
                    1u, 0u, 0u, dl, stop.d, d_wprog, nullptr, t.d_sink);
            if (r_sms)
                nvhbi_stream_read<<<t.sm_count * nbps, block, 0, sr>>>(
                    t.d_data, d_rlist, r_chunks, t.d_sm_side,
                    rp, r_sms, nbps, (unsigned int)t.sm_count,
                    dl, stop.d, d_rprog, t.d_sink);
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
            CHECK_CUDA(cudaStreamSynchronize(sw));
            CHECK_CUDA(cudaStreamSynchronize(sr));

            const double wg = (double)(w1 - w0) * 32.0 / (wall * 1e-3) / 1e9;
            const double rg = (double)(r1 - r0) * 32.0 / (wall * 1e-3) / 1e9;
            // Both payloads travel die A -> die B, so for a non-local read the
            // sum is what that one direction carried.
            printf("CFG,%u,%u,%u,%u,%.2f,%.2f,%.2f\n",
                   w_sms, r_sms, r_local, rep, wg, rg,
                   wg + (r_local ? 0.0 : rg));
            fflush(stdout);
        }
    }}

    CHECK_CUDA(cudaStreamDestroy(sw));
    CHECK_CUDA(cudaStreamDestroy(sr));
    CHECK_CUDA(cudaFree(d_wprog));
    CHECK_CUDA(cudaFree(d_rprog));
    nvhbi_stop_flag_destroy(stop);
    nvhbi_free(t);
    return 0;
}
