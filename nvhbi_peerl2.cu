// nvhbi_peerl2.cu -- when GPU1 reaches into GPU0 over NVLink, WHERE is it served?
//
// exp2/exp3 only ever make GPU1 write. A write tells you almost nothing about
// where the data lives, because B200's L2 is write-no-allocate: the line is
// either already there or it is not, and either way the store is fire-and-forget
// from the issuer's side. A READ has to come back from somewhere, and that
// somewhere is the question:
//
//     GPU0's SMs have just pulled a region into GPU0's L2. GPU1 now reads it.
//     Does the data come out of GPU0's L2, or does the request go all the way
//     to GPU0's HBM?
//
// The instrument is latency, not bandwidth: an L2 hit and an HBM miss differ by
// hundreds of cycles and nothing else in the path changes between the two runs.
// Bandwidth is measured too, but only as corroboration -- it saturates on
// NVLink long before the L2/HBM difference would show.
//
// WHAT IS SWEPT
// -------------
//   die     0 and 1, i.e. the NVLink-attached (NEAR) die and the one an NV-HBI
//           hop further away (FAR). Named by the same latency calibration
//           exp2/exp3 use, so the two experiments agree on which is which.
//   state   WARM  = GPU0's own SMs pulled the region into the owning die's L2
//           COLD  = GPU0's L2 flushed, so the region is only in HBM
//   cv      0 = ld.global.cg, 1 = ld.global.cv (volatile, re-fetch).
//           The GAP BETWEEN THESE TWO IS ITSELF A RESULT. If cg is much faster
//           than cv, GPU1 is keeping a local replica of GPU0's lines and the
//           "peer bandwidth" is partly not crossing NVLink at all. Every peer
//           number in this study depends on that not happening.
//   corun   0 = GPU0 idle while GPU1 reads
//           1 = GPU0's own SMs keep reading THE SAME chunks throughout
//           2 = GPU0's own SMs keep WRITING the same chunks throughout
//           This is the "both GPUs on the same data" case. corun=0 vs 1 vs 2
//           separates "GPU1 is served from a warm L2" from "GPU1 is fighting
//           GPU0 for the same lines".
//
// A GPU0-LOCAL reference latency is measured for every (die, state) with the
// identical kernel, from an SM that sits on the owning die. Without it the peer
// numbers float: 3000 cycles means nothing until you know what a local L2 hit
// and a local HBM miss cost on this allocation.
//
// TWO CACHES HAVE TO BE CONTROLLED, NOT ONE
// -----------------------------------------
// A remote read leaves a REPLICA on the requester's side, so the next access to
// that line is a local hit and crosses nothing. That is why nvhbi_dualdir's
// readers stream over a footprint many times L2 instead of re-reading a resident
// one, and it applies here twice over:
//
//   GPU0's L2  is the variable under test. Flushed before every single run, and
//              the latency probe walks each chunk exactly once, because a peer
//              read allocates in the home L2 and would warm what it is meant to
//              find cold.
//   GPU1's L2  is contamination. Without flushing it too, rep 1 leaves the whole
//              region replicated on GPU1 and reps 2-3 never leave the GPU, which
//              would flatten the warm/cold contrast into noise. Flushed at the
//              same point.
//
// The RESIDENT bandwidth point still cannot escape this with ld.global.cg -- a
// 16 MB footprint re-read for 200 ms fits in GPU1's 126 MB L2 many times over,
// so cg measures GPU1's own cache almost entirely. The cv column is the one that
// means anything there; cg is kept only so the size of the gap is on record.
//
// Env: NVHBI_PL2_LAT_CHUNKS   chunks in one latency pass, default 512
//      NVHBI_PL2_RES_CHUNKS   resident (warm) footprint in 4KiB chunks, default 4096 (16 MB)
//      NVHBI_PL2_STREAM       1 = also measure the >>L2 streaming point, default 1
//      NVHBI_PL2_BLOCKS       peer read grid, default 1024
//      NVHBI_PL2_CORUN        comma list of co-runner modes, default "0,1,2"
//      NVHBI_PL2_CO_SMS       GPU0 co-runner SMs, default 32
//      NVHBI_WINDOW_MS        default 200      NVHBI_REPEAT default 3
//      NVHBI_FAR_DIE          override the near/far labelling (0 or 1)
//      NVHBI_BUF_MULT         default 8

#include "nvhbi_common.cuh"
#include <chrono>

// Blocks per SM for the GPU0 co-runner. The kernel derives its dense warp
// numbering from this, so the launch geometry and the argument must agree.
#define PL2_CO_NBPS 8u

/* -------------------------------------------------------- GPU0 co-runner

   Keeps GPU0 touching the very chunks GPU1 is reading, from the SMs of the die
   that owns them, for as long as the window lasts. Reading keeps the lines
   resident; writing keeps them resident AND dirty, which is the case where a
   peer read cannot be answered without involving GPU0's own caches.           */
__global__ void pl2_local_touch(unsigned int* __restrict__ data,
                                const unsigned int* __restrict__ idx,
                                unsigned int first,
                                unsigned int count,
                                const unsigned int* __restrict__ sm_side,
                                unsigned int owner_partition,
                                unsigned int active_sms,
                                unsigned int nbps,
                                unsigned int sm_count,
                                unsigned int do_write,
                                unsigned long long deadline_cycles,
                                const unsigned int* __restrict__ stop_flag,
                                unsigned long long* __restrict__ progress,
                                unsigned int* __restrict__ sink) {
    const unsigned int smid = nvhbi_smid();
    if ((sm_side[smid] % 2u) != owner_partition) return;
    const unsigned int rank = sm_side[smid] / 2u;
    if (rank >= active_sms) return;
    if (count == 0u) return;

    const unsigned int lane = threadIdx.x % 32u;
    // Dense warp numbering over the PARTICIPATING warps only, the same mapping
    // nvhbi_stress_write uses. Numbering by global thread id instead would leave
    // each surviving warp walking its own residue class of the chunk list, so a
    // co-runner on 32 of 148 SMs would only ever touch a fifth of the region --
    // and "keep the region resident" is the entire point of this kernel.
    const unsigned int wpb  = (blockDim.x + 31u) / 32u;
    const unsigned int wib  = threadIdx.x / 32u;
    const unsigned int q    = blockIdx.x / sm_count;
    const unsigned int gwarp  = wib + wpb * (q + nbps * rank);
    const unsigned int nwarps = wpb * nbps * active_sms;
    const unsigned long long lanes =
        (unsigned long long)min(32u, blockDim.x - wib * 32u);

    unsigned int val = smid * 1000003u + threadIdx.x + 1u;
    unsigned int acc = 0u;
    unsigned long long done = 0ull;
    const unsigned long long t0 = clock64();

#pragma unroll 1
    for (unsigned int it = 0; ; ++it) {
#pragma unroll 1
        for (unsigned int c = gwarp; c < count; c += nwarps) {
            unsigned int* a[4];
            nvhbi_lane_addrs(data, idx[first + c], lane, a);
            if (do_write) {
                nvhbi_st(a[0], val); nvhbi_st(a[1], val);
                nvhbi_st(a[2], val); nvhbi_st(a[3], val);
                ++val;
            } else {
                acc += nvhbi_ld(a[0]); acc += nvhbi_ld(a[1]);
                acc += nvhbi_ld(a[2]); acc += nvhbi_ld(a[3]);
            }
            done += 4ull;
        }
        if (progress && lane == 0u) { atomicAdd(progress, done * lanes); done = 0ull; }
        if ((unsigned long long)(clock64() - t0) > deadline_cycles) break;
        if (stop_flag && *(volatile const unsigned int*)stop_flag) break;
    }
    if (progress && lane == 0u && done) atomicAdd(progress, done * lanes);
    nvhbi_st(&sink[smid], acc + val);
}

/* ------------------------------------------------------------------ host */

static unsigned int env_u(const char* k, unsigned int d) {
    const char* s = getenv(k);
    return (s && *s) ? (unsigned int)atoi(s) : d;
}
static int has_env(const char* k) { const char* s = getenv(k); return s && *s; }

static int parse_list(const char* env, const char* dflt, unsigned int* out, int cap) {
    const char* s = getenv(env);
    if (!s || !*s) s = dflt;
    int n = 0; char buf[256];
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

int main() {
    const unsigned int lat_chunks = env_u("NVHBI_PL2_LAT_CHUNKS", 512u);
    const unsigned int res_chunks = env_u("NVHBI_PL2_RES_CHUNKS", 4096u);
    const unsigned int do_stream  = env_u("NVHBI_PL2_STREAM", 1u);
    const unsigned int pr_blocks  = env_u("NVHBI_PL2_BLOCKS", 1024u);
    const unsigned int co_sms     = env_u("NVHBI_PL2_CO_SMS", 32u);
    const unsigned int window_ms  = env_u("NVHBI_WINDOW_MS", 200u);
    const unsigned int repeat     = env_u("NVHBI_REPEAT", 3u);
    const double       buf_mult   = (double)env_u("NVHBI_BUF_MULT", 8u);

    unsigned int co_list[8];
    const int co_n = parse_list("NVHBI_PL2_CORUN", "0,1,2", co_list, 8);

    int ndev = 0;
    CHECK_CUDA(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { fprintf(stderr, "ERROR: need 2 GPUs, found %d\n", ndev); return 1; }

    NvhbiTopo t;
    nvhbi_probe(t, 0, buf_mult);

    int can = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can, 1, 0));
    if (!can) { fprintf(stderr, "ERROR: GPU1 cannot peer-access GPU0\n"); return 1; }
    CHECK_CUDA(cudaSetDevice(1));
    cudaError_t pe = cudaDeviceEnablePeerAccess(0, 0);
    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) CHECK_CUDA(pe);
    cudaDeviceProp prop1{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop1, 1));
    int clk1 = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&clk1, cudaDevAttrClockRate, 1));

    /* -------- GPU1-side copies of both chunk lists -------- */
    unsigned int *d_near1 = nullptr, *d_far1 = nullptr, *d_sink1 = nullptr;
    unsigned int *d_cyc1 = nullptr;
    unsigned long long* d_prog1 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_near1, t.near_count * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_far1,  t.far_count  * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemcpy(d_near1, t.h_near_idx, t.near_count * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_far1,  t.h_far_idx,  t.far_count  * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_sink1, 256 * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_sink1, 0, 256 * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_cyc1, sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_prog1, sizeof(unsigned long long)));
    // GPU1's own flush buffer. See the header comment: the replica a remote read
    // leaves on this side is what would otherwise serve reps 2 and 3.
    unsigned int* d_flush1 = nullptr;
    const size_t flush1_ints = ((size_t)prop1.l2CacheSize * 2) / sizeof(unsigned int);
    CHECK_CUDA(cudaMalloc(&d_flush1, flush1_ints * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_flush1, 0x01, flush1_ints * sizeof(unsigned int)));
    cudaStream_t s_peer;
    CHECK_CUDA(cudaStreamCreateWithFlags(&s_peer, cudaStreamNonBlocking));
    cudaEvent_t pe0, pe1;
    CHECK_CUDA(cudaEventCreate(&pe0));
    CHECK_CUDA(cudaEventCreate(&pe1));

    /* -------- GPU0-side machinery -------- */
    CHECK_CUDA(cudaSetDevice(0));
    unsigned int* d_cyc0 = nullptr;
    unsigned long long* d_prog0 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_cyc0, sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_prog0, sizeof(unsigned long long)));
    cudaStream_t s_co;
    CHECK_CUDA(cudaStreamCreateWithFlags(&s_co, cudaStreamNonBlocking));
    NvhbiStopFlag stop;
    nvhbi_stop_flag_create(stop);

    // One GPU0 SM per die, for the local reference probe. The probe pins itself
    // to a named smid, and a reference taken from the wrong die would measure an
    // NV-HBI hop instead of a local hit.
    unsigned int probe_sm[2] = { 0u, 0u };
    {
        unsigned int* h_side = (unsigned int*)malloc(t.sm_count * sizeof(unsigned int));
        CHECK_CUDA(cudaMemcpy(h_side, t.d_sm_side, t.sm_count * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        bool got[2] = { false, false };
        for (int i = 0; i < t.sm_count; ++i) {
            const unsigned int p = h_side[i] % 2u;
            if (!got[p]) { probe_sm[p] = (unsigned int)i; got[p] = true; }
        }
        free(h_side);
    }

    auto die_list1  = [&](unsigned int die) { return (die == 1u) ? d_far1 : d_near1; };
    auto die_list0  = [&](unsigned int die) { return (die == 1u) ? t.d_far_idx : t.d_near_idx; };
    auto die_count  = [&](unsigned int die) { return (die == 1u) ? t.far_count : t.near_count; };

    /* ============ which die is NVLink-attached? (same probe as exp2/3) ============
       Reuses nvhbi_peer_latency: a warmed atomic chain, minimum over rounds, dies
       alternating so first-touch cost cannot land on whichever went first.      */
    unsigned int far_die = 1u;
    {
        // On GPU1: the probe kernel runs there, and an output buffer left on
        // GPU0 would have every result store cross NVLink.
        CHECK_CUDA(cudaSetDevice(1));
        unsigned int* d_lat  = nullptr; CHECK_CUDA(cudaMalloc(&d_lat,  sizeof(unsigned int)));
        unsigned int* d_lat2 = nullptr; CHECK_CUDA(cudaMalloc(&d_lat2, sizeof(unsigned int)));
        unsigned int best[2] = { ~0u, ~0u };
        auto probe = [&](unsigned int die) {
            unsigned int mn = 0, me = 0;
            const unsigned int nc = (die_count(die) < 16u) ? die_count(die) : 16u;
            nvhbi_peer_latency<<<prop1.multiProcessorCount, 1>>>(
                t.d_data, die_list1(die), 0u, nc, 200u, 64u, 0u, d_lat, d_lat2, d_sink1);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(&mn, d_lat,  sizeof(mn), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(&me, d_lat2, sizeof(me), cudaMemcpyDeviceToHost));
            (void)me;
            return mn;
        };
        probe(0u); probe(1u);                       // discard: pays peer-mapping setup
        for (unsigned int r = 0; r < 4u; ++r) {
            const unsigned int a0 = (r & 1u) ? 1u : 0u;
            unsigned int v = probe(a0);      if (v < best[a0]) best[a0] = v;
            v = probe(1u - a0);              if (v < best[1u - a0]) best[1u - a0] = v;
        }
        far_die = has_env("NVHBI_FAR_DIE") ? (env_u("NVHBI_FAR_DIE", 1u) ? 1u : 0u)
                                           : ((best[1] > best[0]) ? 1u : 0u);
        printf("\npeer atomic latency: die0=%u cyc  die1=%u cyc  =>  NEAR=die%u  FAR=die%u%s\n",
               best[0], best[1], 1u - far_die, far_die,
               has_env("NVHBI_FAR_DIE") ? "  [forced]" : "");
        CHECK_CUDA(cudaFree(d_lat)); CHECK_CUDA(cudaFree(d_lat2));
    }

    printf("GPU0 local probe SMs: die0 -> SM%u, die1 -> SM%u\n", probe_sm[0], probe_sm[1]);
    printf("resident footprint %u chunks (%.1f MB) vs %.1f MB L2 total (%.1f MB per die)\n",
           res_chunks, res_chunks * 4096.0 / 1048576.0,
           t.l2_bytes / 1048576.0, t.l2_bytes / 2097152.0);
    if ((double)res_chunks * 4096.0 > t.l2_bytes / 2.0)
        fprintf(stderr, "WARNING: the resident footprint does not fit in one die's L2; "
                        "the WARM rows will not be pure L2 hits.\n");
    printf("\n");

    /* ------------------------------------------------------------ helpers */

    // One clamp for every use: both probes index the same list, and a per-die
    // clamp applied only in the print loop would leave the kernels reading past
    // the end of the shorter die's list.
    const unsigned int lat_n = (lat_chunks < t.near_count)
                             ? ((lat_chunks < t.far_count) ? lat_chunks : t.far_count)
                             : ((t.near_count < t.far_count) ? t.near_count : t.far_count);
    if (lat_n < lat_chunks)
        fprintf(stderr, "note: latency pass shortened to %u chunks (a die has only that many)\n",
                lat_n);

    // Put BOTH caches in the state we are about to measure: GPU0's is the
    // variable, GPU1's is contamination that has to be cleared every time.
    auto set_state = [&](unsigned int die, unsigned int chunks, bool warm) {
        CHECK_CUDA(cudaSetDevice(1));
        nvhbi_flush_kernel<<<prop1.multiProcessorCount * 32, 256>>>(
            d_flush1, flush1_ints, d_sink1);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaSetDevice(0));
        nvhbi_flush_l2(t);
        if (warm) {
            nvhbi_warm(t, die_list0(die), 0u, chunks, die);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    };

    auto lat_local = [&](unsigned int die, bool warm, unsigned int cv) {
        set_state(die, lat_n, warm);
        CHECK_CUDA(cudaSetDevice(0));
        nvhbi_peer_load_latency<<<t.sm_count, 1>>>(
            t.d_data, die_list0(die), 0u, lat_n, probe_sm[die], cv, d_cyc0, t.d_sink);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        unsigned int c = 0;
        CHECK_CUDA(cudaMemcpy(&c, d_cyc0, sizeof(c), cudaMemcpyDeviceToHost));
        return c;
    };

    auto lat_peer = [&](unsigned int die, bool warm, unsigned int cv) {
        set_state(die, lat_n, warm);
        CHECK_CUDA(cudaSetDevice(1));
        nvhbi_peer_load_latency<<<prop1.multiProcessorCount, 1>>>(
            t.d_data, die_list1(die), 0u, lat_n, 0u, cv, d_cyc1, d_sink1);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        unsigned int c = 0;
        CHECK_CUDA(cudaMemcpy(&c, d_cyc1, sizeof(c), cudaMemcpyDeviceToHost));
        return c;
    };

    printf("# CFG,kind,die,is_far,state,cv,corun,rep,peer_cyc,local_cyc,peer_GBps,co_GBps,chunks\n");

    /* ==================== PHASE 1: latency ==================== */
    // The measurement this program exists for. Read it as:
    //   peer WARM  ~= peer COLD                -> GPU1 is NOT served by GPU0's L2
    //   peer WARM  <  peer COLD by ~the same
    //     margin as local WARM vs local COLD   -> GPU1 IS served by GPU0's L2
    for (unsigned int die = 0; die < 2u; ++die) {
        for (unsigned int cv = 0; cv < 2u; ++cv) {
            for (unsigned int w = 0; w < 2u; ++w) {
                const bool warm = (w == 1u);
                for (unsigned int rep = 0; rep < repeat; ++rep) {
                    const unsigned int lp = lat_peer(die, warm, cv);
                    const unsigned int ll = lat_local(die, warm, cv);
                    printf("CFG,0,%u,%u,%u,%u,0,%u,%u,%u,0.00,0.00,%u\n",
                           die, (die == far_die) ? 1u : 0u, warm ? 1u : 0u, cv, rep,
                           lp, ll, lat_n);
                    fflush(stdout);
                }
            }
        }
    }

    /* ==================== PHASE 2: bandwidth ==================== */
    // Corroboration only. NVLink saturates well below any L2-vs-HBM difference,
    // so a flat result here does NOT contradict phase 1 -- but if the STREAMING
    // point (footprint >> L2) is much slower than the RESIDENT one, that IS the
    // L2/HBM difference showing through at the bandwidth level.
    for (unsigned int die = 0; die < 2u; ++die) {
        const unsigned int avail = die_count(die);
        const unsigned int res_n = (avail < res_chunks) ? avail : res_chunks;

        struct { const char* name; unsigned int chunks; bool warm; } pts[2] = {
            { "resident", res_n, true },
            { "stream",   avail, false },
        };
        const int npts = do_stream ? 2 : 1;

        for (int p = 0; p < npts; ++p) {
        for (unsigned int cv = 0; cv < 2u; ++cv) {
        for (int ci = 0; ci < co_n; ++ci) {
            const unsigned int corun = co_list[ci];
            // A co-runner on a >>L2 streaming footprint is not the question this
            // asks (it just thrashes); keep it to the resident point.
            if (corun && p != 0) continue;

            for (unsigned int rep = 0; rep < repeat; ++rep) {
                set_state(die, pts[p].chunks, pts[p].warm);

                CHECK_CUDA(cudaSetDevice(0));
                CHECK_CUDA(cudaMemset(d_prog0, 0, sizeof(unsigned long long)));
                nvhbi_stop_flag_reset(stop);
                if (corun) {
                    const unsigned long long dl =
                        (unsigned long long)(4u * window_ms + 2000u) * (unsigned long long)t.clock_khz;
                    pl2_local_touch<<<t.sm_count * PL2_CO_NBPS, 128, 0, s_co>>>(
                        t.d_data, die_list0(die), 0u, pts[p].chunks,
                        t.d_sm_side, die, co_sms, PL2_CO_NBPS,
                        (unsigned int)t.sm_count, (corun == 2u) ? 1u : 0u,
                        dl, stop.d, d_prog0, t.d_sink);
                    CHECK_CUDA(cudaGetLastError());
                    spin_ms(50.0);
                }

                CHECK_CUDA(cudaSetDevice(1));
                CHECK_CUDA(cudaMemset(d_prog1, 0, sizeof(unsigned long long)));
                unsigned long long co0 = 0, co1 = 0;
                CHECK_CUDA(cudaSetDevice(0));
                CHECK_CUDA(cudaMemcpy(&co0, d_prog0, sizeof(co0), cudaMemcpyDeviceToHost));

                CHECK_CUDA(cudaSetDevice(1));
                const unsigned long long pdl =
                    (unsigned long long)window_ms * (unsigned long long)clk1;
                const double wall0 = now_ms();
                CHECK_CUDA(cudaEventRecord(pe0, s_peer));
                nvhbi_peer_read<<<pr_blocks, 128, 0, s_peer>>>(
                    t.d_data, die_list1(die), 0u, pts[p].chunks, pdl, 0u, cv,
                    d_prog1, d_sink1);
                CHECK_CUDA(cudaEventRecord(pe1, s_peer));
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaStreamSynchronize(s_peer));
                float pms = 0.f;
                CHECK_CUDA(cudaEventElapsedTime(&pms, pe0, pe1));
                const double wall = now_ms() - wall0;
                unsigned long long sectors = 0;
                CHECK_CUDA(cudaMemcpy(&sectors, d_prog1, sizeof(sectors), cudaMemcpyDeviceToHost));

                CHECK_CUDA(cudaSetDevice(0));
                CHECK_CUDA(cudaMemcpy(&co1, d_prog0, sizeof(co1), cudaMemcpyDeviceToHost));
                if (corun) {
                    nvhbi_stop_flag_set(stop);
                    CHECK_CUDA(cudaStreamSynchronize(s_co));
                }

                const double pg = (pms > 0.f) ? (double)sectors * 32.0 / (pms * 1e-3) / 1e9 : 0.0;
                const double cg = (corun && wall > 0.0)
                    ? (double)(co1 - co0) * 32.0 / (wall * 1e-3) / 1e9 : 0.0;

                printf("CFG,1,%u,%u,%u,%u,%u,%u,0,0,%.2f,%.2f,%u\n",
                       die, (die == far_die) ? 1u : 0u, pts[p].warm ? 1u : 0u,
                       cv, corun, rep, pg, cg, pts[p].chunks);
                fflush(stdout);
            }
        }}}
    }

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaEventDestroy(pe0)); CHECK_CUDA(cudaEventDestroy(pe1));
    CHECK_CUDA(cudaStreamDestroy(s_peer));
    CHECK_CUDA(cudaFree(d_near1)); CHECK_CUDA(cudaFree(d_far1));
    CHECK_CUDA(cudaFree(d_sink1)); CHECK_CUDA(cudaFree(d_cyc1));
    CHECK_CUDA(cudaFree(d_prog1)); CHECK_CUDA(cudaFree(d_flush1));
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaStreamDestroy(s_co));
    CHECK_CUDA(cudaFree(d_cyc0)); CHECK_CUDA(cudaFree(d_prog0));
    nvhbi_stop_flag_destroy(stop);
    nvhbi_free(t);
    return 0;
}
