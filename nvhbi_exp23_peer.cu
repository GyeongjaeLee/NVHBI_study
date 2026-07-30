// nvhbi_exp23_peer.cu -- EXPERIMENTS 2 and 3
//
// Two GPUs over NVLink. We do NOT assume which die NVLink attaches to -- we
// measure it (a CALIBRATION phase), then orient everything around the result.
//
//   NEAR die  A = the die GPU1 reaches over NVLink without an NV-HBI hop
//   FAR  die  B = the other die (one extra NV-HBI hop from GPU1)
//
//   exp2  peer -> B (far)  : peer traffic crosses NV-HBI, in the SAME direction
//                            as the background load
//   exp3  peer -> A (near) : same NVLink load, no NV-HBI crossing: the CONTROL
//
// The background load is oriented A -> B (die-A SMs writing into die-B memory),
// so that exp2's peer traffic (which enters at A and crosses to B) shares the
// exact same bisection link and direction as the background. This is what makes
// exp2 vs exp3 an isolation of the NV-HBI hop rather than of NVLink.
//
// How the probe bandwidth is measured (there is no ncu here)
// ----------------------------------------------------------
// ncu serializes+replays profiled kernels and would destroy the concurrency, and
// no hardware counter reports NV-HBI traffic live. So every writing kernel keeps
// an in-kernel counter of the 4B stores it issued; the host reads it before and
// after a fixed time window and divides by wall time. When the fabric is the
// bottleneck the store-issue rate equals the fabric drain rate, so:
//
//     achieved bandwidth = issued_stores * 32B / window
//
// The "store == one 32B remote sector" identity is the same one exp1 calibrates
// against ncu (its sector_ratio column). The BACKGROUND bandwidth is the probe
// of remaining bisection; the peer load is the independent variable.
//
// PHYSICS CAVEAT worth stating up front: NV-HBI (die-to-die) bandwidth is far
// larger than a single GPU's NVLink budget. One peer GPU alone cannot dent the
// bisection; the effect only appears once the on-GPU0 background has driven
// NV-HBI near its ceiling, and even then the peer can take at most ~NVLink worth
// of it. Push bg_sms to the maximum for exp2/2c to show anything.
//
// argv: [exp 2|3]   (2 -> peer targets far, 3 -> peer targets near)
//
// Env: NVHBI_FAR_DIE            override auto-detection (0 or 1)
//      NVHBI_BG_SMS_LIST        background SM counts, 0=off. default "0,16,32,64,74"
//      NVHBI_PEER_BLOCKS_LIST   peer grid sizes, 0=off.      default "0,256,1024,4096,16384"
//      NVHBI_BG_LOCAL           1 = background stays on die A (control), default 0
//      NVHBI_WINDOW_MS          measurement window per point, default 200
//      NVHBI_REPEAT             reps per point,               default 3
//      NVHBI_PEER_CHUNKS        peer footprint in 4KiB chunks,default 4096 (16MB)
//      NVHBI_BG_BLOCK / NVHBI_BG_BLOCKS_PER_SM / NVHBI_BG_LINES
//      NVHBI_MEMCPY             1 = also time cudaMemcpyPeer, default 0
//      NVHBI_BUF_MULT           default 8

#include "nvhbi_common.cuh"
#include <chrono>

static unsigned int env_u(const char* k, unsigned int d) {
    const char* s = getenv(k);
    return (s && *s) ? (unsigned int)atoi(s) : d;
}
static int has_env(const char* k) { const char* s = getenv(k); return s && *s; }

static int parse_list(const char* env, const char* dflt, unsigned int* out, int cap) {
    const char* s = getenv(env);
    if (!s || !*s) s = dflt;
    int n = 0;
    char buf[512];
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
    const unsigned int exp = (argc > 1) ? (unsigned int)atoi(argv[1]) : 2u;
    if (exp != 2u && exp != 3u) { fprintf(stderr, "usage: %s [2|3]\n", argv[0]); return 1; }

    const unsigned int peer_chunks_req = env_u("NVHBI_PEER_CHUNKS", 4096u);
    const unsigned int bg_block  = env_u("NVHBI_BG_BLOCK", 64u);
    const unsigned int bg_nbps   = env_u("NVHBI_BG_BLOCKS_PER_SM", 32u);
    const unsigned int bg_lines  = env_u("NVHBI_BG_LINES", 1u);
    const unsigned int bg_local  = env_u("NVHBI_BG_LOCAL", 0u);
    const unsigned int window_ms = env_u("NVHBI_WINDOW_MS", 200u);
    const unsigned int repeat    = env_u("NVHBI_REPEAT", 3u);
    const unsigned int do_memcpy = env_u("NVHBI_MEMCPY", 0u);
    const double       buf_mult  = (double)env_u("NVHBI_BUF_MULT", 8u);

    unsigned int bg_list[32], pk_list[32];
    const int bg_n = parse_list("NVHBI_BG_SMS_LIST", "0,16,32,64,74", bg_list, 32);
    const int pk_n = parse_list("NVHBI_PEER_BLOCKS_LIST", "0,16,64,256,1024", pk_list, 32);

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

    /* -------- GPU1-side copies of BOTH chunk lists (for calibration+peer) -------- */
    unsigned int* d_near1 = nullptr;   // die-0 chunk offsets, on GPU1
    unsigned int* d_far1  = nullptr;   // die-1 chunk offsets, on GPU1
    CHECK_CUDA(cudaMalloc(&d_near1, t.near_count * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_far1,  t.far_count  * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemcpy(d_near1, t.h_near_idx, t.near_count * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_far1,  t.h_far_idx,  t.far_count  * sizeof(unsigned int), cudaMemcpyHostToDevice));
    unsigned int* d_sink1 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_sink1, 256 * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_sink1, 0, 256 * sizeof(unsigned int)));
    unsigned int* d_lat = nullptr;
    CHECK_CUDA(cudaMalloc(&d_lat, sizeof(unsigned int)));
    unsigned long long* d_peer_prog = nullptr;
    CHECK_CUDA(cudaMalloc(&d_peer_prog, sizeof(unsigned long long)));

    cudaStream_t s_peer;
    CHECK_CUDA(cudaStreamCreateWithFlags(&s_peer, cudaStreamNonBlocking));
    cudaEvent_t pe0, pe1;
    CHECK_CUDA(cudaEventCreate(&pe0));
    CHECK_CUDA(cudaEventCreate(&pe1));

    auto die_list1 = [&](unsigned int die) { return (die == 1u) ? d_far1 : d_near1; };
    auto die_count = [&](unsigned int die) { return (die == 1u) ? t.far_count : t.near_count; };

    /* =====================================================================
       CALIBRATION 1: peer atomic latency to each die (attachment probe)

       Times an ATOMIC on a small resident working set, repeated, after a warm-up
       pass, and takes the minimum -- see nvhbi_peer_latency. The dies are probed
       in ALTERNATING order across several rounds so that first-touch cost cannot
       land on whichever die happens to go first. The earlier cold single-pass
       version reported a 3.9x gap (13429 vs 3465 cycles) that was mostly
       measurement order: an NV-HBI hop costs ~300 cycles, not ~10000.
       ===================================================================== */
    unsigned int* d_lat2 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_lat2, sizeof(unsigned int)));

    // A GPU1-LOCAL reference, measured with the identical kernel. It is the
    // sanity check on the timer itself: a local atomic must land in the few
    // hundred cycles an L2 round trip costs. If this reads tens of cycles the
    // timing is broken, not the fabric, and the peer numbers mean nothing.
    unsigned int* d_local_buf = nullptr;
    unsigned int* d_local_idx = nullptr;
    {
        const unsigned int nloc = 64u;
        CHECK_CUDA(cudaMalloc(&d_local_buf, (size_t)nloc * NVHBI_CHUNK_BYTES));
        CHECK_CUDA(cudaMemset(d_local_buf, 0x5a, (size_t)nloc * NVHBI_CHUNK_BYTES));
        unsigned int* h = (unsigned int*)malloc(nloc * sizeof(unsigned int));
        for (unsigned int i = 0; i < nloc; ++i) h[i] = i * NVHBI_CHUNK_INTS;
        CHECK_CUDA(cudaMalloc(&d_local_idx, nloc * sizeof(unsigned int)));
        CHECK_CUDA(cudaMemcpy(d_local_idx, h, nloc * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        free(h);
    }
    const unsigned int lat_chunks = env_u("NVHBI_LAT_CHUNKS", 16u);
    const unsigned int lat_reps   = env_u("NVHBI_LAT_REPS", 200u);
    const unsigned int lat_warm   = env_u("NVHBI_LAT_WARMUP", 64u);
    const unsigned int lat_rounds = env_u("NVHBI_LAT_ROUNDS", 4u);

    auto peer_latency_once = [&](unsigned int die, unsigned int* mn, unsigned int* me) {
        const unsigned int nc = (die_count(die) < lat_chunks) ? die_count(die) : lat_chunks;
        nvhbi_peer_latency<<<1, 1>>>(t.d_data, die_list1(die), 0u, nc,
                                     lat_reps, lat_warm, d_lat, d_lat2, d_sink1);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(mn, d_lat,  sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(me, d_lat2, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    };

    unsigned int lat_loc = ~0u, mean_loc = 0u;
    {
        unsigned int mn, me;
        for (unsigned int r = 0; r < 2u; ++r) {
            nvhbi_peer_latency<<<1, 1>>>(d_local_buf, d_local_idx, 0u, lat_chunks,
                                         lat_reps, lat_warm, d_lat, d_lat2, d_sink1);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(&mn, d_lat,  sizeof(mn), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(&me, d_lat2, sizeof(me), cudaMemcpyDeviceToHost));
            if (mn < lat_loc) lat_loc = mn;
            mean_loc = me;
        }
    }

    unsigned int lat0 = ~0u, lat1 = ~0u, mean0 = 0u, mean1 = 0u;
    {
        unsigned int mn, me;
        peer_latency_once(0u, &mn, &me);   // discarded: pays peer-mapping setup
        peer_latency_once(1u, &mn, &me);
        for (unsigned int round = 0; round < lat_rounds; ++round) {
            // alternate which die goes first
            const unsigned int a0 = (round & 1u) ? 1u : 0u;
            const unsigned int a1 = 1u - a0;
            peer_latency_once(a0, &mn, &me);
            if (a0 == 0u) { if (mn < lat0) lat0 = mn; mean0 = me; }
            else          { if (mn < lat1) lat1 = mn; mean1 = me; }
            peer_latency_once(a1, &mn, &me);
            if (a1 == 0u) { if (mn < lat0) lat0 = mn; mean0 = me; }
            else          { if (mn < lat1) lat1 = mn; mean1 = me; }
        }
    }

    /* =====================================================================
       CALIBRATION 2: uncontended peer WRITE bandwidth to each die
       (with NV-HBI >> NVLink, both dies usually pin to ~NVLink BW, so this is
        mainly a sanity number; latency is the real attachment discriminator.)
       ===================================================================== */
    auto peer_bw = [&](unsigned int die) -> double {
        CHECK_CUDA(cudaSetDevice(0));
        nvhbi_flush_l2(t);
        const unsigned int cnt = (die_count(die) < peer_chunks_req) ? die_count(die) : peer_chunks_req;
        nvhbi_warm_chunks<<<t.sm_count * 8, 128>>>(t.d_data, (die==1u)?t.d_far_idx:t.d_near_idx,
                                                   0u, cnt, t.d_sm_side, die, t.d_sink);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaMemset(d_peer_prog, 0, sizeof(unsigned long long)));
        const unsigned long long dl = (unsigned long long)window_ms * (unsigned long long)prop1.clockRate;
        CHECK_CUDA(cudaEventRecord(pe0, s_peer));
        nvhbi_peer_write<<<t.sm_count * 32, 128, 0, s_peer>>>(
            t.d_data, die_list1(die), 0u, cnt, 0u, dl, d_peer_prog, d_sink1);
        CHECK_CUDA(cudaEventRecord(pe1, s_peer));
        CHECK_CUDA(cudaStreamSynchronize(s_peer));
        float ms = 0.f; CHECK_CUDA(cudaEventElapsedTime(&ms, pe0, pe1));
        unsigned long long st = 0; CHECK_CUDA(cudaMemcpy(&st, d_peer_prog, sizeof(st), cudaMemcpyDeviceToHost));
        return (ms > 0.f) ? (double)st * 32.0 / (ms * 1e-3) / 1e9 : 0.0;
    };
    const double bw0 = peer_bw(0u);
    const double bw1 = peer_bw(1u);

    /* -------- decide NEAR (A) / FAR (B) -------- */
    unsigned int far_die;
    const char* how;
    if (has_env("NVHBI_FAR_DIE")) {
        far_die = env_u("NVHBI_FAR_DIE", 1u) ? 1u : 0u;
        how = "forced by NVHBI_FAR_DIE";
    } else {
        // Far die = higher minimum peer atomic latency (the extra NV-HBI hop).
        far_die = (lat1 > lat0) ? 1u : 0u;
        how = "auto: higher min peer atomic latency";
    }
    const unsigned int near_die = 1u - far_die;

    printf("\n================ CALIBRATION ================\n");
    printf("peer atomic latency (min over %u chunks x %u reps x %u rounds)\n",
           lat_chunks, lat_reps, lat_rounds);
    printf("  GPU1 local (reference): min=%u cyc  mean=%u cyc\n", lat_loc, mean_loc);
    printf("  die0: min=%u cyc  mean=%u cyc\n", lat0, mean0);
    printf("  die1: min=%u cyc  mean=%u cyc\n", lat1, mean1);
    printf("  gap : %d cyc\n", (int)lat1 - (int)lat0);
    printf("peer WRITE bw      : die0=%.1f GB/s, die1=%.1f GB/s\n", bw0, bw1);
    printf("=> NEAR die (NVLink-attached) = die%u\n", near_die);
    printf("   FAR  die (extra NV-HBI hop) = die%u   [%s]\n", far_die, how);
    {
        if (lat_loc < 100u)
            printf("   *** WARNING: the GPU1-local reference is only %u cycles. A local\n"
                   "       atomic cannot be that fast -- the TIMER is broken, so every\n"
                   "       latency above is meaningless. ***\n", lat_loc);
        else if (lat0 < lat_loc || lat1 < lat_loc)
            printf("   *** WARNING: a peer latency came in below the local reference\n"
                   "       (%u cyc). Peer access cannot beat local; suspect the timer. ***\n",
                   lat_loc);
        const int gap = (int)((lat0 > lat1) ? lat0 - lat1 : lat1 - lat0);
        if (gap < 50)
            printf("   *** WARNING: gap is only %d cycles. The two dies are not\n"
                   "       distinguishable, so NVLink is probably not die-attached and\n"
                   "       exp2 vs exp3 does not isolate an NV-HBI hop. ***\n", gap);
        else if (gap > 3000)
            printf("   *** WARNING: gap is %d cycles, far more than the ~300 an NV-HBI\n"
                   "       hop should cost. Suspect a measurement artifact (cold lines,\n"
                   "       TLB, or ordering) rather than topology -- raise\n"
                   "       NVHBI_LAT_WARMUP/NVHBI_LAT_REPS and re-check. ***\n", gap);
        else
            printf("   (gap is in the range an extra die hop should cost)\n");
    }
    printf("=============================================\n\n");

    /* -------- roles resolved -------- */
    const unsigned int peer_die = (exp == 2u) ? far_die : near_die;   // where GPU1 writes
    // Background is oriented A->B: writer SMs on die A (near), targeting die B (far).
    // bg_local flips the target to A (own die) as the control.
    const unsigned int bg_writer_partition = near_die;
    const unsigned int bg_target_die       = bg_local ? near_die : far_die;

    /* -------- background SM clamp + region reservation -------- */
    const unsigned int bg_sms_max = (near_die == 0u) ? t.sms_p0 : t.sms_p1;
    unsigned int bg_max = 0;
    for (int i = 0; i < bg_n; ++i) {
        if (bg_list[i] > bg_sms_max) bg_list[i] = bg_sms_max;
        if (bg_list[i] > bg_max) bg_max = bg_list[i];
    }
    const unsigned int bg_reserve = nvhbi_chunks_used(bg_max, bg_nbps, bg_block, bg_lines);
    if (bg_reserve > die_count(bg_target_die)) {
        fprintf(stderr, "ERROR: background needs %u die-%u chunks, have %u\n",
                bg_reserve, bg_target_die, die_count(bg_target_die));
        return 1;
    }

    const unsigned int  peer_avail = die_count(peer_die);
    const unsigned int* d_peer_own = (peer_die == 1u) ? t.d_far_idx : t.d_near_idx;
    const unsigned int  peer_first = (peer_die == bg_target_die) ? bg_reserve : 0u;
    unsigned int peer_chunks = peer_chunks_req;
    if (peer_first + peer_chunks > peer_avail)
        peer_chunks = (peer_avail > peer_first) ? (peer_avail - peer_first) : 0u;
    if (peer_chunks == 0u) { fprintf(stderr, "ERROR: no room for peer region\n"); return 1; }

    printf("exp%u: GPU1 --NVLink--> GPU0 die%u (%s)\n", exp, peer_die,
           (exp == 2u) ? "FAR: crosses NV-HBI" : "NEAR: no NV-HBI hop");
    printf("  background: die%u SMs -> die%u (%s), up to %u SMs\n",
           bg_writer_partition, bg_target_die,
           bg_local ? "OWN die = control" : "cross-die = bisection probe", bg_max);
    printf("  peer region: chunks [%u,%u) on die%u -> %.1f MB\n\n",
           peer_first, peer_first + peer_chunks, peer_die,
           peer_chunks * (double)NVHBI_CHUNK_BYTES / (1024.0 * 1024.0));

    /* -------- background machinery on GPU0 -------- */
    CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t s_bg;
    CHECK_CUDA(cudaStreamCreateWithFlags(&s_bg, cudaStreamNonBlocking));
    unsigned long long* d_bg_prog = nullptr;
    CHECK_CUDA(cudaMalloc(&d_bg_prog, sizeof(unsigned long long)));
    NvhbiStopFlag stop;
    nvhbi_stop_flag_create(stop);

    const unsigned int* d_peer_idx = die_list1(peer_die);   // GPU1-side list for the target die
    const unsigned int  peer_block = 128u;

    printf("# CFG,exp,far_die,peer_die,bg_local,bg_sms,peer_blocks,rep,"
           "peer_ms,peer_GBps,bg_GBps,crossing_GBps\n");

    for (int bi = 0; bi < bg_n; ++bi) {
    for (int pi = 0; pi < pk_n; ++pi) {
        const unsigned int bg_sms      = bg_list[bi];
        const unsigned int peer_blocks = pk_list[pi];
        if (bg_sms == 0u && peer_blocks == 0u) continue;
        const unsigned int bg_chunks = bg_sms
            ? nvhbi_chunks_used(bg_sms, bg_nbps, bg_block, bg_lines) : 0u;

        for (unsigned int rep = 0; rep < repeat; ++rep) {
            /* flush + warm both regions from their owning die */
            CHECK_CUDA(cudaSetDevice(0));
            nvhbi_flush_l2(t);
            if (bg_chunks) {
                nvhbi_warm_chunks<<<t.sm_count * 8, 128>>>(
                    t.d_data, (bg_target_die==1u)?t.d_far_idx:t.d_near_idx,
                    0u, bg_chunks, t.d_sm_side, bg_target_die, t.d_sink);
            }
            if (peer_blocks) {
                nvhbi_warm_chunks<<<t.sm_count * 8, 128>>>(
                    t.d_data, d_peer_own, peer_first, peer_chunks,
                    t.d_sm_side, peer_die, t.d_sink);
            }
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            /* start background, settle */
            CHECK_CUDA(cudaMemset(d_bg_prog, 0, sizeof(unsigned long long)));
            nvhbi_stop_flag_reset(stop);
            if (bg_sms) {
                // Must outlast settle + the whole peer kernel, or the counter
                // delta gets divided by a window the background was not alive
                // for. That artifact produced the fake "bg dropped to 120 GB/s"
                // rows at peer_blocks=16384.
                const unsigned long long dl =
                    (unsigned long long)(4u * window_ms + 2000u) * (unsigned long long)t.clock_khz;
                nvhbi_stress_write<<<t.sm_count * bg_nbps, bg_block, 0, s_bg>>>(
                    t.d_data, t.d_far_idx, t.d_near_idx, t.d_sm_side,
                    bg_writer_partition, bg_local, bg_sms, bg_nbps,
                    (unsigned int)t.sm_count, bg_lines, 0u,
                    0u, dl, stop.d, d_bg_prog, nullptr, t.d_sink);
                CHECK_CUDA(cudaGetLastError());
                spin_ms(100.0);
            }

            /* measurement window */
            unsigned long long bg_p0 = 0, bg_p1 = 0, peer_stores = 0;
            CHECK_CUDA(cudaMemcpy(&bg_p0, d_bg_prog, sizeof(bg_p0), cudaMemcpyDeviceToHost));
            // Clamp the peer grid. Two ways it silently broke the sweep before:
            //  * more blocks than fit resident -> each WAVE runs the full deadline,
            //    so peer_ms became waves x window (200 -> 1402 ms at 16384 blocks)
            //    and peer_GBps fell by exactly that factor.
            //  * more warps than chunks -> the surplus warps own no chunk and just
            //    spin, adding waves without adding traffic.
            unsigned int pb = peer_blocks;
            if (pb) {
                const unsigned int wpb  = peer_block / 32u;
                const unsigned int wave = (unsigned int)prop1.multiProcessorCount
                                        * (2048u / peer_block);
                const unsigned int useful = (peer_chunks + wpb - 1u) / wpb;
                if (pb > wave)   pb = wave;
                if (pb > useful) pb = useful;
                if (pb != peer_blocks)
                    fprintf(stderr, "note: peer_blocks %u -> %u (wave cap %u, chunk cap %u)\n",
                            peer_blocks, pb, wave, useful);
            }

            float peer_ms = 0.f;
            const double wall0 = now_ms();
            if (pb) {
                CHECK_CUDA(cudaSetDevice(1));
                CHECK_CUDA(cudaMemset(d_peer_prog, 0, sizeof(unsigned long long)));
                const unsigned long long pdl =
                    (unsigned long long)window_ms * (unsigned long long)prop1.clockRate;
                CHECK_CUDA(cudaEventRecord(pe0, s_peer));
                nvhbi_peer_write<<<pb, peer_block, 0, s_peer>>>(
                    t.d_data, d_peer_idx, peer_first, peer_chunks, 0u, pdl, d_peer_prog, d_sink1);
                CHECK_CUDA(cudaEventRecord(pe1, s_peer));
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaStreamSynchronize(s_peer));
                CHECK_CUDA(cudaEventElapsedTime(&peer_ms, pe0, pe1));
                CHECK_CUDA(cudaMemcpy(&peer_stores, d_peer_prog, sizeof(peer_stores), cudaMemcpyDeviceToHost));
            } else {
                spin_ms((double)window_ms);
            }
            const double wall_ms = now_ms() - wall0;

            CHECK_CUDA(cudaSetDevice(0));
            CHECK_CUDA(cudaMemcpy(&bg_p1, d_bg_prog, sizeof(bg_p1), cudaMemcpyDeviceToHost));
            if (bg_sms) { nvhbi_stop_flag_set(stop); CHECK_CUDA(cudaStreamSynchronize(s_bg)); }

            const double bg_alive_ms = (double)(4u * window_ms + 2000u) - 100.0;
            if (bg_sms && wall_ms > bg_alive_ms)
                fprintf(stderr, "WARNING: peer ran %.0f ms but background deadline was %.0f ms "
                                "-- bg_GBps is understated\n", wall_ms, bg_alive_ms);

            const double peer_gbps = (peer_ms > 0.f)
                ? (double)peer_stores * 32.0 / (peer_ms * 1e-3) / 1e9 : 0.0;
            const double bg_gbps = (bg_sms && wall_ms > 0.0)
                ? (double)(bg_p1 - bg_p0) * 32.0 / (wall_ms * 1e-3) / 1e9 : 0.0;
            const double crossing = (bg_local ? 0.0 : bg_gbps)
                                  + ((exp == 2u) ? peer_gbps : 0.0);

            printf("CFG,%u,%u,%u,%u,%u,%u,%u,%.4f,%.2f,%.2f,%.2f\n",
                   exp, far_die, peer_die, bg_local, bg_sms, pb, rep,
                   peer_ms, peer_gbps, bg_gbps, crossing);
            fflush(stdout);
        }
    }}

    if (do_memcpy) {
        const size_t bytes = (size_t)peer_chunks * NVHBI_CHUNK_BYTES;
        unsigned int *src = nullptr, *dst = nullptr;
        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaMalloc(&src, bytes)); CHECK_CUDA(cudaMemset(src, 0x3c, bytes));
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaMalloc(&dst, bytes));
        CHECK_CUDA(cudaSetDevice(1));
        cudaEvent_t m0, m1; CHECK_CUDA(cudaEventCreate(&m0)); CHECK_CUDA(cudaEventCreate(&m1));
        CHECK_CUDA(cudaMemcpyPeerAsync(dst, 0, src, 1, bytes, s_peer));
        CHECK_CUDA(cudaStreamSynchronize(s_peer));
        CHECK_CUDA(cudaEventRecord(m0, s_peer));
        for (int i = 0; i < 10; ++i) CHECK_CUDA(cudaMemcpyPeerAsync(dst, 0, src, 1, bytes, s_peer));
        CHECK_CUDA(cudaEventRecord(m1, s_peer));
        CHECK_CUDA(cudaStreamSynchronize(s_peer));
        float mms = 0.f; CHECK_CUDA(cudaEventElapsedTime(&mms, m0, m1));
        printf("MEMCPY_PEER,%zu,%.4f,%.2f  # bytes,ms(x10),GB/s -- die-agnostic (~50/50)\n",
               bytes, mms, (double)bytes * 10.0 / (mms * 1e-3) / 1e9);
        CHECK_CUDA(cudaEventDestroy(m0)); CHECK_CUDA(cudaEventDestroy(m1));
        CHECK_CUDA(cudaFree(src)); CHECK_CUDA(cudaSetDevice(0)); CHECK_CUDA(cudaFree(dst));
    }

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaEventDestroy(pe0)); CHECK_CUDA(cudaEventDestroy(pe1));
    CHECK_CUDA(cudaStreamDestroy(s_peer));
    CHECK_CUDA(cudaFree(d_near1)); CHECK_CUDA(cudaFree(d_far1));
    CHECK_CUDA(cudaFree(d_sink1)); CHECK_CUDA(cudaFree(d_lat)); CHECK_CUDA(cudaFree(d_lat2));
    CHECK_CUDA(cudaFree(d_peer_prog));
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaStreamDestroy(s_bg)); CHECK_CUDA(cudaFree(d_bg_prog));
    nvhbi_stop_flag_destroy(stop);
    nvhbi_free(t);
    return 0;
}
