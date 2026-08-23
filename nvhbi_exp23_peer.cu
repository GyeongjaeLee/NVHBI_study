// nvhbi_exp23_peer.cu -- EXPERIMENTS 2 and 3
//
// Does NVLink traffic entering GPU0 contend with GPU0's own cross-die traffic
// on NV-HBI?
//
//   NEAR die A = the die GPU1 reaches over NVLink without an NV-HBI hop
//   FAR  die B = the other die (one extra NV-HBI hop from GPU1)
//
//   exp2  peer -> B (far)  : peer traffic crosses NV-HBI, in the SAME direction
//                            as the background
//   exp3  peer -> A (near) : same NVLink load, no NV-HBI crossing: the CONTROL
//
// Which die is which is MEASURED, not assumed -- see the CALIBRATION block.
// nvhbi_route_probe checks the same thing more thoroughly and also confirms
// that a peer write really lands in the far die's L2.
//
// The background is oriented A -> B (die-A SMs writing die-B memory) so exp2's
// peer traffic shares the same link and direction. NVHBI_BG_R_SMS adds crossing
// READS to it, issued by die-B SMs pulling from die A: that payload travels
// A->B too, which is how the background is driven past what die A's stores
// alone can drive.
//
// MEASUREMENT -- one window, one method, every stream
// ---------------------------------------------------
// ncu serializes and replays profiled kernels and would destroy the
// concurrency, and no hardware counter reports NV-HBI traffic live. So every
// kernel keeps an in-kernel counter of the 32B sectors it issued, and the host
// samples ALL of them at the same two instants and divides by the same elapsed
// time. Background writes, background reads and the peer are therefore the same
// statistic over the same interval, and comparable with exp1's sampled_GBps and
// with nvhbi_dualdir.
//
// argv: [exp 2|3]   (2 -> peer targets far, 3 -> peer targets near)
//
// Env: NVHBI_FAR_DIE            override auto-detection (0 or 1)
//      NVHBI_BG_SMS_LIST        background writer SM counts, 0=off.
//                               default "0,16,32,64,74"; clamped to the die
//      NVHBI_BG_R_SMS           background crossing-READ SM counts, a LIST
//      NVHBI_BG_R_LOCAL         1 = those readers read their OWN die (control)
//      NVHBI_BG_LOCAL           1 = background writers stay on die A (control)
//      NVHBI_PEER_BLOCK_SIZES   peer block sizes, 0=off. The injection axis:
//                               below 32 it sets active lanes per warp (=
//                               sectors per store instruction), above 32 it
//                               sets warps per block.
//                               default "0,1,2,4,8,16,32,64,128"
//      NVHBI_PEER_BLOCKS_PER_SM peer blocks per SM, clamped to what stays
//                               resident. default 32
//      NVHBI_PEER_CHUNKS        peer footprint in 4KiB chunks. 0 = one chunk
//                               per injecting warp at the widest sweep point.
//      NVHBI_PEER_IDLE          spin cycles after each store group, 0 = full
//                               rate: the continuous injection-rate knob.
//      NVHBI_PEER_OVERLAP       1 = peer writes the SAME chunks as the
//                               background instead of a disjoint region.
//      NVHBI_TAG_CHECK          1 = stamp each writer and count who owns each
//                               sector of the shared range. Use with OVERLAP=1.
//      NVHBI_WINDOW_MS / _SETTLE_MS / NVHBI_REPEAT   default 200 / 100 / 3
//      NVHBI_PEER_MARGIN_MS     how long the peer's deadline outlasts the
//                               window; it has no stop flag. default 60
//      NVHBI_HBM_PER_DIE_GBPS   plausibility bound on bg_rd_GBps, default 4000
//      NVHBI_BUF_MULT           allocation as a multiple of L2, default 8

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

    unsigned int peer_chunks_req = env_u("NVHBI_PEER_CHUNKS", 0u);   // 0 = auto
    const unsigned int bg_block  = env_u("NVHBI_BG_BLOCK", 64u);
    const unsigned int bg_nbps   = env_u("NVHBI_BG_BLOCKS_PER_SM", 32u);
    const unsigned int bg_lines  = env_u("NVHBI_BG_LINES", 1u);
    const unsigned int bg_local  = env_u("NVHBI_BG_LOCAL", 0u);
    // Crossing READS added to the background: far-die SMs pulling from the
    // near die, so the payload travels near->far like the writes and the peer.
    // This is how the background is driven past what one die's stores can
    // drive. A LIST, so the whole curve is visible; bg_sms=0 with readers gives
    // read-alone rows. Reads must stream over far more than L2 or the reading
    // die's replicas serve them and nothing crosses -- use NVHBI_BUF_MULT=64.
    const unsigned int window_ms = env_u("NVHBI_WINDOW_MS", 200u);
    // Every stream is launched, allowed to settle, then sampled over ONE shared
    // window. peer_margin_ms is how much longer the peer's deadline runs past
    // that window; nvhbi_peer_write has no stop flag, so it drains on its own.
    const unsigned int settle_ms = env_u("NVHBI_SETTLE_MS", 100u);
    const unsigned int peer_margin_ms = env_u("NVHBI_PEER_MARGIN_MS", 60u);
    // Readers stay on their own die: same kernel, same accounting, nothing
    // crosses. The read-side control, and the only in-run calibration of the
    // read byte count.
    const unsigned int bg_r_local = env_u("NVHBI_BG_R_LOCAL", 0u);
    const unsigned int repeat    = env_u("NVHBI_REPEAT", 3u);
    // Only the kernel path (P2P load/store) is supported: every stream is
    // sampled from an in-kernel counter over one shared window, and a
    // copy-engine path has no such counter.
    const unsigned int peer_mode = env_u("NVHBI_PEER_MODE", 0u);
    if (peer_mode != 0u) {
        fprintf(stderr, "ERROR: NVHBI_PEER_MODE=1 (cudaMemcpyPeerAsync) is not supported\n"
                        "       any more. Every stream is now sampled from an in-kernel\n"
                        "       counter over one shared window so that the background and\n"
                        "       the peer are measured the same way; a copy-engine path has\n"
                        "       no such counter and would reintroduce exactly the\n"
                        "       two-different-statistics problem this replaced.\n");
        return 1;
    }
    const double       buf_mult  = (double)env_u("NVHBI_BUF_MULT", 8u);
    const unsigned int peer_idle    = env_u("NVHBI_PEER_IDLE", 0u);
    const unsigned int peer_overlap = env_u("NVHBI_PEER_OVERLAP", 0u);
    // Stamp each writer's stores with a source tag and, after every window,
    // count who owns each sector of the shared range. Only meaningful with
    // NVHBI_PEER_OVERLAP=1, where there IS a shared range. Costs nothing in the
    // store loop (the tag is folded into the initial value) but adds a readback
    // pass per point, so it is off by default.
    const unsigned int tag_check = env_u("NVHBI_TAG_CHECK", 0u);

    // The injection axis is the peer BLOCK SIZE at a fixed blocks-per-SM;
    // sweeping the grid instead put every point past saturation (16 blocks of 128
    // threads already reached 644 GB/s) and, more usefully, block size is the only
    // knob that reaches below one full warp: at blockDim<32 a store instruction
    // touches fewer than 32 sectors, which the grid size cannot express.
    //
    // At 32 blocks/SM the occupancy ceiling is real and shows up inside the
    // sweep: 32 x 128 = 4096 threads per SM against a 2048 limit, so block size
    // 128 can only keep 16 blocks/SM resident. That is clamped rather than run,
    // because a grid that needs two waves makes EACH wave run the full deadline
    // and halves the reported bandwidth for a reason that has nothing to do with
    // the fabric.
    const unsigned int peer_nbps = env_u("NVHBI_PEER_BLOCKS_PER_SM", 32u);
    unsigned int bg_list[32], bs_list[32], bgr_list[32];
    const int bg_n = parse_list("NVHBI_BG_SMS_LIST", "0,16,32,64,74", bg_list, 32);
    const int bgr_n = parse_list("NVHBI_BG_R_SMS", "0", bgr_list, 32);
    const int bs_n = parse_list("NVHBI_PEER_BLOCK_SIZES",
                                "0,1,2,4,8,16,32,64,128", bs_list, 32);

    // Resident blocks per SM for a block size: thread-limited at 2048 threads,
    // and never more than the 32 block slots an SM has.
    auto resident_bps = [&](unsigned int bs, unsigned int want) {
        unsigned int bps = want;
        const unsigned int thr = (bs && bs <= 2048u) ? (2048u / bs) : 1u;
        if (bps > thr) bps = thr;
        if (bps > 32u) bps = 32u;
        return bps ? bps : 1u;
    };

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
    int prop1_clock_khz = 0;   // clockRate left cudaDeviceProp in CUDA 13
    CHECK_CUDA(cudaDeviceGetAttribute(&prop1_clock_khz, cudaDevAttrClockRate, 1));

    // Every injecting warp needs a chunk of its own: nvhbi_peer_write hands warp
    // w the chunks w, w+nwarps, ... so a warp with gwarp >= count never enters the
    // loop body and contributes nothing but occupancy. Size the region from the
    // widest point of the sweep rather than a fixed default, or the top of the
    // block-size axis silently measures fewer warps than it launched.
    unsigned int peer_warps_max = 0;
    for (int i = 0; i < bs_n; ++i) {
        if (!bs_list[i]) continue;
        const unsigned int w = (unsigned int)prop1.multiProcessorCount
                             * resident_bps(bs_list[i], peer_nbps)
                             * ((bs_list[i] + 31u) / 32u);
        if (w > peer_warps_max) peer_warps_max = w;
    }
    const bool peer_chunks_auto = (peer_chunks_req == 0u);
    if (peer_chunks_auto) peer_chunks_req = peer_warps_max ? peer_warps_max : 4096u;

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

    /* =============== CALIBRATION: which die is NVLink on? ===============
       Times an ATOMIC -- it always travels to the home L2 slice, so no local
       copy can serve it -- on a small resident working set, after a warm-up,
       and takes the minimum. The dies are probed in ALTERNATING order across
       rounds so first-touch cost cannot land on whichever goes first.
       nvhbi_route_probe does the same with an SM on each die of BOTH GPUs,
       which is the stronger check. ===================================== */
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

    // Two GPU1 SMs far apart in the id space, so they most likely sit on
    // different GPU1 dies. Probing both against both GPU0 dies gives a 2x2:
    // the spread across GPU0 dies is GPU0's hop, the spread across GPU1 SMs is
    // GPU1's own hop.
    const unsigned int smA = 0u;
    const unsigned int smB = (unsigned int)prop1.multiProcessorCount - 1u;

    auto peer_latency_sm = [&](unsigned int die, unsigned int psm,
                               unsigned int* mn, unsigned int* me) {
        const unsigned int nc = (die_count(die) < lat_chunks) ? die_count(die) : lat_chunks;
        nvhbi_peer_latency<<<prop1.multiProcessorCount, 1>>>(
                                     t.d_data, die_list1(die), 0u, nc,
                                     lat_reps, lat_warm, psm, d_lat, d_lat2, d_sink1);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(mn, d_lat,  sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(me, d_lat2, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    };
    auto peer_latency_once = [&](unsigned int die, unsigned int* mn, unsigned int* me) {
        peer_latency_sm(die, smA, mn, me);
    };

    unsigned int lat_loc = ~0u, mean_loc = 0u;
    {
        unsigned int mn, me;
        for (unsigned int r = 0; r < 2u; ++r) {
            nvhbi_peer_latency<<<prop1.multiProcessorCount, 1>>>(
                                         d_local_buf, d_local_idx, 0u, lat_chunks,
                                         lat_reps, lat_warm, smA, d_lat, d_lat2, d_sink1);
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

    /* -------- uncontended peer WRITE bandwidth to each die: a sanity number.
       With NV-HBI >> NVLink both dies pin to ~NVLink, so the latency above is
       the real attachment discriminator. -------- */
    auto peer_bw = [&](unsigned int die) -> double {
        CHECK_CUDA(cudaSetDevice(0));
        nvhbi_flush_l2(t);
        const unsigned int cnt = (die_count(die) < peer_chunks_req) ? die_count(die) : peer_chunks_req;
        nvhbi_warm(t, (die == 1u) ? t.d_far_idx : t.d_near_idx, 0u, cnt, die);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaMemset(d_peer_prog, 0, sizeof(unsigned long long)));
        const unsigned long long dl = (unsigned long long)window_ms * (unsigned long long)prop1_clock_khz;
        // Same residency clamp the sweep uses. Without it this launched
        // sm_count x 32 = 4736 blocks of 128 threads against a 16-blocks/SM
        // occupancy limit -- exactly two waves. Each wave runs its own full
        // deadline (the deadline is per-warp, from that warp's own t0), and the
        // second wave's warps all have gwarp >= count so they spin without
        // storing anything. The event span became 2 x window while the counter
        // held one window's worth of stores, and this reported exactly half:
        // 315.3 / 316.5 GB/s against the sweep's 631 / 633 on the same path.
        const unsigned int cal_bps = resident_bps(128u, 32u);          // 16
        unsigned int cal_pb = (unsigned int)prop1.multiProcessorCount * cal_bps;
        const unsigned int cal_useful = cnt / 4u;                      // 4 warps/block
        if (cal_useful && cal_pb > cal_useful) cal_pb = cal_useful;
        if (!cal_pb) cal_pb = 1u;
        CHECK_CUDA(cudaEventRecord(pe0, s_peer));
        nvhbi_peer_write<<<cal_pb, 128, 0, s_peer>>>(
            t.d_data, die_list1(die), 0u, cnt, 0u, dl, 0u, d_peer_prog, d_sink1);
        CHECK_CUDA(cudaEventRecord(pe1, s_peer));
        CHECK_CUDA(cudaStreamSynchronize(s_peer));
        float ms = 0.f; CHECK_CUDA(cudaEventElapsedTime(&ms, pe0, pe1));
        unsigned long long st = 0; CHECK_CUDA(cudaMemcpy(&st, d_peer_prog, sizeof(st), cudaMemcpyDeviceToHost));
        return (ms > 0.f) ? (double)st * 32.0 / (ms * 1e-3) / 1e9 : 0.0;
    };
    const double bw0 = peer_bw(0u);
    const double bw1 = peer_bw(1u);

    // 2x2: does the ISSUING GPU's die matter too?
    unsigned int m2[2][2];   // [gpu1 sm A/B][gpu0 die]
    {
        unsigned int mn, me;
        for (int a = 0; a < 2; ++a)
            for (unsigned int d = 0; d < 2u; ++d) {
                peer_latency_sm(d, a ? smB : smA, &mn, &me);
                m2[a][d] = mn;
            }
    }

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
    printf("  2x2 min latency (issuing GPU1 SM x target GPU0 die):\n");
    printf("             GPU0 die0   GPU0 die1   | GPU0 hop\n");
    printf("    GPU1 SM%-3u  %6u      %6u     | %+5d\n", smA, m2[0][0], m2[0][1],
           (int)m2[0][1] - (int)m2[0][0]);
    printf("    GPU1 SM%-3u  %6u      %6u     | %+5d\n", smB, m2[1][0], m2[1][1],
           (int)m2[1][1] - (int)m2[1][0]);
    printf("    GPU1 hop    %+5d       %+5d\n",
           (int)m2[1][0] - (int)m2[0][0], (int)m2[1][1] - (int)m2[0][1]);
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
    // Default: step past the background's chunks so the two never touch the same
    // line. NVHBI_PEER_OVERLAP=1 puts them on the SAME chunks instead, which is a
    // different question -- not "do two streams share the fabric" but "what does
    // the coherence/L2 machinery do when both dies write the same lines". Only
    // meaningful when both are aimed at the same die.
    const bool peer_ovl = peer_overlap && (peer_die == bg_target_die);
    if (peer_overlap && !peer_ovl)
        fprintf(stderr, "note: NVHBI_PEER_OVERLAP ignored -- peer targets die%u but the "
                        "background targets die%u, so there is nothing to overlap\n",
                peer_die, bg_target_die);
    const unsigned int  peer_first = (peer_ovl || peer_die != bg_target_die) ? 0u : bg_reserve;
    unsigned int peer_chunks = peer_chunks_req;
    if (peer_first + peer_chunks > peer_avail)
        peer_chunks = (peer_avail > peer_first) ? (peer_avail - peer_first) : 0u;
    if (peer_chunks == 0u) { fprintf(stderr, "ERROR: no room for peer region\n"); return 1; }

    /* -------- background read region (NVHBI_BG_R_SMS) --------
       Readers sit on the FAR die and pull from the NEAR die, so the payload
       travels near->far: the same direction as the background's writes and as
       the peer. They must stream over far more than L2 or the far die's
       read-only replicas answer them and nothing crosses -- run with
       NVHBI_BUF_MULT=64.

       The offset is bg_reserve + 256, which is exactly what nvhbi_dualdir uses
       (w_max_chunks + 256). Matching it is the point: the two harnesses must be
       byte-for-byte the same experiment before their 34% disagreement can be
       blamed on anything but the reader's starting position. */
    unsigned int bg_r_max = 0;
    for (int i = 0; i < bgr_n; ++i) if (bgr_list[i] > bg_r_max) bg_r_max = bgr_list[i];
    const unsigned int bg_r_src   = near_die;
    const unsigned int bg_r_first = bg_reserve + 256u;
    const unsigned int bg_r_avail = die_count(bg_r_src);
    const unsigned int bg_r_chunks = (bg_r_avail > bg_r_first)
                                   ? (bg_r_avail - bg_r_first) : 0u;
    // A read that lands above one die's HBM cannot be a real remote read; it is
    // replicas on the reading die being counted as crossings. 4 HBM3e stacks per
    // die at ~1 TB/s each is the ceiling to check against.
    const double hbm_die_gbps = (double)env_u("NVHBI_HBM_PER_DIE_GBPS", 4000u);
    if (bg_r_max) {
        if (!bg_r_chunks) {
            fprintf(stderr, "ERROR: no room for the background read sweep\n");
            return 1;
        }
        printf("  background reads: die%u SMs <- die%u, chunks [%u,%u) "
               "(%.0f MB = %.0fx per-die L2), up to %u SMs\n",
               far_die, bg_r_src, bg_r_first, bg_r_first + bg_r_chunks,
               bg_r_chunks * (double)NVHBI_CHUNK_BYTES / 1048576.0,
               bg_r_chunks * (double)NVHBI_CHUNK_BYTES / (t.l2_bytes / 2.0),
               bg_r_max);
        if (bg_r_chunks * (double)NVHBI_CHUNK_BYTES < 8.0 * (t.l2_bytes / 2.0))
            fprintf(stderr, "WARNING: the read sweep is only %.0fx a die's L2. Replicas on\n"
                            "         the reading die will serve most of it and little will\n"
                            "         cross. Raise NVHBI_BUF_MULT (64 is what dualdir uses).\n",
                    bg_r_chunks * (double)NVHBI_CHUNK_BYTES / (t.l2_bytes / 2.0));
    }

    /* -------- background machinery on GPU0 -------- */
    printf("  peer path: GPU1 kernel storing into GPU0 (P2P load/store)\n");

    CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t s_bg;
    CHECK_CUDA(cudaStreamCreateWithFlags(&s_bg, cudaStreamNonBlocking));
    unsigned long long* d_bg_prog = nullptr;
    CHECK_CUDA(cudaMalloc(&d_bg_prog, sizeof(unsigned long long)));
    unsigned long long* d_bg_rprog = nullptr;
    CHECK_CUDA(cudaMalloc(&d_bg_rprog, sizeof(unsigned long long)));
    // The background's achieved SM clock. exp1 measured 83.8 GB/s per SM at
    // 1.22 GHz; this experiment sees 62.8, exactly 0.75x at every SM count. A
    // flat ratio like that is a clock difference, not a structural one -- and
    // exp2's background runs for seconds, so it throttles where exp1's 20-30 ms
    // kernels did not. Without this column the two cannot be compared.
    unsigned long long* d_bg_cyc = nullptr;
    CHECK_CUDA(cudaMalloc(&d_bg_cyc, sizeof(unsigned long long)));
    NvhbiStopFlag stop;
    nvhbi_stop_flag_create(stop);

    unsigned long long* d_hist = nullptr;
    if (tag_check) {
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaMalloc(&d_hist, 16 * sizeof(unsigned long long)));
        const unsigned int tag_bg = 1u, tag_peer = 2u;
        CHECK_CUDA(cudaMemcpyToSymbol(nvhbi_src_tag, &tag_bg, sizeof(tag_bg)));
        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaMemcpyToSymbol(nvhbi_src_tag, &tag_peer, sizeof(tag_peer)));
        CHECK_CUDA(cudaSetDevice(0));
        printf("  tag check ON: background stores carry tag 1, peer stores tag 2.\n"
               "    After each window the shared range is read back and every\n"
               "    sector attributed to whoever wrote it last. A tag that never\n"
               "    shows up is a writer that is not reaching these bytes at all,\n"
               "    whatever its own counter says it issued.\n");
        if (!peer_ovl)
            printf("    NOTE: peer and background do not share a range in this run,\n"
                   "          so the two tags will simply partition by region.\n");
    }

    const unsigned int* d_peer_idx = die_list1(peer_die);   // GPU1-side list for the target die

    // Time-resolved trace of the background alone. exp1 measures over ms 8-25 of
    // a 25 ms kernel; this experiment samples ms 100-300 of a 300 ms one. If the
    // rate decays with time, the two are not measuring the same thing and that
    // alone explains the 33% gap.
    if (env_u("NVHBI_BG_TRACE", 0u)) {
        const unsigned int trace_ms = env_u("NVHBI_BG_TRACE_MS", 1000u);
        const unsigned int step_ms  = env_u("NVHBI_BG_TRACE_STEP", 50u);
        const unsigned int tsms     = (bg_max ? bg_max : bg_sms_max);
        const unsigned int tchunks  = nvhbi_chunks_used(tsms, bg_nbps, bg_block, bg_lines);
        CHECK_CUDA(cudaSetDevice(0));
        nvhbi_flush_l2(t);
        nvhbi_warm(t, (bg_target_die == 1u) ? t.d_far_idx : t.d_near_idx,
                   0u, tchunks, bg_target_die);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemset(d_bg_prog, 0, sizeof(unsigned long long)));
        nvhbi_stop_flag_reset(stop);
        const unsigned long long dl =
            (unsigned long long)(trace_ms + 2000u) * (unsigned long long)t.clock_khz;
        nvhbi_stress_write<<<t.sm_count * bg_nbps, bg_block, 0, s_bg>>>(
            t.d_data, t.d_far_idx, t.d_near_idx, t.d_sm_side,
            bg_writer_partition, bg_local, tsms, bg_nbps,
            (unsigned int)t.sm_count, bg_lines, 0u,
            0u, dl, stop.d, d_bg_prog, d_bg_cyc, nullptr, t.d_sink);
        CHECK_CUDA(cudaGetLastError());
        printf("# TRACE,t_ms,interval_GBps   (background alone, %u SMs)\n", tsms);
        unsigned long long prev = 0ull;
        const double t_start = now_ms();
        double next = t_start + (double)step_ms;
        while (now_ms() - t_start < (double)trace_ms) {
            while (now_ms() < next) { }
            const double now = now_ms();
            unsigned long long cur = 0ull;
            CHECK_CUDA(cudaMemcpy(&cur, d_bg_prog, sizeof(cur), cudaMemcpyDeviceToHost));
            printf("TRACE,%.0f,%.1f\n", now - t_start,
                   (double)(cur - prev) * 32.0 / (step_ms * 1e-3) / 1e9);
            fflush(stdout);
            prev = cur;
            next += (double)step_ms;
        }
        nvhbi_stop_flag_set(stop);
        CHECK_CUDA(cudaStreamSynchronize(s_bg));
        printf("\n");
    }

    // New columns are APPENDED, never inserted: the run scripts' awk still
    // addresses the old ones by position.
    // bg_GBps stays in its old position and keeps its old meaning (the
    // background's WRITE rate). bg_rd_GBps is appended: the background's
    // crossing-read rate, 0 unless NVHBI_BG_R_SMS is set.
    printf("# CFG,exp,far_die,peer_die,bg_local,bg_sms,peer_blocks,rep,"
           "peer_ms,peer_GBps,bg_GBps,crossing_GBps,bg_GHz,peer_ovl,peer_bsize,"
           "bg_rd_GBps,bg_r_sms\n");

    for (int bi = 0; bi < bg_n; ++bi) {
    for (int ri = 0; ri < bgr_n; ++ri) {
    for (int pi = 0; pi < bs_n; ++pi) {
        const unsigned int bg_sms  = bg_list[bi];
        const unsigned int peer_bs = bs_list[pi];
        // nvhbi_dual's writers always target the OTHER die -- it has no own-die
        // mode -- so readers paired with bg_local=1 would give a "control" whose
        // writes still cross. Skip rather than report a broken control.
        if (bg_local && bgr_list[ri]) continue;
        const unsigned int bg_r_sms = bgr_list[ri];
        // bg_sms=0 with readers is a READ-ALONE row, and it is the row that
        // reconciles this program against nvhbi_dualdir's w=0 series. It used to
        // be impossible because the launch was gated on bg_sms alone.
        const bool bg_on = (bg_sms != 0u) || (bg_r_sms != 0u);
        if (!bg_on && peer_bs == 0u) continue;
        const unsigned int bg_chunks = bg_sms
            ? nvhbi_chunks_used(bg_sms, bg_nbps, bg_block, bg_lines) : 0u;

        for (unsigned int rep = 0; rep < repeat; ++rep) {
            /* flush + warm both regions from their owning die */
            CHECK_CUDA(cudaSetDevice(0));
            nvhbi_flush_l2(t);
            if (bg_chunks) {
                nvhbi_warm(t, (bg_target_die == 1u) ? t.d_far_idx : t.d_near_idx,
                           0u, bg_chunks, bg_target_die);
            }
            if (peer_bs) {
                nvhbi_warm(t, d_peer_own, peer_first, peer_chunks, peer_die);
            }
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            /* ---------------------------------------------------------------
               ONE window, ONE method, for every stream.

               Both loads are launched, both settle, and then the host samples
               every in-kernel counter at the same two instants and divides by
               the same elapsed time. That is exactly what exp1's sampled_GBps
               and nvhbi_dualdir already do, so all four programs now report the
               same statistic and their numbers can be put in one table.

               What this replaces: the peer used to be timed with cudaEvents over
               its WHOLE kernel while the background was sampled mid-flight over
               a host window. Two different statistics over two different
               intervals -- which is not a defensible way to ask whether one
               stream is slowing the other down, however small the difference
               turns out to be.

               The peer's deadline therefore has to outlast settle + window; it
               is given a margin and simply runs out afterwards (nvhbi_peer_write
               has no stop flag, and the drain costs one margin per point).
               --------------------------------------------------------------- */
            const unsigned long long bg_dl =
                (unsigned long long)(settle_ms + 3u * window_ms + 2000u)
                * (unsigned long long)t.clock_khz;
            const unsigned long long peer_dl =
                (unsigned long long)(settle_ms + window_ms + peer_margin_ms)
                * (unsigned long long)prop1_clock_khz;

            CHECK_CUDA(cudaMemset(d_bg_prog, 0, sizeof(unsigned long long)));
            CHECK_CUDA(cudaMemset(d_bg_rprog, 0, sizeof(unsigned long long)));
            CHECK_CUDA(cudaMemset(d_bg_cyc, 0, sizeof(unsigned long long)));
            nvhbi_stop_flag_reset(stop);

            // Clamp the peer grid. Two ways it silently broke the sweep before:
            //  * more blocks than fit resident -> each WAVE runs the full
            //    deadline, so the span became waves x window.
            //  * more warps than chunks -> the surplus warps own no chunk and
            //    just spin, adding waves without adding traffic.
            unsigned int pb = 0u, bps = 0u;
            if (peer_bs) {
                const unsigned int wpb = (peer_bs + 31u) / 32u;
                bps = resident_bps(peer_bs, peer_nbps);
                pb  = (unsigned int)prop1.multiProcessorCount * bps;
                unsigned int useful = peer_chunks / wpb;
                if (!useful) useful = 1u;
                if (pb > useful) pb = useful;
            }

            double bg_launch_ms = 0.0;
            if (bg_on) {
                CHECK_CUDA(cudaSetDevice(0));
                if (bg_r_sms) {
                    nvhbi_dual<<<t.sm_count * bg_nbps, bg_block, 0, s_bg>>>(
                        t.d_data, t.d_far_idx, t.d_near_idx, t.d_sm_side,
                        bg_writer_partition, bg_sms, bg_r_sms, bg_r_local, bg_nbps,
                        (unsigned int)t.sm_count, bg_r_chunks, bg_r_first,
                        /*r_evict_first=*/1u, bg_dl, stop.d,
                        d_bg_prog, d_bg_rprog, d_bg_cyc, t.d_sink);
                } else {
                    nvhbi_stress_write<<<t.sm_count * bg_nbps, bg_block, 0, s_bg>>>(
                        t.d_data, t.d_far_idx, t.d_near_idx, t.d_sm_side,
                        bg_writer_partition, bg_local, bg_sms, bg_nbps,
                        (unsigned int)t.sm_count, bg_lines, 0u,
                        0u, bg_dl, stop.d, d_bg_prog, d_bg_cyc, nullptr, t.d_sink);
                }
                CHECK_CUDA(cudaGetLastError());
                bg_launch_ms = now_ms();
            }

            bool peer_running = false;
            if (pb && peer_bs) {
                CHECK_CUDA(cudaSetDevice(1));
                CHECK_CUDA(cudaMemset(d_peer_prog, 0, sizeof(unsigned long long)));
                nvhbi_peer_write<<<pb, peer_bs, 0, s_peer>>>(
                    t.d_data, d_peer_idx, peer_first, peer_chunks, 0u, peer_dl,
                    peer_idle, d_peer_prog, d_sink1);
                CHECK_CUDA(cudaGetLastError());
                peer_running = true;
                CHECK_CUDA(cudaSetDevice(0));
            }

            spin_ms((double)settle_ms);

            /* ---- sample: every counter at the same instant, both ends ---- */
            unsigned long long bg_p0 = 0, bg_p1 = 0, bg_r0 = 0, bg_r1 = 0;
            unsigned long long pq0 = 0, pq1 = 0;
            auto sample = [&](unsigned long long* w, unsigned long long* r,
                              unsigned long long* q) {
                if (bg_on) {
                    CHECK_CUDA(cudaSetDevice(0));
                    CHECK_CUDA(cudaMemcpy(w, d_bg_prog, sizeof(*w), cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaMemcpy(r, d_bg_rprog, sizeof(*r), cudaMemcpyDeviceToHost));
                }
                if (peer_running) {
                    CHECK_CUDA(cudaSetDevice(1));
                    CHECK_CUDA(cudaMemcpy(q, d_peer_prog, sizeof(*q), cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaSetDevice(0));
                }
            };
            // Stamp the clock AFTER the copies land at both ends, so the same
            // bias sits on each and cancels in the interval.
            sample(&bg_p0, &bg_r0, &pq0);
            const double t_a = now_ms();
            while (now_ms() - t_a < (double)window_ms) { }
            sample(&bg_p1, &bg_r1, &pq1);
            const double t_b = now_ms();
            const double win_ms = t_b - t_a;

            const unsigned long long peer_stores = pq1 - pq0;
            const float peer_ms = (float)win_ms;

            unsigned long long bg_cyc = 0ull;
            double bg_span_ms = 0.0;
            if (bg_on) {
                nvhbi_stop_flag_set(stop);
                CHECK_CUDA(cudaSetDevice(0));
                CHECK_CUDA(cudaStreamSynchronize(s_bg));
                CHECK_CUDA(cudaMemcpy(&bg_cyc, d_bg_cyc, sizeof(bg_cyc),
                                      cudaMemcpyDeviceToHost));
                // Stamp the span HERE, before waiting out the peer's drain. Taking
                // it afterwards charged the background with the peer's margin and
                // reported 1.64 GHz on a GPU pinned at 1.96 -- a pure artifact
                // that looked exactly like throttling under peer load.
                bg_span_ms = now_ms() - bg_launch_ms;
            }
            if (peer_running) {
                CHECK_CUDA(cudaSetDevice(1));
                CHECK_CUDA(cudaStreamSynchronize(s_peer));
                CHECK_CUDA(cudaSetDevice(0));
            }
            const double wall_ms = win_ms;

            // Clock the background actually achieved. The span is measured from
            // the host, launch to drain: warps only notice the stop flag at a
            // poll boundary so they overrun it, and a hard-coded span would
            // inflate this.
            const double bg_ghz = (bg_on && bg_cyc && bg_span_ms > 0.0)
                ? (double)bg_cyc / (bg_span_ms * 1e6) : 0.0;

            if (tag_check && peer_bs && bg_sms) {
                // Both writers are stopped by now: the peer hit its deadline and
                // the background's stop flag was set and synchronised above, so
                // the memory state is quiescent and the last writer per sector
                // is stable.
                const unsigned int rng = (bg_chunks < peer_chunks) ? bg_chunks : peer_chunks;
                CHECK_CUDA(cudaSetDevice(0));
                CHECK_CUDA(cudaMemset(d_hist, 0, 16 * sizeof(unsigned long long)));
                nvhbi_count_tags<<<t.sm_count * 8, 128>>>(
                    t.d_data, d_peer_own, peer_first, rng, d_hist, t.d_sink);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());
                unsigned long long h[16];
                CHECK_CUDA(cudaMemcpy(h, d_hist, sizeof(h), cudaMemcpyDeviceToHost));
                unsigned long long tot = 0;
                for (int i = 0; i < 16; ++i) tot += h[i];
                unsigned long long other = tot - h[1] - h[2];
                printf("TAG,%u,%u,%u,%llu,%llu,%llu,%.1f\n",
                       bg_sms, peer_bs, rng,
                       (unsigned long long)h[1], (unsigned long long)h[2],
                       (unsigned long long)other,
                       tot ? 100.0 * (double)h[2] / (double)tot : 0.0);
                fflush(stdout);
            }

            const double peer_gbps = (peer_ms > 0.f)
                ? (double)peer_stores * 32.0 / (peer_ms * 1e-3) / 1e9 : 0.0;
            const double bg_gbps = (bg_on && wall_ms > 0.0)
                ? (double)(bg_p1 - bg_p0) * 32.0 / (wall_ms * 1e-3) / 1e9 : 0.0;
            const double bg_rd_gbps = (bg_on && wall_ms > 0.0)
                ? (double)(bg_r1 - bg_r0) * 32.0 / (wall_ms * 1e-3) / 1e9 : 0.0;
            // Everything the near->far direction is carrying: the background's
            // writes, its crossing reads, and the peer when it targets the far
            // die. bg_local=1 crosses with neither payload.
            const double crossing = (bg_local ? 0.0 : bg_gbps + bg_rd_gbps)
                                  + ((exp == 2u) ? peer_gbps : 0.0);

            if (bg_rd_gbps > hbm_die_gbps)
                fprintf(stderr, "WARNING: bg read reads %.0f GB/s from one die, above the %.0f GB/s\n"
                                "         that die's HBM stacks can supply. Some of it is being\n"
                                "         served by replicas on the reading die and counted as\n"
                                "         crossing traffic -- do not quote this number.\n",
                        bg_rd_gbps, hbm_die_gbps);
            printf("CFG,%u,%u,%u,%u,%u,%u,%u,%.4f,%.2f,%.2f,%.2f,%.3f,%u,%u,%.2f,%u\n",
                   exp, far_die, peer_die, bg_local, bg_sms,
                   peer_bs ? pb : 0u, rep,
                   peer_ms, peer_gbps, bg_gbps, crossing, bg_ghz,
                   peer_ovl ? 1u : 0u, peer_bs, bg_rd_gbps, bg_r_sms);
            fflush(stdout);
        }
    }}}

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaEventDestroy(pe0)); CHECK_CUDA(cudaEventDestroy(pe1));
    CHECK_CUDA(cudaStreamDestroy(s_peer));
    CHECK_CUDA(cudaFree(d_near1)); CHECK_CUDA(cudaFree(d_far1));
    CHECK_CUDA(cudaFree(d_sink1)); CHECK_CUDA(cudaFree(d_lat)); CHECK_CUDA(cudaFree(d_lat2));
    CHECK_CUDA(cudaFree(d_peer_prog));
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaStreamDestroy(s_bg)); CHECK_CUDA(cudaFree(d_bg_prog));
    CHECK_CUDA(cudaFree(d_bg_rprog));
    if (d_hist) CHECK_CUDA(cudaFree(d_hist));
    CHECK_CUDA(cudaFree(d_bg_cyc));
    nvhbi_stop_flag_destroy(stop);
    nvhbi_free(t);
    return 0;
}
