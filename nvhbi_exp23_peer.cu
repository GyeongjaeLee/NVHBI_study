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
//      NVHBI_PEER_BLOCKS_PER_SM peer blocks per SM, FIXED across the sweep. default 32
//                               (the same convention as NVHBI_BG_BLOCKS_PER_SM).
//                               The launched grid is sm_count x this, clamped to
//                               what stays resident -- see the note below.
//      NVHBI_PEER_BLOCK_SIZES   peer block sizes, 0=off. default "0,1,2,4,8,16,32,64,128"
//                               This is the swept axis. Below 32 it varies the
//                               ACTIVE LANES PER WARP, i.e. how many 32B sectors
//                               one store instruction touches (1..32); above 32
//                               it varies warps per block (1..4). One continuous
//                               injection ramp from 32 threads to 4096.
//      NVHBI_BG_LOCAL           1 = background stays on die A (control), default 0
//      NVHBI_WINDOW_MS          measurement window per point, default 200
//      NVHBI_REPEAT             reps per point,               default 3
//      NVHBI_PEER_CHUNKS        peer footprint in 4KiB chunks. 0 (default) sizes
//                               it automatically to one chunk per injecting warp
//                               at the widest point of the sweep -- a warp with no
//                               chunk of its own contributes nothing.
//      NVHBI_BG_BLOCK / NVHBI_BG_BLOCKS_PER_SM / NVHBI_BG_LINES
//      NVHBI_PEER_MODE          0 = GPU1 kernel P2P stores (default)
//                               1 = cudaMemcpyPeerAsync (copy engines, die-agnostic)
//      NVHBI_PEER_IDLE          spin cycles after each store group, 0 = full rate.
//                               The continuous injection-rate knob; use it to get
//                               sub-saturation points on the NVLink-load axis.
//      NVHBI_TAG_CHECK          1 = stamp each writer's stores and count, after
//                               every window, who owns each sector of the shared
//                               range. Prints TAG rows. Use with PEER_OVERLAP=1.
//      NVHBI_PEER_OVERLAP       1 = peer writes the SAME chunks the background
//                               writes, 0 (default) = a disjoint region.
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

    unsigned int peer_chunks_req = env_u("NVHBI_PEER_CHUNKS", 0u);   // 0 = auto
    const unsigned int bg_block  = env_u("NVHBI_BG_BLOCK", 64u);
    const unsigned int bg_nbps   = env_u("NVHBI_BG_BLOCKS_PER_SM", 32u);
    const unsigned int bg_lines  = env_u("NVHBI_BG_LINES", 1u);
    const unsigned int bg_local  = env_u("NVHBI_BG_LOCAL", 0u);
    // Crossing READS added to the background, issued by the FAR die's SMs
    // pulling from the NEAR die. The payload travels near->far, the same
    // direction as the background writes and the same direction as the peer, so
    // all three load one direction of NV-HBI -- and reads are how you get past
    // what one die's SMs can push with stores alone.
    //
    // This exists because the write-only background tops out around 3.4 TB/s
    // while nvhbi_dualdir showed the direction carrying more than that, so
    // bg + peer never reached the fabric's capacity and "no interference" was
    // the expected answer under every hypothesis. With readers the background
    // can saturate first, and only then does the peer's 630 GB/s have to
    // displace something -- or visibly fail to, which is the real result.
    //
    // Reads must stream over a footprint far larger than L2 or the requester
    // die's read-only replicas serve them and nothing crosses: run this with
    // NVHBI_BUF_MULT=64, not the default 8.
    unsigned int bg_r_sms = env_u("NVHBI_BG_R_SMS", 0u);
    const unsigned int window_ms = env_u("NVHBI_WINDOW_MS", 200u);
    const unsigned int repeat    = env_u("NVHBI_REPEAT", 3u);
    // How GPU1 pushes into GPU0:
    //   0 = a kernel on GPU1 dereferencing GPU0 pointers (P2P load/store, SMs)
    //   1 = cudaMemcpyPeerAsync (copy engines / DMA)
    // Different hardware paths, so they need not interact with NV-HBI the same
    // way. The kernel form is what showed zero interference with the
    // background; whether DMA behaves the same is a separate question.
    // CAVEAT for mode 1: a contiguous memcpy cannot be aimed at one die -- the
    // dies interleave at 4KiB -- so it lands ~50/50 and the exp2/exp3 (far/near)
    // distinction does not apply to it.
    const unsigned int peer_mode = env_u("NVHBI_PEER_MODE", 0u);
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
    unsigned int bg_list[32], bs_list[32];
    const int bg_n = parse_list("NVHBI_BG_SMS_LIST", "0,16,32,64,74", bg_list, 32);
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

    // DMA buffers for peer_mode 1. Destination is a plain contiguous GPU0
    // allocation, hence die-agnostic.
    unsigned int *d_dma_src = nullptr, *d_dma_dst = nullptr;
    size_t dma_bytes = 0;

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

    /* =====================================================================
       CALIBRATION 2: uncontended peer WRITE bandwidth to each die
       (with NV-HBI >> NVLink, both dies usually pin to ~NVLink BW, so this is
        mainly a sanity number; latency is the real attachment discriminator.)
       ===================================================================== */
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

    // nvhbi_dual's writers always target the OTHER die -- it has no own-die
    // mode -- so pairing readers with bg_local=1 would give a "control" whose
    // writes still cross. Refuse rather than report a broken control.
    if (bg_r_sms && bg_local) {
        fprintf(stderr, "note: NVHBI_BG_R_SMS ignored for bg_local=1 -- the fused "
                        "write+read background has no own-die write mode, so this "
                        "row would not be a control\n");
        bg_r_sms = 0u;
    }

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
       NVHBI_BUF_MULT=64. Step past the peer region when the peer happens to
       write into the die the readers are reading. */
    const unsigned int bg_r_src   = near_die;
    const unsigned int bg_r_first = (peer_die == bg_r_src)
                                  ? (peer_first + peer_chunks + 256u) : 0u;
    const unsigned int bg_r_avail = die_count(bg_r_src);
    const unsigned int bg_r_chunks = (bg_r_avail > bg_r_first)
                                   ? (bg_r_avail - bg_r_first) : 0u;
    if (bg_r_sms) {
        if (!bg_r_chunks) {
            fprintf(stderr, "ERROR: no room for the background read sweep\n");
            return 1;
        }
        printf("  background reads: die%u SMs <- die%u, chunks [%u,%u) "
               "(%.0f MB = %.0fx per-die L2)%s\n",
               far_die, bg_r_src, bg_r_first, bg_r_first + bg_r_chunks,
               bg_r_chunks * (double)NVHBI_CHUNK_BYTES / 1048576.0,
               bg_r_chunks * (double)NVHBI_CHUNK_BYTES / (t.l2_bytes / 2.0), "");
        if (bg_r_chunks * (double)NVHBI_CHUNK_BYTES < 8.0 * (t.l2_bytes / 2.0))
            fprintf(stderr, "WARNING: the read sweep is only %.0fx a die's L2. Replicas on\n"
                            "         the reading die will serve most of it and little will\n"
                            "         cross. Raise NVHBI_BUF_MULT (64 is what dualdir uses).\n",
                    bg_r_chunks * (double)NVHBI_CHUNK_BYTES / (t.l2_bytes / 2.0));
    }

    printf("exp%u: GPU1 --NVLink--> GPU0 die%u (%s)\n", exp, peer_die,
           (exp == 2u) ? "FAR: crosses NV-HBI" : "NEAR: no NV-HBI hop");
    printf("  background: die%u SMs -> die%u (%s), up to %u SMs\n",
           bg_writer_partition, bg_target_die,
           bg_local ? "OWN die = control" : "cross-die = bisection probe", bg_max);
    printf("  peer region: chunks [%u,%u) on die%u -> %.1f MB%s\n",
           peer_first, peer_first + peer_chunks, peer_die,
           peer_chunks * (double)NVHBI_CHUNK_BYTES / (1024.0 * 1024.0),
           peer_ovl ? "   [OVERLAPPING the background's chunks]" : "");
    printf("  peer grid: %u blocks per SM x %d SMs, block size is the swept axis\n"
           "             (below 32 it sets active lanes per warp = sectors per store\n"
           "              instruction; above 32 it sets warps per block)\n",
           peer_nbps, prop1.multiProcessorCount);
    printf("  peer region sized %s: %u chunks for up to %u injecting warps"
           " (%.0f%% of one die's L2)\n",
           peer_chunks_auto ? "automatically" : "by NVHBI_PEER_CHUNKS",
           peer_chunks, peer_warps_max,
           100.0 * peer_chunks * NVHBI_CHUNK_BYTES / (t.l2_bytes / 2.0));
    {
        // What die `peer_die` has to hold: the background's reserved chunks (when
        // it targets the same die) plus the peer region that sits after them.
        const double used = (double)(peer_first + peer_chunks) * NVHBI_CHUNK_BYTES;
        const double die_l2 = t.l2_bytes / 2.0;
        printf("  die%u L2 footprint: %.1f MB of %.1f MB (%.0f%%)\n\n",
               peer_die, used / 1048576.0, die_l2 / 1048576.0, 100.0 * used / die_l2);
        if (used > 0.8 * die_l2)
            fprintf(stderr, "WARNING: background + peer need %.0f%% of die%u's L2. Not all of\n"
                            "         it stays resident, so some writes miss to HBM and the\n"
                            "         numbers stop being pure remote-L2 traffic. Lower\n"
                            "         NVHBI_PEER_BLOCKS_PER_SM or NVHBI_PEER_CHUNKS.\n",
                    100.0 * used / die_l2, peer_die);
    }
    printf("  peer stores: %u idle cycles per store group\n", peer_idle);

    /* -------- background machinery on GPU0 -------- */
    if (peer_mode == 1u) {
        dma_bytes = (size_t)peer_chunks * NVHBI_CHUNK_BYTES;
        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaMalloc(&d_dma_src, dma_bytes));
        CHECK_CUDA(cudaMemset(d_dma_src, 0x3c, dma_bytes));
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaMalloc(&d_dma_dst, dma_bytes));
        printf("  peer path: cudaMemcpyPeerAsync, %.1f MB per copy (DIE-AGNOSTIC)\n",
               dma_bytes / 1048576.0);
    } else {
        printf("  peer path: GPU1 kernel storing into GPU0 (P2P load/store)\n");
    }

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
    for (int pi = 0; pi < bs_n; ++pi) {
        const unsigned int bg_sms      = bg_list[bi];
        const unsigned int peer_bs = bs_list[pi];
        if (bg_sms == 0u && peer_bs == 0u) continue;
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

            /* start background, settle */
            double bg_launch_ms = 0.0;
            CHECK_CUDA(cudaMemset(d_bg_prog, 0, sizeof(unsigned long long)));
            CHECK_CUDA(cudaMemset(d_bg_rprog, 0, sizeof(unsigned long long)));
            CHECK_CUDA(cudaMemset(d_bg_cyc, 0, sizeof(unsigned long long)));
            nvhbi_stop_flag_reset(stop);
            if (bg_sms) {
                // Must outlast settle + the whole peer kernel, or the counter
                // delta gets divided by a window the background was not alive
                // for. That artifact produced the fake "bg dropped to 120 GB/s"
                // rows at peer_blocks=16384.
                const unsigned long long dl =
                    (unsigned long long)(4u * window_ms + 2000u) * (unsigned long long)t.clock_khz;
                if (bg_r_sms) {
                    // Fused write+read background. r_local follows bg_local, so
                    // the control still crosses nothing on either payload.
                    nvhbi_dual<<<t.sm_count * bg_nbps, bg_block, 0, s_bg>>>(
                        t.d_data, t.d_far_idx, t.d_near_idx, t.d_sm_side,
                        bg_writer_partition, bg_sms, bg_r_sms, /*r_local=*/0u, bg_nbps,
                        (unsigned int)t.sm_count, bg_r_chunks, bg_r_first,
                        /*r_evict_first=*/1u, dl, stop.d,
                        d_bg_prog, d_bg_rprog, t.d_sink);
                } else {
                    nvhbi_stress_write<<<t.sm_count * bg_nbps, bg_block, 0, s_bg>>>(
                        t.d_data, t.d_far_idx, t.d_near_idx, t.d_sm_side,
                        bg_writer_partition, bg_local, bg_sms, bg_nbps,
                        (unsigned int)t.sm_count, bg_lines, 0u,
                        0u, dl, stop.d, d_bg_prog, d_bg_cyc, nullptr, t.d_sink);
                }
                CHECK_CUDA(cudaGetLastError());
                bg_launch_ms = now_ms();
                spin_ms(100.0);
            }

            /* measurement window */
            unsigned long long bg_p0 = 0, bg_p1 = 0, peer_stores = 0;
            unsigned long long bg_r0 = 0, bg_r1 = 0;
            CHECK_CUDA(cudaMemcpy(&bg_p0, d_bg_prog, sizeof(bg_p0), cudaMemcpyDeviceToHost));
            if (bg_r_sms)
                CHECK_CUDA(cudaMemcpy(&bg_r0, d_bg_rprog, sizeof(bg_r0), cudaMemcpyDeviceToHost));
            // Clamp the peer grid. Two ways it silently broke the sweep before:
            //  * more blocks than fit resident -> each WAVE runs the full deadline,
            //    so peer_ms became waves x window (200 -> 1402 ms at 16384 blocks)
            //    and peer_GBps fell by exactly that factor.
            //  * more warps than chunks -> the surplus warps own no chunk and just
            //    spin, adding waves without adding traffic.
            // wpb is forced to at least 1 -- the old peer_block/32 was 0 for any
            // sub-warp block size and the `useful` division below then divided by
            // zero, which would have crashed the moment this axis went below 32.
            unsigned int pb = 0u, bps = 0u;
            if (peer_bs) {
                const unsigned int wpb = (peer_bs + 31u) / 32u;
                bps = resident_bps(peer_bs, peer_nbps);
                pb  = (unsigned int)prop1.multiProcessorCount * bps;
                unsigned int useful = peer_chunks / wpb;      // blocks, so warps <= chunks
                if (!useful) useful = 1u;
                if (pb > useful) {
                    pb = useful;
                    fprintf(stderr, "note: block size %u: grid %u -> %u blocks "
                                    "(only %u chunks for %u warps/block)\n",
                            peer_bs, (unsigned int)prop1.multiProcessorCount * bps,
                            pb, peer_chunks, wpb);
                }
                if (bps != peer_nbps)
                    fprintf(stderr, "note: block size %u: %u -> %u blocks/SM "
                                    "(2048 threads/SM and 32 block slots)\n",
                            peer_bs, peer_nbps, bps);
            }

            float peer_ms = 0.f;
            const double wall0 = now_ms();
            if (pb && peer_bs && peer_mode == 1u) {
                // Keep the copy engines busy for the whole window, then divide
                // the bytes actually copied by the time they took.
                CHECK_CUDA(cudaSetDevice(1));
                CHECK_CUDA(cudaEventRecord(pe0, s_peer));
                unsigned int ncopy = 0u;
                const double until = now_ms() + (double)window_ms;
                while (now_ms() < until) {
                    CHECK_CUDA(cudaMemcpyPeerAsync(d_dma_dst, 0, d_dma_src, 1,
                                                   dma_bytes, s_peer));
                    ++ncopy;
                    if ((ncopy & 15u) == 0u) CHECK_CUDA(cudaStreamSynchronize(s_peer));
                }
                CHECK_CUDA(cudaEventRecord(pe1, s_peer));
                CHECK_CUDA(cudaStreamSynchronize(s_peer));
                CHECK_CUDA(cudaEventElapsedTime(&peer_ms, pe0, pe1));
                // Report DMA in the same units as the kernel path: 32B sectors.
                peer_stores = (unsigned long long)ncopy * (dma_bytes / 32u);
            } else if (pb && peer_bs) {
                CHECK_CUDA(cudaSetDevice(1));
                CHECK_CUDA(cudaMemset(d_peer_prog, 0, sizeof(unsigned long long)));
                const unsigned long long pdl =
                    (unsigned long long)window_ms * (unsigned long long)prop1_clock_khz;
                CHECK_CUDA(cudaEventRecord(pe0, s_peer));
                nvhbi_peer_write<<<pb, peer_bs, 0, s_peer>>>(
                    t.d_data, d_peer_idx, peer_first, peer_chunks, 0u, pdl,
                    peer_idle, d_peer_prog, d_sink1);
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
            if (bg_r_sms)
                CHECK_CUDA(cudaMemcpy(&bg_r1, d_bg_rprog, sizeof(bg_r1), cudaMemcpyDeviceToHost));
            unsigned long long bg_cyc = 0ull;
            if (bg_sms) {
                nvhbi_stop_flag_set(stop);
                CHECK_CUDA(cudaStreamSynchronize(s_bg));
                CHECK_CUDA(cudaMemcpy(&bg_cyc, d_bg_cyc, sizeof(bg_cyc),
                                      cudaMemcpyDeviceToHost));
            }
            // Clock the background actually achieved. Measure the span from the
            // host, launch to drain: warps only notice the stop flag at a poll
            // boundary, so they overrun it, and a hard-coded span would inflate
            // this. (With NVHBI_POLL_MASK=65535 the overrun reached ~267 ms
            // against an assumed 300 ms span and reported 2.1 GHz.)
            const double bg_span_ms = (bg_launch_ms > 0.0) ? (now_ms() - bg_launch_ms) : 0.0;
            const double bg_ghz = (bg_sms && bg_cyc && bg_span_ms > 0.0)
                ? (double)bg_cyc / (bg_span_ms * 1e6) : 0.0;

            const double bg_alive_ms = (double)(4u * window_ms + 2000u) - 100.0;
            if (bg_sms && wall_ms > bg_alive_ms)
                fprintf(stderr, "WARNING: peer ran %.0f ms but background deadline was %.0f ms "
                                "-- bg_GBps is understated\n", wall_ms, bg_alive_ms);

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
            const double bg_gbps = (bg_sms && wall_ms > 0.0)
                ? (double)(bg_p1 - bg_p0) * 32.0 / (wall_ms * 1e-3) / 1e9 : 0.0;
            const double bg_rd_gbps = (bg_r_sms && wall_ms > 0.0)
                ? (double)(bg_r1 - bg_r0) * 32.0 / (wall_ms * 1e-3) / 1e9 : 0.0;
            // Everything the near->far direction is carrying: the background's
            // writes, its crossing reads, and the peer when it targets the far
            // die. bg_local=1 crosses with neither payload.
            const double crossing = (bg_local ? 0.0 : bg_gbps + bg_rd_gbps)
                                  + ((exp == 2u) ? peer_gbps : 0.0);

            printf("CFG,%u,%u,%u,%u,%u,%u,%u,%.4f,%.2f,%.2f,%.2f,%.3f,%u,%u,%.2f,%u\n",
                   exp, far_die, peer_die, bg_local, bg_sms,
                   peer_bs ? pb : 0u, rep,
                   peer_ms, peer_gbps, bg_gbps, crossing, bg_ghz,
                   peer_ovl ? 1u : 0u, peer_bs, bg_rd_gbps, bg_r_sms);
            fflush(stdout);
        }
    }}

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
    if (d_dma_dst) CHECK_CUDA(cudaFree(d_dma_dst));
    CHECK_CUDA(cudaFree(d_bg_cyc));
    nvhbi_stop_flag_destroy(stop);
    nvhbi_free(t);
    return 0;
}
