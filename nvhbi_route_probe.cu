// nvhbi_route_probe.cu -- where does a peer write actually land?
//
// EXPERIMENT 2 assumes: GPU1 --NVLink--> GPU0's NVLink die --NV-HBI--> the far
// die's L2. Everything exp2 concludes rests on that chain, and exp2 cannot test
// it, because the only thing it observes is a bandwidth number that comes out
// the same under every hypothesis. (It has to: NV-HBI is 10 TB/s BIDIRECTIONAL,
// so ~5 TB/s per direction, while one die's 74 SMs cannot push past ~4.0 TB/s
// even writing to their OWN die -- exp1's own_die control. The link is never
// close to full, so 630 GB/s of peer traffic displaces nothing whether or not
// it shares the link.)
//
// So test the chain directly, one link at a time, with instruments that give a
// yes/no rather than a number to squint at.
//
//   PART B  warm-up coverage. Does nvhbi_warm_chunks actually load all `count`
//           chunks it is handed? It filters SMs by die (correct) but splits the
//           work by GLOBAL warp index (not), so every chunk whose warp landed
//           on the other die is skipped. Printed as a percentage against a
//           work-queue version that cannot miss. Everything downstream depends
//           on the target lines really being resident, so this goes first.
//
//   PART C  attachment. Atomic latency, GPU0-internal (the ~300 cycle NV-HBI
//           hop, re-measured here so the comparison is within one run) and
//           GPU1->GPU0, taken from an SM on EACH die of BOTH GPUs. The full
//           2x2x2 is the point: if NVLink lands on one die of each GPU, the
//           table must be ADDITIVE -- one ~300 cycle step along the GPU0 axis
//           and one along the GPU1 axis. A single gap could be many things; two
//           independent gaps of the same size as the internal hop is a
//           signature. (The old calibration used smid 0 and 147 and got only
//           -71/-135 on the GPU1 axis, which says those two SMs were probably
//           on the same GPU1 die -- that axis was never actually measured.)
//
//   PART D  THE POINT. GPU1 stamps a deterministic, address-derived pattern
//           into the far die's chunks. Then GPU0:
//             (1) reads it back and counts mismatches
//                 -> proves the bytes reached those addresses at all, and that
//                    the die map GPU1 was given is the right one.
//             (2) times a FIRST-TOUCH read of those lines, from an SM on the
//                 home die and from an SM on the other die, against a ladder
//                 measured with the identical instrument in the same run:
//                      cold (flushed, no warm)      -> the HBM reference
//                      warm                          -> the L2 reference
//                      warm + a GPU0 local store     -> L2, known dirty
//                      warm + the GPU1 peer store    -> the unknown
//                 Calibration from earlier work: local L2 read 260-300 cyc,
//                 local HBM read 560-620 cyc. If "warm + peer store" sits at
//                 the L2 reference the peer write landed in the home die's L2;
//                 if it sits at the HBM reference the peer write pushed the
//                 line out to memory and exp2 has not been measuring remote-L2
//                 writes at all.
//
//   PART E  optional (NVHBI_RP_PART_E=1): peer latency while GPU0 holds a
//           background load, in three modes that separate the link from the
//           far die's L2. Only meaningful once PART C has named the attach
//           die, so it refuses to run if PART C was inconclusive.
//
// Reads nothing and writes nothing that the experiment programs use.
//
// Env:
//   NVHBI_RP_CHUNKS      chunks per measurement region,   default 256
//   NVHBI_RP_LAT_CHUNKS  accesses per timed read chain,   default 64
//   NVHBI_RP_REPS        timed chains per atomic probe,   default 200
//   NVHBI_RP_WARMUP      warm passes before atomic timing,default 64
//   NVHBI_RP_ROUNDS      alternating rounds (min taken),  default 4
//   NVHBI_RP_TRIALS      repeats of the PART D ladder,    default 5
//   NVHBI_RP_PART_E      1 = also run the contention part, default 0
//   NVHBI_RP_BG_SMS      PART E background SM counts,     default "0,32,64,74"
//   NVHBI_RP_BG_BLOCK / NVHBI_RP_BG_BLOCKS_PER_SM         default 64 / 32
//   NVHBI_RP_SETTLE_MS   settle after a bg launch,        default 100
//   NVHBI_FAR_DIE        override the near/far decision (0 or 1)
//   NVHBI_BUF_MULT       default 8

#include "nvhbi_common.cuh"
#include <chrono>

/* ---------------------------------------------------------------- helpers */

static unsigned int env_u(const char* k, unsigned int d) {
    const char* s = getenv(k);
    return (s && *s) ? (unsigned int)atoi(s) : d;
}
static int has_env(const char* k) { const char* s = getenv(k); return s && *s; }

static int parse_list(const char* env, const char* dflt, unsigned int* out, int cap) {
    const char* s = getenv(env);
    if (!s || !*s) s = dflt;
    int n = 0;
    char buf[256];
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

/* ---------------------------------------------------------------- kernels */

// Records, for every block of a full-occupancy grid, which SM it landed on and
// what the OLD code would have computed as its per-SM slot (blockIdx.x /
// sm_count). The host then counts, per SM, how many distinct slots that yields.
// 32 means the old assumption held; 1 means all 32 resident blocks on that SM
// would have been given the same slot -- and with it the same 4 KiB chunk.
__global__ void rp_block_map(unsigned int* __restrict__ smid_of_block) {
    if (threadIdx.x == 0) smid_of_block[blockIdx.x] = nvhbi_smid();
}

// The value that belongs at absolute word index `w` for a given tag. Derived
// from the ADDRESS, so a mismatch tells us not just "wrong value" but that the
// write went somewhere other than where we aimed it.
__device__ __host__ __forceinline__
unsigned int rp_pattern(unsigned int w, unsigned int tag) {
    return (tag * 0x9E3779B9u) ^ (w * 2654435761u) ^ 0xC0DE0000u;
}

// Warm-up, work-queue form. Warps pull chunk ids from a shared cursor instead
// of deriving them from their global index, so coverage is complete whichever
// SMs the die filter left running and however blocks happened to land. One
// atomic per chunk is nothing next to the four L2 loads it guards.
__global__ void rp_warm_queue(unsigned int* __restrict__ data,
                              const unsigned int* __restrict__ idx_list,
                              unsigned int first, unsigned int count,
                              const unsigned int* __restrict__ sm_side,
                              unsigned int owner_partition,
                              unsigned int* __restrict__ cursor,
                              unsigned int* __restrict__ visited,
                              unsigned int* __restrict__ sink) {
    const unsigned int smid = nvhbi_smid();
    if ((sm_side[smid] % 2u) != owner_partition) return;

    const unsigned int lane = threadIdx.x % 32u;
    unsigned int consume = 0u;
    for (;;) {
        unsigned int c = 0u;
        if (lane == 0u) c = atomicAdd(cursor, 1u);
        c = __shfl_sync(0xffffffffu, c, 0);
        if (c >= count) break;
        unsigned int* a[4];
        nvhbi_lane_addrs(data, idx_list[first + c], lane, a);
        consume += nvhbi_ld(a[0]); consume += nvhbi_ld(a[1]);
        consume += nvhbi_ld(a[2]); consume += nvhbi_ld(a[3]);
        if (visited && lane == 0u) visited[c] = 1u;
    }
    nvhbi_st(&sink[smid], consume);
}

// The OLD nvhbi_warm_chunks indexing, kept here on purpose as the A/B control
// after the header switched to a work queue. Same die filter, plus a visited
// flag. Whatever this reports below 100% is what every measurement taken before
// that change was actually getting.
__global__ void rp_warm_asis(unsigned int* __restrict__ data,
                             const unsigned int* __restrict__ idx_list,
                             unsigned int first, unsigned int count,
                             const unsigned int* __restrict__ sm_side,
                             unsigned int owner_partition,
                             unsigned int* __restrict__ visited,
                             unsigned int* __restrict__ sink) {
    const unsigned int smid = nvhbi_smid();
    if ((sm_side[smid] % 2u) != owner_partition) return;

    const unsigned int lane   = threadIdx.x % 32u;
    const unsigned int gwarp  = (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    const unsigned int nwarps = (gridDim.x * blockDim.x) / 32u;

    unsigned int consume = 0u;
    for (unsigned int c = gwarp; c < count; c += nwarps) {
        unsigned int* a[4];
        nvhbi_lane_addrs(data, idx_list[first + c], lane, a);
        consume += nvhbi_ld(a[0]); consume += nvhbi_ld(a[1]);
        consume += nvhbi_ld(a[2]); consume += nvhbi_ld(a[3]);
        if (lane == 0u) visited[c] = 1u;
    }
    nvhbi_st(&sink[smid], consume);
}

// Stamp the pattern into exactly the words nvhbi_store_group touches: 4 sectors
// per lane, a lane quad covering one 128B line, a warp covering a whole 4KiB
// chunk at 32B granularity. Same store shape as exp2's peer kernel, so this
// tests the traffic exp2 actually generates.
//
// `sm_side == nullptr` means no die filter -- that is the form used on GPU1,
// where the caller has no map of the target GPU's SMs and does not need one.
__global__ void rp_stamp(unsigned int* __restrict__ data,
                         const unsigned int* __restrict__ idx_list,
                         unsigned int first, unsigned int count,
                         unsigned int tag,
                         const unsigned int* __restrict__ sm_side,
                         unsigned int owner_partition,
                         unsigned int* __restrict__ cursor,
                         unsigned int* __restrict__ sink) {
    const unsigned int smid = nvhbi_smid();
    if (sm_side && (sm_side[smid] % 2u) != owner_partition) return;

    const unsigned int lane = threadIdx.x % 32u;
    unsigned int last = 0u;
    for (;;) {
        unsigned int c = 0u;
        if (lane == 0u) c = atomicAdd(cursor, 1u);
        c = __shfl_sync(0xffffffffu, c, 0);
        if (c >= count) break;
        const unsigned int cidx = idx_list[first + c];
        const unsigned int base = cidx + 128u * (lane / 4u) + 8u * (lane % 4u);
        unsigned int* a[4];
        nvhbi_lane_addrs(data, cidx, lane, a);
        nvhbi_st(a[0], rp_pattern(base,        tag));
        nvhbi_st(a[1], rp_pattern(base + 32u,  tag));
        nvhbi_st(a[2], rp_pattern(base + 64u,  tag));
        nvhbi_st(a[3], rp_pattern(base + 96u,  tag));
        last = base;
    }
    nvhbi_st(&sink[smid], last);
}

// Read back every stamped word and count the ones that do not match. No die
// filter and no gwarp/participation mismatch here: every launched warp works,
// so the whole region really is checked.
__global__ void rp_verify(const unsigned int* __restrict__ data,
                          const unsigned int* __restrict__ idx_list,
                          unsigned int first, unsigned int count,
                          unsigned int tag,
                          unsigned int* __restrict__ bad,
                          unsigned int* __restrict__ sink) {
    const unsigned int wpb    = (blockDim.x + 31u) / 32u;
    const unsigned int wib    = threadIdx.x / 32u;
    const unsigned int lane   = threadIdx.x % 32u;
    const unsigned int gwarp  = blockIdx.x * wpb + wib;
    const unsigned int nwarps = gridDim.x * wpb;
    if (!nwarps) return;

    unsigned int miss = 0u, seen = 0u;
    for (unsigned int c = gwarp; c < count; c += nwarps) {
        const unsigned int cidx = idx_list[first + c];
        const unsigned int base = cidx + 128u * (lane / 4u) + 8u * (lane % 4u);
        for (unsigned int k = 0; k < 4u; ++k) {
            const unsigned int w = base + 32u * k;
            // .cv: never let a local replica answer for the home copy.
            unsigned int got;
            asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(got)
                         : "l"(&data[w]));
            miss += (got != rp_pattern(w, tag));
            ++seen;
        }
    }
    if (miss) atomicAdd(bad, miss);
    nvhbi_st(&sink[nvhbi_smid()], seen);
}

/* ------------------------------------------------------------------- main */

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    const unsigned int region      = env_u("NVHBI_RP_CHUNKS", 256u);
    const unsigned int lat_chunks  = env_u("NVHBI_RP_LAT_CHUNKS", 64u);
    const unsigned int lat_reps    = env_u("NVHBI_RP_REPS", 200u);
    const unsigned int lat_warm    = env_u("NVHBI_RP_WARMUP", 64u);
    const unsigned int lat_rounds  = env_u("NVHBI_RP_ROUNDS", 4u);
    const unsigned int trials      = env_u("NVHBI_RP_TRIALS", 5u);
    const unsigned int do_part_e   = env_u("NVHBI_RP_PART_E", 0u);
    const unsigned int bg_block    = env_u("NVHBI_RP_BG_BLOCK", 64u);
    // 31, not 32, on purpose: PART E has to schedule a one-thread probe kernel
    // onto an SM the background already occupies, and at 32 blocks/SM every
    // block slot is taken and the probe would queue behind the whole background
    // instead of running beside it. 31 x 64 = 1984 threads leaves one slot.
    const unsigned int bg_nbps     = env_u("NVHBI_RP_BG_BLOCKS_PER_SM", 31u);
    const unsigned int settle_ms   = env_u("NVHBI_RP_SETTLE_MS", 100u);
    const double       buf_mult    = (double)env_u("NVHBI_BUF_MULT", 8u);

    unsigned int bg_list[16];
    const int bg_n = parse_list("NVHBI_RP_BG_SMS", "0,32,64,74", bg_list, 16);

    int ndev = 0;
    CHECK_CUDA(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { fprintf(stderr, "ERROR: need 2 GPUs, found %d\n", ndev); return 1; }

    /* ============================ PART A: topology ======================== */

    NvhbiTopo t0, t1;
    nvhbi_probe(t0, 0, buf_mult);
    nvhbi_probe(t1, 1, 1.0);        // small buffer: only its sm_side map is wanted

    CHECK_CUDA(cudaSetDevice(0));
    unsigned int* h_side0 = (unsigned int*)malloc(t0.sm_count * sizeof(unsigned int));
    CHECK_CUDA(cudaMemcpy(h_side0, t0.d_sm_side, t0.sm_count * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaSetDevice(1));
    unsigned int* h_side1 = (unsigned int*)malloc(t1.sm_count * sizeof(unsigned int));
    CHECK_CUDA(cudaMemcpy(h_side1, t1.d_sm_side, t1.sm_count * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    // smid -> die is an irregular interleave, so "smid 0 and smid last" is not a
    // way to get one SM per die. Take the map at its word.
    auto pick_sm = [](const unsigned int* side, int n, unsigned int part) -> int {
        for (int i = 0; i < n; ++i) if ((side[i] % 2u) == part) return i;
        return -1;
    };
    const int sm0_of[2] = { pick_sm(h_side0, t0.sm_count, 0u),
                            pick_sm(h_side0, t0.sm_count, 1u) };
    const int sm1_of[2] = { pick_sm(h_side1, t1.sm_count, 0u),
                            pick_sm(h_side1, t1.sm_count, 1u) };
    if (sm0_of[0] < 0 || sm0_of[1] < 0 || sm1_of[0] < 0 || sm1_of[1] < 0) {
        fprintf(stderr, "ERROR: a GPU reported SMs on only one side; the latency "
                        "threshold is wrong and nothing below would mean anything\n");
        return 1;
    }
    printf("\nprobe SMs   GPU0: die0=sm%-3d die1=sm%-3d   GPU1: die0=sm%-3d die1=sm%-3d\n",
           sm0_of[0], sm0_of[1], sm1_of[0], sm1_of[1]);

    int can = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can, 1, 0));
    if (!can) { fprintf(stderr, "ERROR: GPU1 cannot peer-access GPU0\n"); return 1; }
    CHECK_CUDA(cudaSetDevice(1));
    cudaError_t pe = cudaDeviceEnablePeerAccess(0, 0);
    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) CHECK_CUDA(pe);

    // GPU1-resident copies of GPU0's per-die chunk lists.
    unsigned int* list_on1[2] = {nullptr, nullptr};
    CHECK_CUDA(cudaMalloc(&list_on1[0], t0.near_count * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&list_on1[1], t0.far_count  * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemcpy(list_on1[0], t0.h_near_idx, t0.near_count * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(list_on1[1], t0.h_far_idx,  t0.far_count  * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    unsigned int *d_out1a = nullptr, *d_out1b = nullptr, *d_cursor1 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out1a, sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_out1b, sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_cursor1, sizeof(unsigned int)));

    CHECK_CUDA(cudaSetDevice(0));
    unsigned int *d_out0a = nullptr, *d_out0b = nullptr, *d_bad = nullptr, *d_cursor = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out0a, sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_out0b, sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_bad,   sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_cursor, sizeof(unsigned int)));

    unsigned int* list0[2]        = { t0.d_near_idx, t0.d_far_idx };
    const unsigned int count0[2]  = { t0.near_count, t0.far_count };

    // The measurement region sits past anything PART E's background can touch
    // (74 SMs x 32 blocks x 2 warps = 4736 chunks), so a latency reading is
    // never just contention for the very lines the background is hammering.
    const unsigned int base_chunk = 8192u;
    if (count0[0] < base_chunk + region || count0[1] < base_chunk + region) {
        fprintf(stderr, "ERROR: only %u/%u chunks per die; raise NVHBI_BUF_MULT\n",
                count0[0], count0[1]);
        return 1;
    }
    if (lat_chunks > region) {
        fprintf(stderr, "ERROR: NVHBI_RP_LAT_CHUNKS (%u) exceeds NVHBI_RP_CHUNKS (%u)\n",
                lat_chunks, region);
        return 1;
    }

    /* ================== PART B: does the warm-up cover the region? ================== */
    {
        // The configuration exp1/exp2 run at the top of their sweeps.
        const unsigned int nsm    = t0.sms_p0;
        const unsigned int count  = nvhbi_chunks_used(nsm, bg_nbps, bg_block, 1u);
        const unsigned int target = 1u;
        unsigned int* d_vis = nullptr;
        CHECK_CUDA(cudaMalloc(&d_vis, count * sizeof(unsigned int)));
        unsigned int* h_vis = (unsigned int*)malloc(count * sizeof(unsigned int));

        auto coverage = [&](bool queue) -> double {
            CHECK_CUDA(cudaMemset(d_vis, 0, count * sizeof(unsigned int)));
            CHECK_CUDA(cudaMemset(d_cursor, 0, sizeof(unsigned int)));
            if (queue)
                rp_warm_queue<<<t0.sm_count * 8, 128>>>(
                    t0.d_data, list0[target], 0u, count, t0.d_sm_side, target,
                    d_cursor, d_vis, t0.d_sink);
            else
                rp_warm_asis<<<t0.sm_count * 8, 128>>>(
                    t0.d_data, list0[target], 0u, count, t0.d_sm_side, target,
                    d_vis, t0.d_sink);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(h_vis, d_vis, count * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
            unsigned int hit = 0;
            for (unsigned int i = 0; i < count; ++i) hit += (h_vis[i] != 0u);
            return 100.0 * hit / count;
        };

        printf("\n================= PART B: warm-up coverage =================\n");
        printf("die%u, %u chunks (%u SMs x %u blk/SM x %u thr -- exp1/exp2's top point)\n",
               target, count, nsm, bg_nbps, bg_block);
        printf("  old indexing (gwarp split)        : %6.1f%%\n", coverage(false));
        printf("  work-queue split (now in the header): %5.1f%%\n", coverage(true));
        printf("  Both filter SMs by die identically. They differ only in how the\n"
               "  chunk list is split across warps. Any gap is chunks the stress\n"
               "  kernel writes into but nothing ever pulled into L2 -- and under\n"
               "  write-no-allocate those stores go to HBM, not to the remote L2.\n");
        free(h_vis);
        CHECK_CUDA(cudaFree(d_vis));
    }

    /* ====== PART B2: does block b really land on SM (b % sm_count)? ====== */
    {
        const unsigned int nbps_chk = bg_nbps;                 // 32, as the sweeps use
        const unsigned int nblk = (unsigned int)t0.sm_count * nbps_chk;
        unsigned int* d_map = nullptr;
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaMalloc(&d_map, nblk * sizeof(unsigned int)));
        CHECK_CUDA(cudaMemset(d_map, 0xff, nblk * sizeof(unsigned int)));
        // 64 threads/block x 32 blocks/SM = 2048 = full occupancy, exactly the
        // shape the stress kernels launch, so the distribution is the same one.
        rp_block_map<<<nblk, bg_block>>>(d_map);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        unsigned int* h_map = (unsigned int*)malloc(nblk * sizeof(unsigned int));
        CHECK_CUDA(cudaMemcpy(h_map, d_map, nblk * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        // per SM: how many DISTINCT values of (blockIdx.x / sm_count) occur
        unsigned int worst = ~0u, best = 0u; double avg = 0.0; int nsm_seen = 0;
        for (int sm = 0; sm < t0.sm_count; ++sm) {
            bool seen[64] = {false};
            unsigned int nblk_here = 0, distinct = 0;
            for (unsigned int b = 0; b < nblk; ++b) {
                if (h_map[b] != (unsigned int)sm) continue;
                ++nblk_here;
                const unsigned int q = b / (unsigned int)t0.sm_count;
                if (q < 64u && !seen[q]) { seen[q] = true; ++distinct; }
            }
            if (!nblk_here) continue;
            ++nsm_seen; avg += distinct;
            if (distinct < worst) worst = distinct;
            if (distinct > best)  best  = distinct;
        }
        avg /= (nsm_seen ? nsm_seen : 1);

        printf("\n====== PART B2: block -> SM slot assignment ======\n");
        printf("grid %u blocks x %u threads (full occupancy, %u blocks/SM)\n",
               nblk, bg_block, nbps_chk);
        printf("  distinct values of (blockIdx.x / sm_count) per SM:"
               "  min %u, max %u, mean %.1f   (want %u)\n",
               worst, best, avg, nbps_chk);
        if (best < nbps_chk)
            printf("  => the old slot formula COLLIDES. %u blocks per SM were sharing\n"
                   "     as few as %u slots, so that many warps were pointed at the\n"
                   "     SAME 4 KiB chunk. Writers only lose footprint; readers get\n"
                   "     L2 hits counted as fresh fetches, which is the read\n"
                   "     overcount. The kernels now take a per-SM ticket instead.\n",
                   nbps_chk, worst);
        else
            printf("  => the old formula happened to hold here. The per-SM ticket the\n"
                   "     kernels now use gives the same answer, without depending on it.\n");
        free(h_map);
        CHECK_CUDA(cudaFree(d_map));
    }

    /* ============= PART C: attachment, from the full 2x2x2 of dies ============= */

    // One atomic-latency sample. Atomics always travel to the home L2 slice, so
    // no local replica can answer, which is what makes them a topology probe.
    auto atomic_lat = [&](int dev, unsigned int* data, const unsigned int* idx,
                          int probe_smid) -> unsigned int {
        CHECK_CUDA(cudaSetDevice(dev));
        const int     nsm = (dev == 0) ? t0.sm_count : t1.sm_count;
        unsigned int* snk = (dev == 0) ? t0.d_sink   : t1.d_sink;
        unsigned int* oa  = (dev == 0) ? d_out0a     : d_out1a;
        unsigned int* ob  = (dev == 0) ? d_out0b     : d_out1b;
        unsigned int best = ~0u;
        for (unsigned int r = 0; r < lat_rounds; ++r) {
            nvhbi_peer_latency<<<nsm, 1>>>(data, idx, base_chunk, 16u,
                                           lat_reps, lat_warm,
                                           (unsigned int)probe_smid, oa, ob, snk);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            unsigned int v = 0u;
            CHECK_CUDA(cudaMemcpy(&v, oa, sizeof(v), cudaMemcpyDeviceToHost));
            if (v < best) best = v;
        }
        return best;
    };

    unsigned int L0[2][2], L1[2][2];      // [issuing die][GPU0 target die]
    for (int i = 0; i < 2; ++i)
        for (int d = 0; d < 2; ++d) {
            L0[i][d] = atomic_lat(0, t0.d_data, list0[d],    sm0_of[i]);
            L1[i][d] = atomic_lat(1, t0.d_data, list_on1[d], sm1_of[i]);
        }

    const int hop0_from0 = (int)L0[0][1] - (int)L0[0][0];   // GPU0 die0 SM: far - near
    const int hop0_from1 = (int)L0[1][0] - (int)L0[1][1];
    const double H_internal = 0.5 * (hop0_from0 + hop0_from1);
    const int hopT_from0 = (int)L1[0][1] - (int)L1[0][0];   // GPU0 axis, seen from GPU1 die0
    const int hopT_from1 = (int)L1[1][1] - (int)L1[1][0];
    const int hopI_to0   = (int)L1[1][0] - (int)L1[0][0];   // GPU1 axis
    const int hopI_to1   = (int)L1[1][1] - (int)L1[0][1];
    const double H_target = 0.5 * (hopT_from0 + hopT_from1);
    const double H_issuer = 0.5 * (hopI_to0  + hopI_to1);

    printf("\n=============== PART C: attachment ===============\n");
    printf("min atomic latency, cycles\n");
    printf("  GPU0 internal          -> die0    -> die1   | GPU0 hop\n");
    printf("    from GPU0 sm%-3d(die0)  %6u    %6u   | %+d\n",
           sm0_of[0], L0[0][0], L0[0][1], hop0_from0);
    printf("    from GPU0 sm%-3d(die1)  %6u    %6u   | %+d\n",
           sm0_of[1], L0[1][0], L0[1][1], -hop0_from1);
    printf("    => H_internal (one NV-HBI hop) = %.0f cycles\n", H_internal);
    printf("  GPU1 over NVLink       -> die0    -> die1   | GPU0 hop\n");
    printf("    from GPU1 sm%-3d(die0)  %6u    %6u   | %+d\n",
           sm1_of[0], L1[0][0], L1[0][1], hopT_from0);
    printf("    from GPU1 sm%-3d(die1)  %6u    %6u   | %+d\n",
           sm1_of[1], L1[1][0], L1[1][1], hopT_from1);
    printf("    GPU1 hop (its die1 - its die0):  %+d      %+d\n", hopI_to0, hopI_to1);
    printf("    => H_target = %.0f cyc (GPU0 axis), H_issuer = %.0f cyc (GPU1 axis)\n",
           H_target, H_issuer);

    unsigned int far_die;
    const char* how;
    if (has_env("NVHBI_FAR_DIE")) {
        far_die = env_u("NVHBI_FAR_DIE", 1u) ? 1u : 0u; how = "forced by NVHBI_FAR_DIE";
    } else {
        far_die = (H_target >= 0.0) ? 1u : 0u;          how = "auto: slower from GPU1";
    }
    const unsigned int near_die = 1u - far_die;
    bool attach_clear = false;
    {
        const double rt = (H_internal != 0.0) ? fabs(H_target) / fabs(H_internal) : 0.0;
        const double ri = (H_internal != 0.0) ? fabs(H_issuer) / fabs(H_internal) : 0.0;
        printf("  H_target/H_internal = %.2f    H_issuer/H_internal = %.2f\n", rt, ri);
        if (rt < 0.3) {
            printf("  VERDICT: no extra hop on the GPU0 axis. NVLink reaches both GPU0\n"
                   "           dies without NV-HBI, and exp2 vs exp3 isolates nothing.\n");
        } else if (rt > 0.6 && rt < 1.6) {
            attach_clear = true;
            printf("  VERDICT: GPU0 axis = one clean NV-HBI hop. NVLink is die-attached\n"
                   "           on GPU0; NEAR = die%u, FAR = die%u  [%s]\n",
                   near_die, far_die, how);
            if (ri > 0.6 && ri < 1.6)
                printf("           GPU1 axis shows the same size step -- both GPUs are\n"
                       "           die-attached and the table is additive, which is the\n"
                       "           signature you want before trusting PART E.\n");
            else
                printf("           GPU1 axis is %.2f of a hop, so the ISSUING side may not\n"
                       "           be die-attached (or both probe SMs sit on the attached\n"
                       "           die). Pin GPU1's SM deliberately in exp2.\n", ri);
        } else {
            printf("  VERDICT: GPU0 axis is %.2f of a hop -- not one clean hop. Do not\n"
                   "           call this routing until PART D agrees.\n", rt);
        }
    }

    /* ============ PART D: where does the peer write actually land? ============ */

    // One timed FIRST-TOUCH read chain: dependent loads, one thread, pinned SM,
    // .cg so L1 is out of the path and L2 is not. Single pass over `lat_chunks`
    // distinct chunks, so nothing it reads was warmed by the chain itself.
    auto read_lat = [&](int reader_die, unsigned int target_die) -> unsigned int {
        CHECK_CUDA(cudaSetDevice(0));
        nvhbi_peer_load_latency<<<t0.sm_count, 1>>>(
            t0.d_data, list0[target_die], base_chunk, lat_chunks,
            (unsigned int)sm0_of[reader_die], /*cv=*/0u, d_out0a, t0.d_sink);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        unsigned int v = 0u;
        CHECK_CUDA(cudaMemcpy(&v, d_out0a, sizeof(v), cudaMemcpyDeviceToHost));
        return v;
    };

    auto warm_region = [&](unsigned int target_die) {
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaMemset(d_cursor, 0, sizeof(unsigned int)));
        rp_warm_queue<<<t0.sm_count * 8, 128>>>(
            t0.d_data, list0[target_die], base_chunk, region,
            t0.d_sm_side, target_die, d_cursor, nullptr, t0.d_sink);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    };

    enum Prep { PREP_COLD = 0, PREP_WARM, PREP_LOCAL_WRITE, PREP_PEER_WRITE };
    static const char* prep_name[4] = {
        "cold  (flushed, no warm)   ", "warm  (home-die loads)     ",
        "warm + GPU0 local store    ", "warm + GPU1 PEER store     " };
    const unsigned int TAG_LOCAL = 0x11u, TAG_PEER = 0x22u;

    // Runs one ladder point end to end and returns the two read latencies plus,
    // for the peer case, the mismatch count from the read-back.
    auto ladder_point = [&](unsigned int target_die, Prep prep,
                            unsigned int* lat_home, unsigned int* lat_other,
                            unsigned int* bad_out) {
        CHECK_CUDA(cudaSetDevice(0));
        nvhbi_flush_l2(t0);
        if (prep != PREP_COLD) warm_region(target_die);

        if (prep == PREP_LOCAL_WRITE) {
            CHECK_CUDA(cudaMemset(d_cursor, 0, sizeof(unsigned int)));
            rp_stamp<<<t0.sm_count * 8, 128>>>(
                t0.d_data, list0[target_die], base_chunk, region, TAG_LOCAL,
                t0.d_sm_side, target_die, d_cursor, t0.d_sink);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        } else if (prep == PREP_PEER_WRITE) {
            CHECK_CUDA(cudaSetDevice(1));
            CHECK_CUDA(cudaMemset(d_cursor1, 0, sizeof(unsigned int)));
            rp_stamp<<<t1.sm_count * 4, 128>>>(
                t0.d_data, list_on1[target_die], base_chunk, region, TAG_PEER,
                /*sm_side=*/nullptr, 0u, d_cursor1, t1.d_sink);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaSetDevice(0));
        }

        // Timed reads BEFORE the verify pass, or the verify would warm
        // everything it touches. Home die first, then the other die -- the
        // other-die read is a first touch of different chunks only if we let
        // the home read run first, so re-flush and re-prep between them.
        *lat_home = read_lat((int)target_die, target_die);

        nvhbi_flush_l2(t0);
        if (prep != PREP_COLD) warm_region(target_die);
        if (prep == PREP_LOCAL_WRITE) {
            CHECK_CUDA(cudaMemset(d_cursor, 0, sizeof(unsigned int)));
            rp_stamp<<<t0.sm_count * 8, 128>>>(
                t0.d_data, list0[target_die], base_chunk, region, TAG_LOCAL,
                t0.d_sm_side, target_die, d_cursor, t0.d_sink);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else if (prep == PREP_PEER_WRITE) {
            CHECK_CUDA(cudaSetDevice(1));
            CHECK_CUDA(cudaMemset(d_cursor1, 0, sizeof(unsigned int)));
            rp_stamp<<<t1.sm_count * 4, 128>>>(
                t0.d_data, list_on1[target_die], base_chunk, region, TAG_PEER,
                nullptr, 0u, d_cursor1, t1.d_sink);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaSetDevice(0));
        }
        *lat_other = read_lat((int)(1u - target_die), target_die);

        *bad_out = 0u;
        if (prep == PREP_LOCAL_WRITE || prep == PREP_PEER_WRITE) {
            CHECK_CUDA(cudaSetDevice(0));
            CHECK_CUDA(cudaMemset(d_bad, 0, sizeof(unsigned int)));
            rp_verify<<<t0.sm_count * 8, 128>>>(
                t0.d_data, list0[target_die], base_chunk, region,
                (prep == PREP_PEER_WRITE) ? TAG_PEER : TAG_LOCAL,
                d_bad, t0.d_sink);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(bad_out, d_bad, sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }
    };

    printf("\n============ PART D: where the peer write lands ============\n");
    printf("region: %u chunks (%.1f MB) at chunk offset %u, %u words checked per chunk\n",
           region, region * (double)NVHBI_CHUNK_BYTES / 1048576.0, base_chunk, 128u);
    printf("read chain: %u first-touch accesses, ld.global.cg, single thread\n", lat_chunks);
    printf("reference from earlier work: local L2 read 260-300 cyc, HBM read 560-620 cyc\n\n");

    for (unsigned int d = 0; d < 2u; ++d) {
        const unsigned int target = (d == 0u) ? near_die : far_die;
        printf("  ---- target die%u (%s) ----\n", target,
               (target == far_die) ? "FAR: exp2's destination" : "NEAR: exp3's destination");
        printf("  %-28s  read from die%u   read from die%u   mismatches\n",
               "prepared as", target, 1u - target);
        for (int p = 0; p < 4; ++p) {
            unsigned int bh = ~0u, bo = ~0u, bad = 0u, badmax = 0u;
            for (unsigned int r = 0; r < trials; ++r) {
                unsigned int lh, lo, b;
                ladder_point(target, (Prep)p, &lh, &lo, &b);
                if (lh < bh) bh = lh;
                if (lo < bo) bo = lo;
                if (b > badmax) badmax = b;
            }
            bad = badmax;
            printf("  %-28s      %6u           %6u        %s\n",
                   prep_name[p], bh, bo,
                   (p == PREP_LOCAL_WRITE || p == PREP_PEER_WRITE)
                       ? (bad ? "FAIL" : "0 (ok)") : "-");
            if ((p == PREP_LOCAL_WRITE || p == PREP_PEER_WRITE) && bad)
                printf("      *** %u words wrong. The store did not reach these addresses.\n",
                       bad);
        }
        printf("\n");
    }
    printf("How to read PART D\n"
           "------------------\n"
           "mismatches = 0 for the PEER row proves GPU1's stores reached exactly the\n"
           "addresses the die map says belong to that die -- the destination is right.\n"
           "Then the two latency columns say where the line ended up:\n"
           "  peer row ~ the 'warm' row            -> the peer write left the line in\n"
           "                                          that die's L2. exp2 really is\n"
           "                                          measuring remote-L2 writes.\n"
           "  peer row ~ the 'cold' row            -> the peer write went to HBM. exp2\n"
           "                                          has been measuring an HBM write\n"
           "                                          path, and its bandwidth number is\n"
           "                                          not a remote-L2 number.\n"
           "  peer row between the two             -> partial residency; raise\n"
           "                                          NVHBI_RP_TRIALS and check whether\n"
           "                                          the region fits the die's L2.\n"
           "The other-die column should exceed the home-die column by about H_internal\n"
           "in every row; that is the same NV-HBI hop PART C measured, seen from the\n"
           "read side, and it is the internal consistency check on the whole table.\n");

    /* ================ PART E: contention (optional) ================ */
    if (do_part_e) {
        if (!attach_clear && !has_env("NVHBI_FAR_DIE")) {
            printf("\nPART E skipped: PART C did not identify an attach die. Settle that\n"
                   "first (or force it with NVHBI_FAR_DIE) -- otherwise 'near' and 'far'\n"
                   "are just labels and the contrast means nothing.\n");
        } else {
            struct BgMode { const char* name; unsigned int wp; unsigned int own; };
            const BgMode modes[3] = {
                { "A->B cross (link + farL2)", near_die, 0u },
                { "B->B local (farL2 only)  ", far_die,  1u },
                { "A->A local (nearL2 only) ", near_die, 1u },
            };

            CHECK_CUDA(cudaSetDevice(0));
            cudaStream_t s_bg;
            CHECK_CUDA(cudaStreamCreateWithFlags(&s_bg, cudaStreamNonBlocking));
            unsigned long long* d_prog = nullptr;
            CHECK_CUDA(cudaMalloc(&d_prog, sizeof(unsigned long long)));
            NvhbiStopFlag stop;
            nvhbi_stop_flag_create(stop);
            const unsigned int cap = (t0.sms_p0 < t0.sms_p1) ? t0.sms_p0 : t0.sms_p1;

            printf("\n========= PART E: peer latency under GPU0 load =========\n");
            printf("probe = GPU1 sm%d, one thread, so it adds no load of its own\n", sm1_of[0]);
            printf("own_far = GPU0's OWN die%u SM -> die%u L2, the same crossing the\n"
                   "          background is loading. It is the discriminator: if the\n"
                   "          congestion point is upstream of where NVLink joins the\n"
                   "          fabric, own_far climbs while lat_far does not.\n",
                   near_die, far_die);
            printf("# RP,bg_mode,bg_sms,bg_GBps,lat_near,lat_far,own_far,d_near,d_far,d_own\n");

            for (int m = 0; m < 3; ++m)
            for (int bi = 0; bi < bg_n; ++bi) {
                unsigned int bg_sms = bg_list[bi] > cap ? cap : bg_list[bi];
                if (bg_sms == 0u && m > 0) continue;

                const unsigned int wp     = modes[m].wp;
                const unsigned int own    = modes[m].own;
                const unsigned int target = own ? wp : (1u - wp);
                const unsigned int cnt    = bg_sms
                    ? nvhbi_chunks_used(bg_sms, bg_nbps, bg_block, 1u) : 0u;

                CHECK_CUDA(cudaSetDevice(0));
                nvhbi_flush_l2(t0);
                if (cnt) {
                    CHECK_CUDA(cudaMemset(d_cursor, 0, sizeof(unsigned int)));
                    rp_warm_queue<<<t0.sm_count * 8, 128>>>(
                        t0.d_data, list0[target], 0u, cnt, t0.d_sm_side, target,
                        d_cursor, nullptr, t0.d_sink);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                }

                double bg_gbps = 0.0, t_a = 0.0;
                unsigned long long p0 = 0ull, p1 = 0ull;
                CHECK_CUDA(cudaMemset(d_prog, 0, sizeof(unsigned long long)));
                nvhbi_stop_flag_reset(stop);
                if (bg_sms) {
                    const unsigned long long dl =
                        (unsigned long long)60000u * (unsigned long long)t0.clock_khz;
                    nvhbi_stress_write<<<t0.sm_count * bg_nbps, bg_block, 0, s_bg>>>(
                        t0.d_data, t0.d_far_idx, t0.d_near_idx, t0.d_sm_side,
                        wp, own, bg_sms, bg_nbps, (unsigned int)t0.sm_count,
                        1u, 0u, 0u, dl, stop.d, d_prog, nullptr, nullptr, t0.d_sink);
                    CHECK_CUDA(cudaGetLastError());
                    spin_ms((double)settle_ms);
                    // Stamp the clock right after the copy lands, at BOTH ends,
                    // so the same bias sits on each and cancels in the interval.
                    CHECK_CUDA(cudaMemcpy(&p0, d_prog, sizeof(p0), cudaMemcpyDeviceToHost));
                    t_a = now_ms();
                }

                unsigned int lnear = ~0u, lfar = ~0u, lown = ~0u;
                for (unsigned int r = 0; r < lat_rounds; ++r) {
                    const unsigned int a = atomic_lat(1, t0.d_data, list_on1[near_die],
                                                      sm1_of[0]);
                    const unsigned int b = atomic_lat(1, t0.d_data, list_on1[far_die],
                                                      sm1_of[0]);
                    const unsigned int c = atomic_lat(0, t0.d_data, list0[far_die],
                                                      sm0_of[near_die]);
                    if (a < lnear) lnear = a;
                    if (b < lfar)  lfar  = b;
                    if (c < lown)  lown  = c;
                }

                CHECK_CUDA(cudaSetDevice(0));
                if (bg_sms) {
                    CHECK_CUDA(cudaMemcpy(&p1, d_prog, sizeof(p1), cudaMemcpyDeviceToHost));
                    const double t_b = now_ms();
                    bg_gbps = (t_b > t_a)
                        ? (double)(p1 - p0) * 32.0 / ((t_b - t_a) * 1e-3) / 1e9 : 0.0;
                    nvhbi_stop_flag_set(stop);
                    CHECK_CUDA(cudaStreamSynchronize(s_bg));
                }

                printf("RP,%d,%u,%.1f,%u,%u,%u,%+d,%+d,%+d\n",
                       m + 1, bg_sms, bg_gbps, lnear, lfar, lown,
                       (int)lnear - (int)L1[0][near_die],
                       (int)lfar  - (int)L1[0][far_die],
                       (int)lown  - (int)L0[near_die][far_die]);
                fflush(stdout);
            }

            printf("\nRead d_own first: it is GPU0's own SMs measuring the very crossing\n"
                   "the background saturates, so in mode 1 it MUST climb. If it does not,\n"
                   "the background is not congesting a die crossing at all and nothing\n"
                   "else in this table means anything.\n"
                   "Then, with d_own climbing in mode 1:\n"
                   "  d_far climbs too, d_near flat   -> the peer shares the A->B link.\n"
                   "  d_far flat while d_own climbs   -> the peer joins the fabric\n"
                   "     DOWNSTREAM of the congestion. The 3.44 TB/s plateau is then the\n"
                   "     writer die's L2->fabric egress, not the die link, which is\n"
                   "     exactly why adding 630 GB/s of NVLink traffic costs nothing.\n"
                   "  d_far and d_near climb together -> the effect is at NVLink ingress\n"
                   "     on GPU0, common to both dies.\n"
                   "Mode 2 (B->B local) loads the far die's L2 without touching the link,\n"
                   "so anything that moves in mode 1 and not in mode 2 is the link.\n");

            CHECK_CUDA(cudaFree(d_prog));
            CHECK_CUDA(cudaStreamDestroy(s_bg));
            nvhbi_stop_flag_destroy(stop);
        }
    }

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaFree(list_on1[0])); CHECK_CUDA(cudaFree(list_on1[1]));
    CHECK_CUDA(cudaFree(d_out1a)); CHECK_CUDA(cudaFree(d_out1b));
    CHECK_CUDA(cudaFree(d_cursor1));
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(d_out0a)); CHECK_CUDA(cudaFree(d_out0b));
    CHECK_CUDA(cudaFree(d_bad));   CHECK_CUDA(cudaFree(d_cursor));
    free(h_side0); free(h_side1);
    nvhbi_free(t1);
    nvhbi_free(t0);
    return 0;
}
