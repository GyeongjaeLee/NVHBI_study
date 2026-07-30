// nvhbi_common.cuh -- shared machinery for the B200 NV-HBI (die-to-die) study.
//
// Used by:
//   nvhbi_exp1_bisection.cu  -- exp 1: remote-L2 write bisection vs #SMs (ncu)
//   nvhbi_exp23_peer.cu      -- exp 2/3: GPU1 -> GPU0 die1 / die0 over NVLink
//   nvhbi_exp4_nccl.cu       -- exp 4: NCCL all-to-all under NV-HBI contention
//
// Model
// -----
// B200 is two dies joined by NV-HBI. Each die has its own L2 slice set, and the
// address space is interleaved between them, so "which die owns this address"
// is a property of the address, not of the allocation. We recover both maps by
// latency probing (nvhbi_probe):
//
//   sm_side[smid]  even -> partition 0 (SM0's die), value/2 = rank in partition
//                  odd  -> partition 1,             value/2 = rank in partition
//   near_idx[]     4KiB chunk start offsets owned by partition 0
//   far_idx[]      4KiB chunk start offsets owned by partition 1
//
// B200's L2 is write-no-allocate, so a store only *hits* if the line was
// already pulled in by a load. Every experiment therefore warms the target
// die's lines using that die's OWN SMs (nvhbi_warm_chunks) -- warming from the
// far side would leave a local copy on the wrong die and hide fabric traffic.
//
// lines_mult
// -----------
// How many distinct 4KiB chunks each warp cycles through. Kept at 1 everywhere:
// the footprint then fits in the target die's L2, so every store is a remote L2
// hit and HBM stays out of the path. Larger values were explored as a "no L2
// locality" regime and dropped -- how much of L2 is actually resident is not
// something we can pin down, so it could not carry an argument.

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                                  \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA Error %s:%d: %s\n",                              \
                __FILE__, __LINE__, cudaGetErrorString(_e));                   \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define NVHBI_CHUNK_INTS  1024u          // 4KiB working/probing chunk
#define NVHBI_CHUNK_BYTES (NVHBI_CHUNK_INTS * 4u)

/* ------------------------------------------------------------------ device */

__device__ __forceinline__ unsigned int nvhbi_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// L1-bypassing 4B store that the compiler may not remove, reorder away, or
// merge across loop iterations.
__device__ __forceinline__ void nvhbi_st(unsigned int* addr, unsigned int val) {
    asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(addr), "r"(val));
}

__device__ __forceinline__ unsigned int nvhbi_ld(const unsigned int* addr) {
    unsigned int v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}

// Lane -> address mapping inside one 4KiB chunk.
//
//   lane = 0..31,  main = lane/4,  sub = lane%4
//   byte offset of store k = 512*main + 128*k + 32*sub      (k = 0..3)
//
// Lanes 4*main .. 4*main+3 cover the four 32B sectors of one 128B line, so each
// store instruction of that quad coalesces into exactly one 128B line, and the
// four stores walk four consecutive lines. A full warp covers the whole 4KiB
// chunk at 32B sector granularity -- which matches the measured 32B granularity
// of remote writes.
__device__ __forceinline__ void nvhbi_lane_addrs(unsigned int* data,
                                                 unsigned int cidx,
                                                 unsigned int lane,
                                                 unsigned int** out) {
    unsigned int base = cidx + 128u * (lane / 4u) + 8u * (lane % 4u);
    out[0] = &data[base];
    out[1] = &data[base + 32u];
    out[2] = &data[base + 64u];
    out[3] = &data[base + 96u];
}

// One store group = 4 remote 32B sectors per lane, 4B written into each.
//
// SETTLED on B200 (exp1, 74 SMs, cross-die): a 4B store really does ship a whole
// 32B sector across the fabric, so counting stores x 32B is sound. Two variants
// were measured to establish that, then retired:
//   16B per sector, same 4 instructions -> 2585 GB/s vs 2659 for 4B (within 3%)
//   32B per sector, but 8 instructions  -> 1318 GB/s, exactly half
// Bytes per sector do not move the sector rate; instruction count does. This
// shape is therefore optimal for loading the fabric: 32 lanes each hitting a
// distinct sector is 32 sectors per instruction, the hardware maximum.
__device__ __forceinline__ void nvhbi_store_group(unsigned int* data,
                                                  unsigned int cidx,
                                                  unsigned int lane,
                                                  unsigned int val) {
    unsigned int* a[4];
    nvhbi_lane_addrs(data, cidx, lane, a);
    nvhbi_st(a[0], val); nvhbi_st(a[1], val);
    nvhbi_st(a[2], val); nvhbi_st(a[3], val);
}

/* -------------------------------------------------------- topology probing */

__global__ void nvhbi_find_sm_side(unsigned int* __restrict__ data,
                                   unsigned int* __restrict__ latency_out,
                                   int target_smid,
                                   unsigned int* __restrict__ sink) {
    const unsigned int smid = nvhbi_smid();
    if (smid != (unsigned int)target_smid || threadIdx.x != 0) return;
    unsigned int consume = 0;
#pragma unroll 1
    for (int r = 0; r < 100; r++) {
        unsigned long long start = clock64();
        consume += data[0];
        unsigned long long end = clock64();
        atomicAdd(&data[0], 1);
        latency_out[smid] = (unsigned int)(end - start);
    }
    __stcg(&sink[smid], consume);
}

__global__ void nvhbi_find_data_side(unsigned int* __restrict__ data,
                                     size_t num_ints,
                                     unsigned int threshold_cycles,
                                     unsigned int* __restrict__ far_idx,
                                     unsigned int* __restrict__ near_idx,
                                     unsigned int* __restrict__ far_count,
                                     unsigned int* __restrict__ near_count,
                                     unsigned int* __restrict__ sink) {
    const unsigned int smid = nvhbi_smid();
    if (smid != 0 || threadIdx.x != 0) return;

    *far_count = 0u;
    *near_count = 0u;
    unsigned int consume = 0u;

#pragma unroll 1
    for (size_t it = 0; it < num_ints; it += NVHBI_CHUNK_INTS) {
        unsigned int idx = (unsigned int)it;
        unsigned int cyc = 0;
#pragma unroll 1
        for (unsigned int r = 0; r < 3; ++r) {
            unsigned long long start = clock64();
            consume += data[idx];
            unsigned long long end = clock64();
            atomicAdd(&data[idx], r + 1);
            cyc = (unsigned int)(end - start);   // last pass wins
        }
        __stcg(&sink[0], consume + smid);

        if (cyc > threshold_cycles) far_idx[atomicAdd(far_count, 1)]  = idx;
        else                        near_idx[atomicAdd(near_count, 1)] = idx;
    }
}

__global__ void nvhbi_flush_kernel(unsigned int* __restrict__ buf, size_t n,
                                   unsigned int* __restrict__ sink) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned int s = 0;
    for (; i < n; i += stride) s += buf[i];
    if (threadIdx.x == 0) atomicAdd(sink, s);
}

/* ------------------------------------------------------------- warm-up */

// Pull chunks [first, first+count) into the L2 of the die that owns them.
// Only SMs of `owner_partition` participate, so no copy is created on the far
// die. Each (chunk, lane) pair loads the same 4 lines the stress kernels store
// to, which is exactly what write-no-allocate needs to turn stores into hits.
__global__ void nvhbi_warm_chunks(unsigned int* __restrict__ data,
                                  const unsigned int* __restrict__ idx_list,
                                  unsigned int first,
                                  unsigned int count,
                                  const unsigned int* __restrict__ sm_side,
                                  unsigned int owner_partition,
                                  unsigned int* __restrict__ sink) {
    const unsigned int smid = nvhbi_smid();
    if ((sm_side[smid] % 2u) != owner_partition) return;

    const unsigned int lane = threadIdx.x % 32u;
    const unsigned int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    const unsigned int nwarps = (gridDim.x * blockDim.x) / 32u;

    unsigned int consume = 0u;
    for (unsigned int c = gwarp; c < count; c += nwarps) {
        unsigned int* a[4];
        nvhbi_lane_addrs(data, idx_list[first + c], lane, a);
        consume += nvhbi_ld(a[0]);
        consume += nvhbi_ld(a[1]);
        consume += nvhbi_ld(a[2]);
        consume += nvhbi_ld(a[3]);
    }
    nvhbi_st(&sink[smid], consume);
}

/* ------------------------------------------------------- remote-write stress

   Work mapping (dense, so the warmed chunk range is contiguous):

     warps_per_block = ceil(block_size / 32)
     plane_stride    = warps_per_block * num_blocks_per_sm * num_active_sm
     chunk_id(j)     = chunk_offset
                     + j * plane_stride                       (j = 0 .. L-1)
                     + warp_in_block
                     + warps_per_block * (q + num_blocks_per_sm * sm_rank)

   so the whole kernel touches chunks [chunk_offset, chunk_offset + L*plane_stride).

   Two termination modes:
     iteration > 0        -> fixed trip count (exp 1, ncu-friendly: deterministic)
     deadline_cycles > 0  -> run until the deadline or until *stop_flag != 0
                             (background load for exp 2/3/4)
   `progress` accumulates the number of 4B stores issued, which is what the host
   turns into achieved sector bandwidth.                                      */

__global__ void nvhbi_stress_write(unsigned int* __restrict__ data,
                                   const unsigned int* __restrict__ far_idx,
                                   const unsigned int* __restrict__ near_idx,
                                   const unsigned int* __restrict__ sm_side,
                                   unsigned int writer_partition,
                                   unsigned int target_own_die,
                                   unsigned int num_active_sm,
                                   unsigned int num_blocks_per_sm,
                                   unsigned int sm_count,
                                   unsigned int lines_mult,
                                   unsigned int chunk_offset,
                                   unsigned int iteration,
                                   unsigned long long deadline_cycles,
                                   const unsigned int* __restrict__ stop_flag,
                                   unsigned long long* __restrict__ progress,
                                   unsigned long long* __restrict__ cycles_out,
                                   unsigned int* __restrict__ sink) {
    const unsigned int smid = nvhbi_smid();
    if ((sm_side[smid] % 2u) != writer_partition) return;
    const unsigned int sm_rank = sm_side[smid] / 2u;
    if (sm_rank >= num_active_sm) return;

    // Cross-die (the measurement): partition-0 writers target die-1 memory,
    // which is the "far from SM0" list, and vice versa.
    // Own-die (target_own_die=1, the CONTROL): identical instruction stream,
    // identical SM occupancy, identical store rate, but nothing crosses the
    // fabric. Comparing the two separates fabric contention from SM/L2
    // contention -- without it, "the collective slowed down" could just mean
    // "something else was using the SMs".
    const unsigned int* list = target_own_die
        ? ((writer_partition == 0u) ? near_idx : far_idx)
        : ((writer_partition == 0u) ? far_idx  : near_idx);

    const unsigned int wpb   = (blockDim.x + 31u) / 32u;
    const unsigned int wib   = threadIdx.x / 32u;
    const unsigned int lane  = threadIdx.x % 32u;
    const unsigned int q     = blockIdx.x / sm_count;
    const unsigned int plane_stride = wpb * num_blocks_per_sm * num_active_sm;
    const unsigned int slot  = wib + wpb * (q + num_blocks_per_sm * sm_rank);

    // Lanes actually present in this warp (block_size may be < 32).
    const unsigned long long lanes =
        (unsigned long long)min(32u, blockDim.x - wib * 32u);

    unsigned int val = smid * 1000003u + threadIdx.x + 1u;
    unsigned long long done = 0ull;
    const unsigned long long t0 = clock64();

    if (deadline_cycles == 0ull) {
#pragma unroll 1
        for (unsigned int it = 0; it < iteration; ++it) {
#pragma unroll 1
            for (unsigned int j = 0; j < lines_mult; ++j) {
                nvhbi_store_group(data, list[chunk_offset + j * plane_stride + slot],
                                  lane, val);
                ++val;
            }
            done += 4ull * lines_mult;
        }
    } else {
#pragma unroll 1
        for (unsigned int it = 0; ; ++it) {
#pragma unroll 1
            for (unsigned int j = 0; j < lines_mult; ++j) {
                nvhbi_store_group(data, list[chunk_offset + j * plane_stride + slot],
                                  lane, val);
                ++val;
            }
            done += 4ull * lines_mult;
            if ((it & 63u) == 63u) {
                // Publish progress while still running, so the host can sample
                // background throughput over the foreground's exact window.
                if (progress && lane == 0u) { atomicAdd(progress, done * lanes); done = 0ull; }
                if ((unsigned long long)(clock64() - t0) > deadline_cycles) break;
                if (stop_flag && *(volatile const unsigned int*)stop_flag) break;
            }
        }
    }

    if (progress && lane == 0u && done) atomicAdd(progress, done * lanes);
    // One designated thread reports the cycles it spent in the loop. Comparing
    // that against the wall-clock ms gives the SM clock actually achieved, which
    // is the only way to tell a slower fabric from a throttled GPU.
    if (cycles_out && sm_rank == 0u && threadIdx.x == 0u && blockIdx.x < sm_count)
        *cycles_out = (unsigned long long)(clock64() - t0);
    nvhbi_st(&sink[smid], val);
}

/* ----------------------------------------------- peer (NVLink) write kernel

   Runs on the *remote* GPU and stores into `peer_data`, a pointer into the
   other GPU's allocation reached through peer access. `idx_list` is a device
   copy (on the running GPU) of the target GPU's chunk offsets for one die, so
   every byte written lands on the die we chose.                              */

// Injection rate is controlled by the grid size (how many warps inject), and
// every point runs for a fixed wall time via deadline_cycles, so a sweep over
// grid sizes gives evenly-timed samples along the NVLink-load axis.
__global__ void nvhbi_peer_write(unsigned int* __restrict__ peer_data,
                                 const unsigned int* __restrict__ idx_list,
                                 unsigned int first,
                                 unsigned int count,
                                 unsigned int iteration,
                                 unsigned long long deadline_cycles,
                                 unsigned long long* __restrict__ progress,
                                 unsigned int* __restrict__ sink) {
    const unsigned int lane   = threadIdx.x % 32u;
    const unsigned int gwarp  = (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    const unsigned int nwarps = (gridDim.x * blockDim.x) / 32u;
    if (count == 0u) return;

    const unsigned long long lanes =
        (unsigned long long)min(32u, blockDim.x - (threadIdx.x / 32u) * 32u);

    unsigned int val = gwarp * 2654435761u + lane + 1u;
    unsigned long long done = 0ull;
    const unsigned long long t0 = clock64();
    bool stop = false;

#pragma unroll 1
    for (unsigned int it = 0; (deadline_cycles ? !stop : (it < iteration)); ++it) {
#pragma unroll 1
        for (unsigned int c = gwarp; c < count; c += nwarps) {
            unsigned int* a[4];
            nvhbi_lane_addrs(peer_data, idx_list[first + c], lane, a);
            nvhbi_st(a[0], val); nvhbi_st(a[1], val);
            nvhbi_st(a[2], val); nvhbi_st(a[3], val);
            ++val;
            done += 4ull;
        }
        if (deadline_cycles &&
            (unsigned long long)(clock64() - t0) > deadline_cycles) stop = true;
    }
    if (progress && lane == 0u) atomicAdd(progress, done * lanes);
    nvhbi_st(&sink[blockIdx.x & 127u], val);
}

/* ------------------------------------------- peer attach-point pre-check

   Experiments 2 and 3 assume NVLink lands on GPU0's die 0, so that peer traffic
   aimed at die 1 has to take an extra NV-HBI hop while traffic aimed at die 0
   does not. If instead the NVLink ports are spread over both dies and routing
   is address-aware, both targets are one hop away and the exp2/exp3 contrast
   collapses -- so this must be checked before believing any of those numbers.

   Runs on the peer GPU, one thread, cold (call before any warm-up): a dependent
   chain of loads from the target GPU's chunks. Compare the per-access latency
   for die-0 chunks against die-1 chunks. A clear gap means the far die really
   is an extra hop away; near-equal latencies mean the assumption is wrong.    */

__global__ void nvhbi_peer_latency(const unsigned int* __restrict__ peer_data,
                                   const unsigned int* __restrict__ idx_list,
                                   unsigned int first,
                                   unsigned int count,
                                   unsigned int* __restrict__ out_cycles,
                                   unsigned int* __restrict__ sink) {
    if (threadIdx.x != 0 || blockIdx.x != 0 || count == 0u) return;
    unsigned int v = 0u;
    const unsigned long long t0 = clock64();
#pragma unroll 1
    for (unsigned int i = 0; i < count; ++i) {
        // (v & 31) makes each address depend on the previous load, so the
        // accesses cannot overlap and the total is count x latency.
        const unsigned int* p = &peer_data[idx_list[first + i] + (v & 31u)];
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
    }
    const unsigned long long t1 = clock64();
    *out_cycles = (unsigned int)((t1 - t0) / count);
    nvhbi_st(&sink[0], v);
}

/* -------------------------------------------------------------- host side */

struct NvhbiTopo {
    int            device      = 0;
    int            sm_count    = 0;
    size_t         l2_bytes    = 0;
    size_t         num_ints    = 0;
    int            clock_khz   = 0;

    unsigned int*  d_data      = nullptr;   // the probed allocation
    unsigned int*  d_sm_side   = nullptr;   // [sm_count]
    unsigned int*  d_far_idx   = nullptr;   // chunks on partition 1
    unsigned int*  d_near_idx  = nullptr;   // chunks on partition 0
    unsigned int   far_count   = 0;
    unsigned int   near_count  = 0;
    unsigned int   sms_p0      = 0;         // #SMs on SM0's die
    unsigned int   sms_p1      = 0;

    unsigned int*  d_sink      = nullptr;
    unsigned int*  d_flush     = nullptr;
    size_t         flush_ints  = 0;

    unsigned int*  h_far_idx   = nullptr;   // host mirrors, for peer GPUs
    unsigned int*  h_near_idx  = nullptr;
};

// Latency threshold separating near/far chunk probes.
// Pre-measured: B200 ~500, H100 ~400, A100 ~300 cycles.
static inline unsigned int nvhbi_default_threshold() {
    const char* e = getenv("NVHBI_LAT_THRESHOLD");
    return e ? (unsigned int)atoi(e) : 500u;
}

static void nvhbi_flush_l2(const NvhbiTopo& t) {
    nvhbi_flush_kernel<<<t.sm_count * 32, 256>>>(t.d_flush, t.flush_ints, t.d_sink);
    CHECK_CUDA(cudaDeviceSynchronize());
}

// buf_mult: allocation size as a multiple of L2 size. Bigger = more chunks per
// die (needed for large lines_mult sweeps) but a slower probe.
static void nvhbi_probe(NvhbiTopo& t, int device, double buf_mult, bool verbose = true) {
    t.device = device;
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    t.sm_count  = prop.multiProcessorCount;
    t.l2_bytes  = (size_t)prop.l2CacheSize;
    t.clock_khz = prop.clockRate;
    t.num_ints  = (size_t)(t.l2_bytes * buf_mult) / sizeof(unsigned int);
    t.num_ints  = (t.num_ints / NVHBI_CHUNK_INTS) * NVHBI_CHUNK_INTS;

    if (verbose)
        printf("[gpu%d] %s  SMs=%d  L2=%zu MB  buffer=%zu MB (%zu chunks)\n",
               device, prop.name, t.sm_count, t.l2_bytes / (1024 * 1024),
               t.num_ints * 4 / (1024 * 1024), t.num_ints / NVHBI_CHUNK_INTS);

    CHECK_CUDA(cudaMalloc(&t.d_data, t.num_ints * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(t.d_data, 0x5a, t.num_ints * sizeof(unsigned int)));

    CHECK_CUDA(cudaMalloc(&t.d_sink, t.sm_count * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(t.d_sink, 0, t.sm_count * sizeof(unsigned int)));

    t.flush_ints = (t.l2_bytes * 2) / sizeof(unsigned int);
    CHECK_CUDA(cudaMalloc(&t.d_flush, t.flush_ints * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(t.d_flush, 0x01, t.flush_ints * sizeof(unsigned int)));

    /* ---- which die is each SM on? ---- */
    unsigned int* d_lat = nullptr;
    CHECK_CUDA(cudaMalloc(&d_lat, t.sm_count * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_lat, 0, t.sm_count * sizeof(unsigned int)));
    if (verbose) printf("[gpu%d] probing SM sides...\n", device);
    for (int i = 0; i < t.sm_count; ++i) {
        nvhbi_find_sm_side<<<t.sm_count, 1>>>(t.d_data, d_lat, i, t.d_sink);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    unsigned int* h_lat = (unsigned int*)malloc(t.sm_count * sizeof(unsigned int));
    CHECK_CUDA(cudaMemcpy(h_lat, d_lat, t.sm_count * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_lat));

    unsigned int* h_side = (unsigned int*)calloc(t.sm_count, sizeof(unsigned int));
    unsigned int n0 = 0, n1 = 1;
    for (int i = 0; i < t.sm_count; ++i) {
        if (std::abs((long long)h_lat[i] - (long long)h_lat[0]) < 50) { h_side[i] = n0; n0 += 2; }
        else                                                          { h_side[i] = n1; n1 += 2; }
    }
    t.sms_p0 = n0 / 2u;
    t.sms_p1 = (unsigned int)t.sm_count - t.sms_p0;
    free(h_lat);

    CHECK_CUDA(cudaMalloc(&t.d_sm_side, t.sm_count * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemcpy(t.d_sm_side, h_side, t.sm_count * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    free(h_side);

    /* ---- which die owns each 4KiB chunk? ---- */
    size_t nchunks = t.num_ints / NVHBI_CHUNK_INTS;
    CHECK_CUDA(cudaMalloc(&t.d_far_idx,  nchunks * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&t.d_near_idx, nchunks * sizeof(unsigned int)));
    unsigned int *d_fc = nullptr, *d_nc = nullptr;
    CHECK_CUDA(cudaMalloc(&d_fc, sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_nc, sizeof(unsigned int)));

    if (verbose) printf("[gpu%d] probing data sides (%zu chunks)...\n", device, nchunks);
    // Launch one block per SM: the classification is relative to SM 0's own
    // latency, and only the block that lands on SM 0 does the work. A single
    // block is not guaranteed to be scheduled there.
    nvhbi_find_data_side<<<t.sm_count, 1>>>(t.d_data, t.num_ints, nvhbi_default_threshold(),
                                   t.d_far_idx, t.d_near_idx, d_fc, d_nc, t.d_sink);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&t.far_count,  d_fc, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&t.near_count, d_nc, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_fc));
    CHECK_CUDA(cudaFree(d_nc));

    t.h_far_idx  = (unsigned int*)malloc(t.far_count  * sizeof(unsigned int));
    t.h_near_idx = (unsigned int*)malloc(t.near_count * sizeof(unsigned int));
    CHECK_CUDA(cudaMemcpy(t.h_far_idx,  t.d_far_idx,  t.far_count  * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(t.h_near_idx, t.d_near_idx, t.near_count * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    if (verbose) {
        printf("[gpu%d] partition0 (SM0 die): %u SMs, %u chunks (%.1f MB)\n",
               device, t.sms_p0, t.near_count,
               t.near_count * (double)NVHBI_CHUNK_BYTES / (1024.0 * 1024.0));
        printf("[gpu%d] partition1          : %u SMs, %u chunks (%.1f MB)\n",
               device, t.sms_p1, t.far_count,
               t.far_count * (double)NVHBI_CHUNK_BYTES / (1024.0 * 1024.0));
        if (t.far_count == 0 || t.near_count == 0)
            fprintf(stderr, "[gpu%d] WARNING: one side is empty -- the latency threshold "
                            "(NVHBI_LAT_THRESHOLD=%u) is probably wrong for this GPU.\n",
                    device, nvhbi_default_threshold());
    }
}

static void nvhbi_free(NvhbiTopo& t) {
    CHECK_CUDA(cudaSetDevice(t.device));
    if (t.d_data)     CHECK_CUDA(cudaFree(t.d_data));
    if (t.d_sm_side)  CHECK_CUDA(cudaFree(t.d_sm_side));
    if (t.d_far_idx)  CHECK_CUDA(cudaFree(t.d_far_idx));
    if (t.d_near_idx) CHECK_CUDA(cudaFree(t.d_near_idx));
    if (t.d_sink)     CHECK_CUDA(cudaFree(t.d_sink));
    if (t.d_flush)    CHECK_CUDA(cudaFree(t.d_flush));
    free(t.h_far_idx);
    free(t.h_near_idx);
    t = NvhbiTopo{};
}

// Host mirror of the device-side chunk mapping: how many chunks a stress
// configuration touches, starting at chunk_offset.
static inline unsigned int nvhbi_chunks_used(unsigned int num_active_sm,
                                             unsigned int num_blocks_per_sm,
                                             unsigned int block_size,
                                             unsigned int lines_mult) {
    unsigned int wpb = (block_size + 31u) / 32u;
    return lines_mult * wpb * num_blocks_per_sm * num_active_sm;
}

static inline double nvhbi_footprint_mb(unsigned int num_active_sm,
                                        unsigned int num_blocks_per_sm,
                                        unsigned int block_size,
                                        unsigned int lines_mult) {
    return nvhbi_chunks_used(num_active_sm, num_blocks_per_sm, block_size, lines_mult)
           * (double)NVHBI_CHUNK_BYTES / (1024.0 * 1024.0);
}

/* -------- mapped host flag, so the host can stop a running background kernel */

struct NvhbiStopFlag {
    unsigned int* h = nullptr;   // host-visible
    unsigned int* d = nullptr;   // device-visible alias
};

static void nvhbi_stop_flag_create(NvhbiStopFlag& f) {
    CHECK_CUDA(cudaHostAlloc((void**)&f.h, sizeof(unsigned int), cudaHostAllocMapped));
    *f.h = 0u;
    CHECK_CUDA(cudaHostGetDevicePointer((void**)&f.d, f.h, 0));
}
static void nvhbi_stop_flag_set(NvhbiStopFlag& f) {
    *(volatile unsigned int*)f.h = 1u;
}
static void nvhbi_stop_flag_destroy(NvhbiStopFlag& f) {
    if (f.h) CHECK_CUDA(cudaFreeHost(f.h));
    f = NvhbiStopFlag{};
}
