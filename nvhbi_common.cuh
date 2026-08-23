// nvhbi_common.cuh -- shared machinery for the B200 NV-HBI (die-to-die) study.
//
// Used by nvhbi_exp1_bisection, nvhbi_dualdir, nvhbi_exp23_peer and
// nvhbi_route_probe.
//
// Model
// -----
// B200 is two dies joined by NV-HBI. Each die has its own L2 slice set and the
// address space is interleaved between them, so "which die owns this address"
// is a property of the address. nvhbi_probe() recovers both maps by latency
// probing:
//
//   sm_side[smid]  even -> partition 0 (SM0's die), value/2 = rank in partition
//                  odd  -> partition 1,             value/2 = rank in partition
//   near_idx[]     4KiB chunk offsets owned by partition 0
//   far_idx[]      4KiB chunk offsets owned by partition 1
//
// B200's L2 is write-no-allocate, so a store only hits if the line was already
// pulled in by a load. Targets are therefore warmed by the OWNING die's SMs
// (nvhbi_warm) -- warming from the far side would leave a replica on the wrong
// die and hide the fabric traffic.
//
// Access shapes
// -------------
//   write  4B into each of 4 distinct 32B sectors per lane. A store ships one
//          32B sector, so this is 32 sectors per instruction, the maximum.
//   read   one 4B load per 128B line, lanes 128B apart. A 4B load drags the
//          whole line across, so one instruction covers a 4KiB chunk.
//
// Every kernel keeps an in-kernel counter of the 32B sectors it moved; the host
// samples it mid-flight and divides by wall time. All four programs report that
// same statistic.

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

// How often a deadline loop stops to publish its counter and check for a stop.
// The stop flag lives in device memory, so a poll costs an L2 hit; 1024
// iterations still leaves ~50 samples inside a 200 ms window.
#ifndef NVHBI_POLL_MASK
#define NVHBI_POLL_MASK 1023u
#endif

// Same idea for nvhbi_peer_write, but its outer iteration is a whole pass over a
// warp's chunk list rather than one store group, so it needs a shorter interval
// to publish often enough for mid-flight sampling.
#ifndef NVHBI_PEER_POLL_MASK
#define NVHBI_PEER_POLL_MASK 63u
#endif

// Source tag in the top 4 bits of every value a writer stores. 0 = off, which
// is what bandwidth runs use. Folded into the initial value once, before the
// loop, so the store loop is unchanged and the measurement is unperturbed.
//
// It answers one question no bandwidth number can: when two sources write the
// SAME addresses at once, is each of them really landing there? Set it per
// device, run the window, then count tags with nvhbi_count_tags.
__device__ unsigned int nvhbi_src_tag = 0u;

/* ---------------------------------------------------------------- SM ticket

   Which resident slot a block occupies on its SM, which is what fixes the chunk
   each of its warps owns. Deriving it as blockIdx.x / sm_count would assume
   block b lands on SM b % sm_count, and CUDA guarantees no such thing; where
   that assumption fails, several warps are pointed at the SAME chunk and a
   reader then counts L2 hits as fresh fetches.

   The ticket is free-running and taken modulo num_blocks_per_sm, so no host
   reset is needed: each participating SM hands out exactly that many per
   launch, so consecutive tickets mod that count are always all-distinct and the
   phase just rotates between launches. Blocks that return earlier never take
   one. nvhbi_route_probe PART B2 reports how far the old assumption was off. */
#define NVHBI_MAX_SMS 256
__device__ unsigned int nvhbi_sm_ticket[NVHBI_MAX_SMS];

__device__ __forceinline__ unsigned int nvhbi_block_slot(unsigned int smid,
                                                         unsigned int blocks_per_sm) {
    __shared__ unsigned int s_q;
    if (threadIdx.x == 0)
        s_q = atomicAdd(&nvhbi_sm_ticket[smid], 1u) % (blocks_per_sm ? blocks_per_sm : 1u);
    __syncthreads();
    return s_q;
}

__device__ __forceinline__ unsigned int nvhbi_stamp(unsigned int val, unsigned int tag) {
    return tag ? ((tag << 28) | (val & 0x0FFFFFFFu)) : val;
}

#define NVHBI_CHUNK_INTS  1024u          // 4KiB working/probing chunk
#define NVHBI_CHUNK_BYTES (NVHBI_CHUNK_INTS * 4u)

/* ------------------------------------------------------------------ device */

__device__ __forceinline__ unsigned int nvhbi_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// L1-bypassing 4B store. asm volatile, so the compiler cannot remove it, merge
// it across iterations, or CSE it.
__device__ __forceinline__ void nvhbi_st(unsigned int* addr, unsigned int val) {
    asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(addr), "r"(val));
}

__device__ __forceinline__ unsigned int nvhbi_ld(const unsigned int* addr) {
    unsigned int v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}

// Streaming load: evict-first in L1 and L2, so a stream that touches each line
// once stops displacing everything else. This is the only eviction control
// expressible on the instruction itself. To PROTECT a working set instead, use
// the runtime L2 residency control (cudaLimitPersistingL2CacheSize plus an
// accessPolicyWindow), which is a real reservation rather than a replacement
// preference -- see nvhbi_dualdir.
__device__ __forceinline__ unsigned int nvhbi_ld_stream(const unsigned int* addr) {
    unsigned int v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}

// clock64() with a memory clobber. Plain clock64() is free to be scheduled
// across memory operations, which is fatal for latency timing.
__device__ __forceinline__ unsigned long long nvhbi_clock() {
    unsigned long long c;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c) :: "memory");
    return c;
}

// Lane -> address mapping inside one 4KiB chunk.
//
//   byte offset of store k = 512*(lane/4) + 128*k + 32*(lane%4)   (k = 0..3)
//
// A lane quad covers the four 32B sectors of one 128B line, so each store
// instruction coalesces into one line and the four stores walk four consecutive
// lines. A full warp covers the whole chunk at sector granularity.
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

// One store group: 4 remote 32B sectors per lane, 4B written into each. A 4B
// store ships a whole 32B sector, so 32 lanes x 4 stores = 128 sectors per
// group and 32 sectors per instruction -- the most one instruction can touch,
// and therefore the heaviest fabric load an SM can generate.
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
    // MIN over the samples: the classifier downstream splits on
    // |lat[i] - lat[0]| < 100 cycles, and one noisy sample would flip an SM to
    // the wrong die -- which then writes across the boundary while being
    // counted as own-die.
    unsigned int best = ~0u;
#pragma unroll 1
    for (int r = 0; r < 100; r++) {
        unsigned long long start = clock64();
        consume += data[0];
        unsigned long long end = clock64();
        atomicAdd(&data[0], 1);
        const unsigned int cyc = (unsigned int)(end - start);
        if (cyc < best) best = cyc;
    }
    latency_out[smid] = best;
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
//
// Warps pull chunk ids from a shared cursor rather than from a global warp
// index. That is what makes the die filter safe: splitting by global warp index
// gives each chunk to exactly one launched warp, and the warps sitting on the
// other die return above without ever claiming theirs, so only the
// participating-SM fraction of the list gets loaded. Callers go through
// nvhbi_warm(); nvhbi_route_probe PART B measures the coverage both ways.
__global__ void nvhbi_warm_chunks(unsigned int* __restrict__ data,
                                  const unsigned int* __restrict__ idx_list,
                                  unsigned int first,
                                  unsigned int count,
                                  const unsigned int* __restrict__ sm_side,
                                  unsigned int owner_partition,
                                  unsigned int* __restrict__ cursor,
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
        consume += nvhbi_ld(a[0]);
        consume += nvhbi_ld(a[1]);
        consume += nvhbi_ld(a[2]);
        consume += nvhbi_ld(a[3]);
    }
    nvhbi_st(&sink[smid], consume);
}

/* ------------------------------------------------------- remote-write stress

   Each participating warp owns one chunk, assigned densely so the warmed range
   is contiguous:

     slot = warp_in_block + warps_per_block * (q + num_blocks_per_sm * sm_rank)

   with q from the per-SM ticket and sm_rank the SM's dense rank inside its
   partition. Only SMs of `writer_partition` with sm_rank < num_active_sm run.

   Two termination modes:
     iteration > 0        fixed trip count      (exp1's slope pair)
     deadline_cycles > 0  run until the deadline or *stop_flag  (every
                          background load, and exp1's sampled point)

   `progress` accumulates 32B sectors issued; the host turns it into bandwidth. */

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
                                   unsigned long long* __restrict__ cycles_min_out,
                                   unsigned int* __restrict__ sink) {
    const unsigned int smid = nvhbi_smid();
    if ((sm_side[smid] % 2u) != writer_partition) return;
    const unsigned int sm_rank = sm_side[smid] / 2u;
    if (sm_rank >= num_active_sm) return;

    // Cross-die: partition-0 writers target die-1 memory, and vice versa.
    // target_own_die=1 is the control: same instruction stream, same occupancy,
    // nothing crosses.
    const unsigned int* list = target_own_die
        ? ((writer_partition == 0u) ? near_idx : far_idx)
        : ((writer_partition == 0u) ? far_idx  : near_idx);

    const unsigned int wpb   = (blockDim.x + 31u) / 32u;
    const unsigned int wib   = threadIdx.x / 32u;
    const unsigned int lane  = threadIdx.x % 32u;
    // Both early returns above are block-uniform (smid is the same for every
    // thread in a block), so every thread that reaches this point does, and the
    // __syncthreads() inside is safe.
    const unsigned int q     = nvhbi_block_slot(smid, num_blocks_per_sm);
    const unsigned int plane_stride = wpb * num_blocks_per_sm * num_active_sm;
    const unsigned int slot  = wib + wpb * (q + num_blocks_per_sm * sm_rank);

    // Lanes actually present in this warp (block_size may be < 32).
    const unsigned long long lanes =
        (unsigned long long)min(32u, blockDim.x - wib * 32u);

    unsigned int val = nvhbi_stamp(smid * 1000003u + threadIdx.x + 1u, nvhbi_src_tag);
    unsigned long long done = 0ull;
    const unsigned long long t0 = clock64();

    // lines_mult is a runtime argument, so the general inner loop reloads the
    // index array every iteration -- ~18% on a body that is only four stores.
    // L=1 is what every experiment uses, so it gets a path with the chunk
    // address loaded once.
    const unsigned int cidx1 = list[chunk_offset + slot];

    if (deadline_cycles == 0ull) {
        if (lines_mult == 1u) {
#pragma unroll 1
            for (unsigned int it = 0; it < iteration; ++it) {
                nvhbi_store_group(data, cidx1, lane, val);
                ++val;
                done += 4ull;
            }
        } else {
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
        }
    } else {
#pragma unroll 1
        for (unsigned int it = 0; ; ++it) {
            if (lines_mult == 1u) {
                nvhbi_store_group(data, cidx1, lane, val);
                ++val;
                done += 4ull;
            } else {
#pragma unroll 1
            for (unsigned int j = 0; j < lines_mult; ++j) {
                nvhbi_store_group(data, list[chunk_offset + j * plane_stride + slot],
                                  lane, val);
                ++val;
            }
            done += 4ull * lines_mult;
            }
            // Publish and check for a stop, rarely: the flag is in device
            // memory (a mapped host page here cost PCIe reads and pinned the
            // kernel to ~337 GB/s), and 1024 iterations still leaves ~50
            // samples inside a 200 ms window.
            if ((it & NVHBI_POLL_MASK) == NVHBI_POLL_MASK) {
                if (progress && lane == 0u) { atomicAdd(progress, done * lanes); done = 0ull; }
                if ((unsigned long long)(clock64() - t0) > deadline_cycles) break;
                if (stop_flag && *(volatile const unsigned int*)stop_flag) break;
            }
        }
    }

    if (progress && lane == 0u && done) atomicAdd(progress, done * lanes);
    // Longest and shortest loop span over the participating warps. The longest
    // gives the achieved SM clock; their ratio (span_ratio) says how evenly the
    // fabric shared itself out, which is why a fixed-iteration run -- timed by
    // its slowest warp -- can read below a mid-flight sampled run of the same
    // kernel. Callers must zero *cycles_out and set *cycles_min_out to all-ones.
    if (cycles_out && lane == 0u)
        atomicMax(cycles_out, (unsigned long long)(clock64() - t0));
    if (cycles_min_out && lane == 0u)
        atomicMin(cycles_min_out, (unsigned long long)(clock64() - t0));
    nvhbi_st(&sink[smid], val);
}

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
                           unsigned int r_first,         // first chunk of the read sweep
                           unsigned int r_evict_first,   // L2 hint on the reads
                           unsigned long long deadline_cycles,
                           const unsigned int* __restrict__ stop_flag,
                           unsigned long long* __restrict__ w_prog,
                           unsigned long long* __restrict__ r_prog,
                           unsigned long long* __restrict__ cycles_out,
                           unsigned int* __restrict__ sink) {
    const unsigned int smid = nvhbi_smid();
    const unsigned int part = sm_side[smid] % 2u;
    const unsigned int rank = sm_side[smid] / 2u;

    const unsigned int wpb  = (blockDim.x + 31u) / 32u;
    const unsigned int wib  = threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x % 32u;
    const unsigned long long lanes =
        (unsigned long long)min(32u, blockDim.x - wib * 32u);

    const bool is_writer = (part == wp);
    if (rank >= (is_writer ? w_active_sm : r_active_sm)) return;

    // Block-uniform above, so the __syncthreads() inside is safe.
    const unsigned int q    = nvhbi_block_slot(smid, nbps);
    const unsigned int slot = wib + wpb * (q + nbps * rank);
    unsigned long long done = 0ull;
    unsigned int acc = 0u;
    const unsigned long long t0 = clock64();

    if (is_writer) {
        // Writers on die A target die B: "far from SM0" when A is partition 0.
        const unsigned int* list = (wp == 0u) ? far_idx : near_idx;
        unsigned int val = nvhbi_stamp(smid * 1000003u + threadIdx.x + 1u, nvhbi_src_tag);
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
        //
        // r_first steps the CONTROL (r_local=1) past the writers' chunks. Without
        // it the local readers walk the whole die-B list starting at chunk 0,
        // which is exactly the range the writers are storing into: the control
        // then measures read/write sharing on the same lines on top of the L2
        // pressure it is supposed to isolate.
        const unsigned int* list = r_local ? ((wp == 0u) ? far_idx  : near_idx)
                                           : ((wp == 0u) ? near_idx : far_idx);
        const unsigned int nwarps = wpb * nbps * r_active_sm;
        unsigned int c = (slot >= r_count) ? (slot % r_count) : slot;
        // The hint choice is loop-invariant and the loop body is ONE load, so
        // testing it per iteration would be a large fraction of the work.
        if (r_evict_first) {
#pragma unroll 1
            for (unsigned int it = 0; ; ++it) {
                // byte 128*lane: 32 lanes cover the 32 lines of the 4KiB chunk,
                // one instruction, 4 sectors pulled per lane.
                acc += nvhbi_ld_stream(&data[list[r_first + c] + 32u * lane]);
                done += 4ull;
                c += nwarps;
                if (c >= r_count) c = (c >= 2u * r_count) ? (c % r_count) : (c - r_count);
                if ((it & NVHBI_POLL_MASK) == NVHBI_POLL_MASK) {
                    if (r_prog && lane == 0u) { atomicAdd(r_prog, done * lanes); done = 0ull; }
                    if ((unsigned long long)(clock64() - t0) > deadline_cycles) break;
                    if (stop_flag && *(volatile const unsigned int*)stop_flag) break;
                }
            }
        } else {
#pragma unroll 1
            for (unsigned int it = 0; ; ++it) {
                acc += nvhbi_ld1(&data[list[r_first + c] + 32u * lane]);
                done += 4ull;
                c += nwarps;
                if (c >= r_count) c = (c >= 2u * r_count) ? (c % r_count) : (c - r_count);
                if ((it & NVHBI_POLL_MASK) == NVHBI_POLL_MASK) {
                    if (r_prog && lane == 0u) { atomicAdd(r_prog, done * lanes); done = 0ull; }
                    if ((unsigned long long)(clock64() - t0) > deadline_cycles) break;
                    if (stop_flag && *(volatile const unsigned int*)stop_flag) break;
                }
            }
        }
        if (r_prog && lane == 0u && done) atomicAdd(r_prog, done * lanes);
    }
    // Longest loop span over the participating warps, so the caller can turn it
    // into an achieved clock. Callers must zero *cycles_out first.
    if (cycles_out && lane == 0u)
        atomicMax(cycles_out, (unsigned long long)(clock64() - t0));
    nvhbi_st(&sink[smid], acc + (unsigned int)done);
}

/* ----------------------------------------------- peer (NVLink) write kernel

   Runs on the *remote* GPU and stores into `peer_data`, a pointer into the
   other GPU's allocation reached through peer access. `idx_list` is a device
   copy (on the running GPU) of the target GPU's chunk offsets for one die, so
   every byte written lands on the die we chose.                              */

// Injection rate comes from the grid size and the block size; idle_cycles adds
// a spin after each store group so the duty cycle, and hence the offered load,
// is continuously tunable below one warp's worth.
__device__ __forceinline__ void nvhbi_spin(unsigned long long cycles) {
    if (!cycles) return;
    const unsigned long long s = clock64();
    while ((unsigned long long)(clock64() - s) < cycles) { }
}

__global__ void nvhbi_peer_write(unsigned int* __restrict__ peer_data,
                                 const unsigned int* __restrict__ idx_list,
                                 unsigned int first,
                                 unsigned int count,
                                 unsigned int iteration,
                                 unsigned long long deadline_cycles,
                                 unsigned int idle_cycles,
                                 unsigned long long* __restrict__ progress,
                                 unsigned int* __restrict__ sink) {
    // Warp id from (block, warp-in-block), not from the flat thread id: only
    // this form survives a SUB-WARP block size, where the flat one would give
    // whole groups of blocks the same id and an nwarps of 0.
    const unsigned int wpb    = (blockDim.x + 31u) / 32u;
    const unsigned int wib    = threadIdx.x / 32u;
    const unsigned int lane   = threadIdx.x % 32u;
    const unsigned int gwarp  = blockIdx.x * wpb + wib;
    const unsigned int nwarps = gridDim.x * wpb;
    if (count == 0u) return;

    const unsigned long long lanes =
        (unsigned long long)min(32u, blockDim.x - wib * 32u);

    unsigned int val = nvhbi_stamp(gwarp * 2654435761u + lane + 1u, nvhbi_src_tag);
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
            done += 4ull;                    // 4 remote 32B sectors
            nvhbi_spin(idle_cycles);
        }
        // Publish periodically, exactly as nvhbi_stress_write does. This used to
        // happen only at kernel exit, which is invisible to a host that samples
        // the counter WHILE the kernel runs -- the shared-window harness read the
        // same 0 at both ends and reported 0.00 GB/s for the peer.
        //
        // A separate, finer mask than the stress kernel's: one outer iteration
        // here is a whole pass over the warp's chunks, so at the top of the block
        // size sweep a warp completes only ~16k of them per second and a 1024
        // interval would publish three times in a 200 ms window.
        if ((it & NVHBI_PEER_POLL_MASK) == NVHBI_PEER_POLL_MASK) {
            if (progress && lane == 0u) { atomicAdd(progress, done * lanes); done = 0ull; }
        }
        if (deadline_cycles &&
            (unsigned long long)(clock64() - t0) > deadline_cycles) stop = true;
    }
    if (progress && lane == 0u && done) atomicAdd(progress, done * lanes);
    nvhbi_st(&sink[blockIdx.x & 127u], val);
}

/* --------------------------------------------------- who wrote it last?

   Reads back the words nvhbi_store_group writes and histograms them by source
   tag, so a range shared by two writers can be checked for what a bandwidth
   number cannot show: that both are really landing on those bytes. .cv on the
   loads, so no local replica answers for the home copy. */
__global__ void nvhbi_count_tags(const unsigned int* __restrict__ data,
                                 const unsigned int* __restrict__ idx_list,
                                 unsigned int first, unsigned int count,
                                 unsigned long long* __restrict__ hist,  // [16]
                                 unsigned int* __restrict__ sink) {
    const unsigned int wpb    = (blockDim.x + 31u) / 32u;
    const unsigned int wib    = threadIdx.x / 32u;
    const unsigned int lane   = threadIdx.x % 32u;
    const unsigned int gwarp  = blockIdx.x * wpb + wib;
    const unsigned int nwarps = gridDim.x * wpb;
    if (!nwarps) return;

    unsigned int last = 0u;
    for (unsigned int c = gwarp; c < count; c += nwarps) {
        const unsigned int cidx = idx_list[first + c];
        const unsigned int base = cidx + 128u * (lane / 4u) + 8u * (lane % 4u);
        for (unsigned int k = 0; k < 4u; ++k) {
            unsigned int v;
            asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(&data[base + 32u * k]));
            atomicAdd(&hist[v >> 28], 1ull);
            last = v;
        }
    }
    nvhbi_st(&sink[nvhbi_smid()], last);
}

/* ------------------------------------------------ peer (NVLink) read kernel

   The mirror of nvhbi_peer_write: GPU1 pulls from GPU0. Needed for a question
   the write-only experiments cannot answer -- when GPU1 touches a region GPU0
   has just warmed, is it served out of GPU0's L2 or does it go to GPU0's HBM?

   One 4B load per 128B line, 32 lanes covering the 32 lines of a 4KiB chunk, so
   one instruction per chunk. A 4B remote load drags a whole 128B line, so 4
   sectors are billed per lane -- the same accounting nvhbi_dualdir uses.

   THE REPLICA IS THE TRAP. A remote read leaves a copy on the REQUESTER's side,
   so the second visit to a line is a local L2 hit and crosses nothing. That is
   why nvhbi_dualdir streams over a footprint many times L2 instead of re-reading
   a resident one, and any caller of this kernel has to do the same or accept
   that it is measuring a local cache.

   `cv` selects the load flavour and the difference between the two IS a
   measurement:
     0 -> ld.global.cg   a replica on the reading side may serve the access
     1 -> ld.global.cv   volatile: re-fetch, so every access leaves this GPU
   Over NVLink the replica question is open in a way it is not for a die hop:
   if cg is much faster than cv on a re-read footprint, GPU1 is caching GPU0's
   lines locally and the "bandwidth" is not crossing NVLink at all.            */
__global__ void nvhbi_peer_read(const unsigned int* __restrict__ peer_data,
                                const unsigned int* __restrict__ idx_list,
                                unsigned int first,
                                unsigned int count,
                                unsigned long long deadline_cycles,
                                unsigned int idle_cycles,
                                unsigned int cv,
                                unsigned long long* __restrict__ progress,
                                unsigned int* __restrict__ sink) {
    // See nvhbi_peer_write: (block, warp-in-block) rather than the flat thread
    // id, so a sub-warp block size still gives every warp its own chunk stream.
    const unsigned int wpb    = (blockDim.x + 31u) / 32u;
    const unsigned int wib    = threadIdx.x / 32u;
    const unsigned int lane   = threadIdx.x % 32u;
    const unsigned int gwarp  = blockIdx.x * wpb + wib;
    const unsigned int nwarps = gridDim.x * wpb;
    // deadline_cycles is the ONLY exit; a zero deadline would spin forever.
    if (count == 0u || deadline_cycles == 0ull) return;

    const unsigned long long lanes =
        (unsigned long long)min(32u, blockDim.x - wib * 32u);

    unsigned int acc = 0u;
    unsigned long long done = 0ull;
    const unsigned long long t0 = clock64();
    bool stop = false;

#pragma unroll 1
    for (unsigned int it = 0; !stop; ++it) {
#pragma unroll 1
        for (unsigned int c = gwarp; c < count; c += nwarps) {
            const unsigned int* p = &peer_data[idx_list[first + c] + 32u * lane];
            unsigned int v;
            if (cv) asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p));
            else    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
            acc += v;
            done += 4ull;                    // one 4B load pulls a 128B line
            nvhbi_spin(idle_cycles);
        }
        if (deadline_cycles &&
            (unsigned long long)(clock64() - t0) > deadline_cycles) stop = true;
    }
    if (progress && lane == 0u) atomicAdd(progress, done * lanes);
    nvhbi_st(&sink[blockIdx.x & 127u], acc);
}

/* --------------------------------------------- peer LOAD latency, cold or warm

   nvhbi_peer_latency times an ATOMIC after a warm-up, on purpose: an atomic
   always travels to the home L2 slice, which is what makes it a clean topology
   probe. That is the wrong instrument for "L2 or HBM?", because the warm-up is
   exactly the variable under test.

   This one times a dependency chain of ordinary LOADS with no warm-up, so the
   caller controls the cache state: warm the region from GPU0's own SMs and the
   chain should report an L2-hit round trip; flush GPU0's L2 first and it should
   report an HBM round trip. The difference between the two is the answer.

   `stride_chunks` walks the list with a stride so a single pass over `nchunks`
   never revisits a line -- once GPU1 has read a line, GPU0's L2 holds it and
   the second visit is warm no matter what the caller set up.                   */
__global__ void nvhbi_peer_load_latency(const unsigned int* __restrict__ peer_data,
                                        const unsigned int* __restrict__ idx_list,
                                        unsigned int first,
                                        unsigned int nchunks,
                                        unsigned int probe_smid,
                                        unsigned int cv,
                                        unsigned int* __restrict__ out_cyc,
                                        unsigned int* __restrict__ sink) {
    if (nvhbi_smid() != probe_smid || threadIdx.x != 0 || nchunks == 0u) return;

    // A pointer chase would need the region pre-linked; instead make the NEXT
    // address depend on the value just loaded, which the memset pattern makes
    // nonzero but bounded. Accesses cannot overlap either way.
    unsigned int v = 0u;
    const unsigned long long t0 = nvhbi_clock();
#pragma unroll 1
    for (unsigned int c = 0; c < nchunks; ++c) {
        const unsigned int* p = &peer_data[idx_list[first + c] + (v & 31u)];
        unsigned int x;
        if (cv) asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(x) : "l"(p));
        else    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(p));
        v = x;
    }
    const unsigned long long t1 = nvhbi_clock();
    *out_cyc = (unsigned int)((t1 - t0) / nchunks);
    nvhbi_st(&sink[0], v);
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

__global__ void nvhbi_peer_latency(unsigned int* __restrict__ peer_data,
                                   const unsigned int* __restrict__ idx_list,
                                   unsigned int first,
                                   unsigned int nchunks,
                                   unsigned int reps,
                                   unsigned int warmup,
                                   unsigned int probe_smid,
                                   unsigned int* __restrict__ out_min,
                                   unsigned int* __restrict__ out_mean,
                                   unsigned int* __restrict__ sink) {
    // Pin the probe to a named SM. The issuing GPU is itself two dies, so an SM
    // on its far die crosses that GPU's OWN NV-HBI before reaching NVLink.
    // Leaving placement to the scheduler made the calibration irreproducible and
    // mixed that hop into the result.
    if (nvhbi_smid() != probe_smid || threadIdx.x != 0 || nchunks == 0u) return;

    // Time an ATOMIC, not a plain load. An atomic has to travel to the line's
    // home L2 slice every single time, so no local copy on this GPU can serve it
    // -- which matters because a remote read was measured to leave a local copy.
    // Its return value feeds the next address, so accesses cannot overlap.
    unsigned int v = 0u;
    unsigned int best = ~0u;
    unsigned long long acc = 0ull;
    unsigned int n = 0u;

    // Warm the peer mapping, the TLB and the remote lines before timing
    // anything. The previous version skipped this and walked 512 distinct cold
    // chunks once each, so whichever die was probed FIRST absorbed all of the
    // setup cost and came out thousands of cycles slower -- an artifact of
    // measurement order, not a topology difference.
#pragma unroll 1
    for (unsigned int w = 0; w < warmup; ++w) {
#pragma unroll 1
        for (unsigned int c = 0; c < nchunks; ++c)
            v += atomicAdd(&peer_data[idx_list[first + c]], 1u);
    }

    // Time a dependency CHAIN, not one access at a time. The previous version
    // stamped the clock around a single atomic and tried to force completion
    // with asm volatile("" :: "r"(got)) -- but an empty asm emits no
    // instruction, so ptxas had nothing to hang a scoreboard wait on, and the
    // real consumer (v = got) sat after the second stamp. It timed the atomic's
    // ISSUE and reported 26 cycles, which is register-level, not a memory round
    // trip.
    //
    // Here each atomic's return value feeds the next address, so the hardware
    // cannot overlap them, and nchunks accesses are timed together so the clock
    // overhead is amortized away.
#pragma unroll 1
    for (unsigned int r = 0; r < reps; ++r) {
        const unsigned long long t0 = nvhbi_clock();
#pragma unroll 1
        for (unsigned int c = 0; c < nchunks; ++c) {
            unsigned int* p = &peer_data[idx_list[first + c] + (v & 31u)];
            v = atomicAdd(p, 1u);          // result feeds the next address
        }
        const unsigned long long t1 = nvhbi_clock();
        const unsigned int per = (unsigned int)((t1 - t0) / nchunks);
        if (per < best) best = per;
        acc += per;
        ++n;
    }
    *out_min  = best;
    *out_mean = (unsigned int)(acc / (n ? n : 1u));
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
    unsigned int*  d_warm_cursor = nullptr;   // work queue head for nvhbi_warm
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

// Warm chunks [first, first+count) of `d_list` into the L2 of the die that owns
// them, using only that die's SMs. Resets the work-queue cursor first, so the
// coverage is the whole range every time. Issued on the default stream, like
// the raw launch it replaces, so an existing cudaDeviceSynchronize() at the
// call site still orders it.
// `stream` matters when the caller has attached a cudaAccessPolicyWindow to a
// stream: the window only applies to kernels launched on it, and the warm-up is
// what actually installs the lines, so it has to run there too.
static void nvhbi_warm(const NvhbiTopo& t, const unsigned int* d_list,
                       unsigned int first, unsigned int count,
                       unsigned int owner_partition, cudaStream_t stream = 0) {
    if (!count) return;
    CHECK_CUDA(cudaMemsetAsync(t.d_warm_cursor, 0, sizeof(unsigned int), stream));
    nvhbi_warm_chunks<<<t.sm_count * 8, 128, 0, stream>>>(
        t.d_data, d_list, first, count, t.d_sm_side, owner_partition,
        t.d_warm_cursor, t.d_sink);
    CHECK_CUDA(cudaGetLastError());
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
    // cudaDeviceProp::clockRate was removed in CUDA 13; the attribute query is
    // the portable spelling and returns the same kHz value.
    {
        int khz = 0;
        CHECK_CUDA(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, device));
        t.clock_khz = khz;
    }
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
    CHECK_CUDA(cudaMalloc(&t.d_warm_cursor, sizeof(unsigned int)));

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
        if (std::abs((long long)h_lat[i] - (long long)h_lat[0]) < 100) { h_side[i] = n0; n0 += 2; }
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
    if (t.d_warm_cursor) CHECK_CUDA(cudaFree(t.d_warm_cursor));
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

/* -------- stop flag, in DEVICE memory --------
   Mapped host memory would make every polling warp issue a PCIe read; device
   memory costs an L2 hit instead. The host sets it with a 4-byte H2D copy while
   the kernel runs, which is safe because the kernel is on a non-blocking
   stream. */

struct NvhbiStopFlag {
    unsigned int* d = nullptr;
};

static void nvhbi_stop_flag_create(NvhbiStopFlag& f) {
    CHECK_CUDA(cudaMalloc((void**)&f.d, sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(f.d, 0, sizeof(unsigned int)));
}
static void nvhbi_stop_flag_reset(NvhbiStopFlag& f) {
    CHECK_CUDA(cudaMemset(f.d, 0, sizeof(unsigned int)));
}
static void nvhbi_stop_flag_set(NvhbiStopFlag& f) {
    const unsigned int one = 1u;
    CHECK_CUDA(cudaMemcpy(f.d, &one, sizeof(one), cudaMemcpyHostToDevice));
}
static void nvhbi_stop_flag_destroy(NvhbiStopFlag& f) {
    if (f.d) CHECK_CUDA(cudaFree(f.d));
    f = NvhbiStopFlag{};
}
