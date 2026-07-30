// nvhbi_exp1_bisection.cu -- EXPERIMENT 1
//
// NV-HBI (die-to-die) write bisection bandwidth of a single B200, swept over the
// number of writing SMs.
//
// SMs of one die issue only stores into the other die's L2 lines, which that
// die's own SMs warmed first (B200 L2 is write-no-allocate, so a store only hits
// on a resident line). No loads, so all traffic crosses the fabric in one
// direction. Each lane writes 4B into each of 4 distinct 32B sectors, giving 32
// sectors per store instruction -- the hardware maximum, hence the heaviest
// fabric load an SM can generate.
//
// TWO-POINT SLOPE TIMING
// ----------------------
// Each configuration runs twice, at ITER_LO and ITER_HI, and the bandwidth is
//
//     BW = (bytes_hi - bytes_lo) / (t_hi - t_lo)
//
// so everything that does not scale with the loop count cancels: launch, grid
// ramp-up and drain, the cold miss on sm_side, the first index-array load, the
// tail. This matters enormously -- at iteration=100 the fixed cost was ~1.5 ms
// and dominated everything, with kernel time rising only 1.45x while the work
// rose 74x and two SM steps where MORE work finished FASTER. `naive_GBps` is
// still reported so the size of that effect stays visible; `overhead_ms` is the
// fixed cost the two points imply and should come out near zero.
//
// own_die: THE CONTROL, AND A RESULT
// ----------------------------------
// Repeating the sweep with stores kept on the writer's own die gives the same
// instruction stream and occupancy with nothing crossing the fabric. Measured:
//
//     <= 32 SMs   cross == own to 4 significant figures (83.8 GB/s per SM)
//        64 SMs   own / cross = 1.18
//        77 SMs   own / cross = 1.33   (cross 64% of linear, own 86%)
//
// Below 32 SMs each SM is limited by its own store issue rate and crossing the
// die boundary is free. Above that only the cross-die curve flattens, which is
// what makes the plateau a property of the fabric rather than of the SMs or L2.
// Plateau: ~4.4-4.8 TB/s (4753 +/- 4% over three sessions at 74 SMs).
//
// eff_GHz is the SM clock actually achieved, from in-kernel cycles over wall
// time. Bandwidth is only comparable between rows at the same eff_GHz --
// identical work once measured 1.56x apart across sessions before this was
// tracked. It comes from one designated thread, so it also picks up load
// imbalance; compare it between rows of equal SM count, not across SM counts.
//
// The store-shape question ("does a 4B store really ship a whole 32B sector?")
// was settled and the diagnostic modes retired -- see nvhbi_store_group.
//
// Env knobs:
//   NVHBI_BLOCK_SIZES   default "32,64" (block_size/32 = warps per SM slot)
//   NVHBI_SMS           SM-count list, default "" = powers of two up to the max
//   NVHBI_OWN_DIE       "0" cross-die, "1" own-die, "0,1" both. default "0,1"
//   NVHBI_ITER_LO       default 2000
//   NVHBI_ITER_HI       default 6000
//   NVHBI_BLOCKS_PER_SM default 32
//   NVHBI_BUF_MULT      default 8
//   NVHBI_LAT_THRESHOLD probe threshold in cycles, default 500
//
// argv: [writer_partition]   0 | 1 | 2(=both, default)

#include "nvhbi_common.cuh"

static int parse_list(const char* env, const char* dflt, unsigned int* out, int cap) {
    const char* s = getenv(env);
    if (!s || !*s) s = dflt;
    if (!s || !*s) return 0;
    int n = 0;
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", s);
    for (char* tok = strtok(buf, ","); tok && n < cap; tok = strtok(nullptr, ",")) {
        int v = atoi(tok);
        if (v >= 0) out[n++] = (unsigned int)v;
    }
    return n;
}

static unsigned int env_u(const char* k, unsigned int d) {
    const char* s = getenv(k);
    return (s && *s) ? (unsigned int)atoi(s) : d;
}

int main(int argc, char** argv) {
    const unsigned int arg_partition = (argc > 1) ? (unsigned int)atoi(argv[1]) : 2u;

    const unsigned int nbps     = env_u("NVHBI_BLOCKS_PER_SM", 32u);
    const unsigned int iter_lo  = env_u("NVHBI_ITER_LO", 2000u);
    const unsigned int iter_hi  = env_u("NVHBI_ITER_HI", 6000u);
    const double       buf_mult = (double)env_u("NVHBI_BUF_MULT", 8u);

    if (iter_hi <= iter_lo) {
        fprintf(stderr, "ERROR: NVHBI_ITER_HI (%u) must exceed NVHBI_ITER_LO (%u)\n",
                iter_hi, iter_lo);
        return 1;
    }

    unsigned int bs_list[16], sm_env[32], od_list[4];
    const int bs_n     = parse_list("NVHBI_BLOCK_SIZES", "32,64", bs_list, 16);
    const int sm_env_n = parse_list("NVHBI_SMS", "", sm_env, 32);
    const int od_n     = parse_list("NVHBI_OWN_DIE", "0,1", od_list, 4);

    NvhbiTopo t;
    nvhbi_probe(t, 0, buf_mult);

    printf("measuring exp1: NV-HBI write bisection\n");
    printf("  two-point slope: iteration %u vs %u\n", iter_lo, iter_hi);
    printf("# CFG,writer_partition,own_die,num_active_sm,num_blocks_per_sm,block_size,"
           "footprint_MB,sectors_lo,sectors_hi,ms_lo,ms_hi,"
           "naive_GBps,slope_GBps,overhead_ms,eff_GHz\n");

    const unsigned int p_lo = (arg_partition <= 1u) ? arg_partition : 0u;
    const unsigned int p_hi = (arg_partition <= 1u) ? arg_partition : 1u;

    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    unsigned long long* d_cycles = nullptr;
    CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(unsigned long long)));

    for (unsigned int wp = p_lo; wp <= p_hi; ++wp) {
    for (int oi = 0; oi < od_n; ++oi) {
        const unsigned int own_die = od_list[oi] ? 1u : 0u;

        // Cross-die writers on partition wp target the other die; own-die writers
        // target their own. Whichever die owns the lines must warm them.
        const unsigned int  target_die = own_die ? wp : (1u - wp);
        const unsigned int  avail   = (target_die == 1u) ? t.far_count : t.near_count;
        const unsigned int* d_list  = (target_die == 1u) ? t.d_far_idx : t.d_near_idx;
        const unsigned int  max_sms = (wp == 0u) ? t.sms_p0 : t.sms_p1;

        unsigned int sm_list[32];
        int sm_n = 0;
        if (sm_env_n > 0) {
            for (int i = 0; i < sm_env_n; ++i)
                if (sm_env[i] > 0 && sm_env[i] <= max_sms) sm_list[sm_n++] = sm_env[i];
        } else {
            for (unsigned int s = 1; s <= max_sms; s *= 2) sm_list[sm_n++] = s;
            if (sm_n == 0 || sm_list[sm_n - 1] != max_sms) sm_list[sm_n++] = max_sms;
        }

        for (int bi = 0; bi < bs_n; ++bi) {
        for (int si = 0; si < sm_n; ++si) {
            const unsigned int block_size = bs_list[bi];
            const unsigned int nsm        = sm_list[si];

            // lines_mult stays 1: the footprint must fit the target die's L2 so
            // every store is a hit and HBM stays out of the path.
            const unsigned int chunks = nvhbi_chunks_used(nsm, nbps, block_size, 1u);
            if (chunks > avail) {
                fprintf(stderr, "skip: p%u sm%u bs%u needs %u chunks, have %u\n",
                        wp, nsm, block_size, chunks, avail);
                continue;
            }

            dim3 grid(t.sm_count * nbps), block(block_size);
            float ms[2] = {0.f, 0.f};
            unsigned long long cyc[2] = {0ull, 0ull};
            const unsigned int iters[2] = {iter_lo, iter_hi};

            for (int k = 0; k < 2; ++k) {
                // Both points start from the same cache state. Anything not
                // proportional to the loop count cancels in the slope anyway.
                nvhbi_flush_l2(t);
                nvhbi_warm_chunks<<<t.sm_count * 8, 128>>>(
                    t.d_data, d_list, 0u, chunks, t.d_sm_side, target_die, t.d_sink);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());

                CHECK_CUDA(cudaEventRecord(e0));
                nvhbi_stress_write<<<grid, block>>>(
                    t.d_data, t.d_far_idx, t.d_near_idx, t.d_sm_side,
                    wp, own_die, nsm, nbps, (unsigned int)t.sm_count,
                    /*lines_mult=*/1u, /*chunk_offset=*/0u,
                    iters[k], /*deadline=*/0ull, /*stop_flag=*/nullptr,
                    /*progress=*/nullptr, d_cycles, t.d_sink);
                CHECK_CUDA(cudaEventRecord(e1));
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaEventElapsedTime(&ms[k], e0, e1));
                CHECK_CUDA(cudaMemcpy(&cyc[k], d_cycles, sizeof(cyc[k]),
                                      cudaMemcpyDeviceToHost));
            }

            const double eff_ghz = (ms[1] > 0.f) ? (double)cyc[1] / (ms[1] * 1e6) : 0.0;

            // 4 sectors per lane per iteration, 32B of fabric traffic each.
            const double sec_per_iter = (double)nsm * nbps * block_size * 4.0;
            const double sec_lo = sec_per_iter * iter_lo;
            const double sec_hi = sec_per_iter * iter_hi;

            const double dt = (ms[1] - ms[0]) * 1e-3;
            const double slope_gbps = (dt > 0.0) ? (sec_hi - sec_lo) * 32.0 / dt / 1e9 : 0.0;
            const double naive_gbps = (ms[1] > 0.f)
                ? sec_hi * 32.0 / (ms[1] * 1e-3) / 1e9 : 0.0;
            const double overhead_ms = (slope_gbps > 0.0)
                ? ms[1] - (sec_hi * 32.0 / (slope_gbps * 1e9)) * 1e3 : 0.0;

            printf("CFG,%u,%u,%u,%u,%u,%.2f,%.0f,%.0f,%.4f,%.4f,%.2f,%.2f,%.4f,%.3f\n",
                   wp, own_die, nsm, nbps, block_size,
                   nvhbi_footprint_mb(nsm, nbps, block_size, 1u),
                   sec_lo, sec_hi, ms[0], ms[1],
                   naive_gbps, slope_gbps, overhead_ms, eff_ghz);
            fflush(stdout);
        }}
    }}

    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaFree(d_cycles));
    nvhbi_free(t);
    return 0;
}
