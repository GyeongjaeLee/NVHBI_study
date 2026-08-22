// nvhbi_validate_counter.cu -- TEMPORARY: validate the store-counter methodology
//
// exp1/exp2/exp3 all assume that one 4B `st.global.cg` produces exactly one 32B
// sector crossing the fabric, and that repeated stores to the SAME address are
// NOT merged by a store buffer or a fabric write-combiner before they cross.
// On B200 ncu had no counter permission, so this went unchecked. On B300 the
// profiler is available, so check it directly.
//
// This launches the REAL stress kernel (nvhbi_stress_write, the exact pattern
// the experiments use: each thread hammering its fixed 4 addresses every
// iteration) once per config, and lets ncu count fabric sectors around it.
//
// Compared against the deterministic, host-computed store count:
//
//     analytic_sectors = num_active_sm * num_blocks_per_sm * block_size
//                        * 4 * lines_mult * iteration
//
// two ncu metrics decompose where any absorption happens:
//
//     tex_op_write / analytic  ~ 1  => the SM issued every store (compiler and
//                                      SM did not coalesce the repeated writes)
//     ltcfabric    / analytic  ~ 1  => every issued store crossed as a sector
//                                      (no write-combining at L2/fabric)
//
// If ltcfabric/analytic < 1 while tex_op_write/analytic ~ 1, the SM issues the
// stores but the fabric merges them -- and the counter methodology overstates
// bandwidth by exactly that factor. If both are < 1, the compiler/SM dropped
// stores. Either way the ratio is the correction (or the refutation).
//
// The host-side counter (counted_sectors) is read too, but only trust it in a
// non-ncu run: ncu replays the kernel and the atomicAdd would accumulate across
// passes. Under ncu, compare ncu's counts against analytic_sectors instead.
//
// argv: [writer_partition]   0 | 1   (default 0; own-die contrast via NVHBI_OWN_DIE)
//
// Env: NVHBI_SMS          default "32,74"
//      NVHBI_BLOCK_SIZES  default "64"
//      NVHBI_LINES        distinct chunks per warp, default "1,8"
//                         (1 = same-address hammering, the worst case for a
//                          write-combiner; 8 = more distinct sectors, a control)
//      NVHBI_ITERATION    default 200
//      NVHBI_OWN_DIE      "0" cross-die, "1" own-die, "0,1" both. default "0"
//      NVHBI_BLOCKS_PER_SM default 32
//      NVHBI_BUF_MULT     default 8
//      NVHBI_LAT_THRESHOLD default 500

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
    const unsigned int wp = (argc > 1) ? (unsigned int)atoi(argv[1]) : 0u;

    const unsigned int nbps      = env_u("NVHBI_BLOCKS_PER_SM", 32u);
    const unsigned int iteration = env_u("NVHBI_ITERATION", 200u);
    const double       buf_mult  = (double)env_u("NVHBI_BUF_MULT", 8u);

    unsigned int sm_list[32], bs_list[16], ln_list[16], od_list[4];
    const int sm_n = parse_list("NVHBI_SMS", "32,74", sm_list, 32);
    const int bs_n = parse_list("NVHBI_BLOCK_SIZES", "64", bs_list, 16);
    const int ln_n = parse_list("NVHBI_LINES", "1,8", ln_list, 16);
    const int od_n = parse_list("NVHBI_OWN_DIE", "0", od_list, 4);

    NvhbiTopo t;
    nvhbi_probe(t, 0, buf_mult);

    unsigned long long* d_prog = nullptr;
    CHECK_CUDA(cudaMalloc(&d_prog, sizeof(unsigned long long)));
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));

    printf("validating store counter: one nvhbi_stress_write launch per row\n");
    printf("# CFG,writer_partition,own_die,num_active_sm,num_blocks_per_sm,block_size,"
           "lines_mult,iteration,analytic_sectors,counted_sectors,count_ratio,ms\n");

    for (int oi = 0; oi < od_n; ++oi) {
        const unsigned int own_die    = od_list[oi] ? 1u : 0u;
        const unsigned int target_die = own_die ? wp : (1u - wp);
        const unsigned int  avail  = (target_die == 1u) ? t.far_count : t.near_count;
        const unsigned int* d_list = (target_die == 1u) ? t.d_far_idx : t.d_near_idx;
        const unsigned int  max_sms = (wp == 0u) ? t.sms_p0 : t.sms_p1;

        for (int si = 0; si < sm_n; ++si) {
        for (int bi = 0; bi < bs_n; ++bi) {
        for (int li = 0; li < ln_n; ++li) {
            const unsigned int nsm = (sm_list[si] > max_sms) ? max_sms : sm_list[si];
            const unsigned int bs  = bs_list[bi];
            const unsigned int L   = ln_list[li] ? ln_list[li] : 1u;

            const unsigned int chunks = nvhbi_chunks_used(nsm, nbps, bs, L);
            if (chunks > avail) {
                fprintf(stderr, "skip: wp%u own%u sm%u bs%u L%u needs %u chunks, have %u\n",
                        wp, own_die, nsm, bs, L, chunks, avail);
                continue;
            }

            dim3 grid(t.sm_count * nbps), block(bs);

            nvhbi_flush_l2(t);
            nvhbi_warm_chunks<<<t.sm_count * 8, 128>>>(
                t.d_data, d_list, 0u, chunks, t.d_sm_side, target_die, t.d_sink);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemset(d_prog, 0, sizeof(unsigned long long)));
            CHECK_CUDA(cudaEventRecord(e0));
            nvhbi_stress_write<<<grid, block>>>(
                t.d_data, t.d_far_idx, t.d_near_idx, t.d_sm_side,
                wp, own_die, nsm, nbps, (unsigned int)t.sm_count,
                L, /*chunk_offset=*/0u,
                iteration, /*deadline=*/0ull, /*stop_flag=*/nullptr,
                d_prog, /*cycles_out=*/nullptr, /*cycles_min_out=*/nullptr, t.d_sink);
            CHECK_CUDA(cudaEventRecord(e1));
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            float ms = 0.f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
            unsigned long long counted = 0ull;
            CHECK_CUDA(cudaMemcpy(&counted, d_prog, sizeof(counted), cudaMemcpyDeviceToHost));

            const double analytic = (double)nsm * nbps * bs * 4.0 * L * iteration;
            const double ratio = (analytic > 0.0) ? (double)counted / analytic : 0.0;

            printf("CFG,%u,%u,%u,%u,%u,%u,%u,%.0f,%llu,%.4f,%.4f\n",
                   wp, own_die, nsm, nbps, bs, L, iteration,
                   analytic, counted, ratio, ms);
            fflush(stdout);
        }}}
    }

    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaFree(d_prog));
    nvhbi_free(t);
    return 0;
}
