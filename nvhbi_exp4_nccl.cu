// nvhbi_exp4_nccl.cu -- EXPERIMENT 4
//
// NCCL all-to-all across 2 GPUs while GPU0 runs the experiment-1 remote-L2
// write load. Single process driving both devices (ncclCommInitAll).
//
// The control that makes this experiment mean anything
// ---------------------------------------------------
// NCCL's kernels run on GPU0 and compete with the background load for SMs, not
// only for the fabric. So "NCCL got slower" on its own proves nothing. Run it
// twice:
//
//   NVHBI_BG_LOCAL=0  background writes ACROSS the dies  (loads the bisection)
//   NVHBI_BG_LOCAL=1  background writes to its OWN die   (same SMs, same store
//                     rate, same L2 pressure, nothing crosses the fabric)
//
// The difference between those two is the bisection effect. The part common to
// both is SM/L2 contention. Sweeping NVHBI_BG_SMS_LIST gives the whole curve.
//
// Two limits to state when reporting this:
//   * NCCL buffers are ordinary cudaMalloc, and die ownership is a property of
//     the physical address (dies interleave at 4KiB). No CUDA API places a
//     contiguous buffer on one die, so the collective's traffic lands ~50/50 on
//     both. This is "a collective under NV-HBI contention", not "a collective
//     targeting die1" -- exp 2/3 are the die-targeted experiments.
//   * With 2 ranks an all-to-all degenerates into a pairwise exchange, the same
//     traffic as a bidirectional sendrecv. Use 4 or 8 GPUs for a real one.
//
// Env: NVHBI_NCCL_BYTES_LIST  per-pair message sizes, default "1048576,8388608,67108864"
//      NVHBI_BG_SMS_LIST      background SM counts, 0 = off, default "0,16,32,64"
//      NVHBI_BG_LOCAL         1 = own-die background (control), default 0
//      NVHBI_REPEAT           timed iterations per point, default 10
//      NVHBI_BG_BLOCK / NVHBI_BG_BLOCKS_PER_SM / NVHBI_BG_LINES
//      NVHBI_BUF_MULT         default 8

#include "nvhbi_common.cuh"
#include <nccl.h>
#include <chrono>
#include <vector>

#define CHECK_NCCL(call) do {                                                  \
    ncclResult_t _r = (call);                                                  \
    if (_r != ncclSuccess) {                                                   \
        fprintf(stderr, "NCCL Error %s:%d: %s\n",                              \
                __FILE__, __LINE__, ncclGetErrorString(_r));                   \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

static unsigned int env_u(const char* k, unsigned int d) {
    const char* s = getenv(k);
    return (s && *s) ? (unsigned int)atoi(s) : d;
}

static int parse_list_sz(const char* env, const char* dflt, size_t* out, int cap) {
    const char* s = getenv(env);
    if (!s || !*s) s = dflt;
    int n = 0;
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", s);
    for (char* tok = strtok(buf, ","); tok && n < cap; tok = strtok(nullptr, ",")) {
        long long v = atoll(tok);
        if (v >= 0) out[n++] = (size_t)v;
    }
    return n;
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static void spin_ms(double ms) {
    const double until = now_ms() + ms;
    while (now_ms() < until) { }
}

int main() {
    const unsigned int repeat   = env_u("NVHBI_REPEAT", 10u);
    const unsigned int bg_block = env_u("NVHBI_BG_BLOCK", 64u);
    const unsigned int bg_nbps  = env_u("NVHBI_BG_BLOCKS_PER_SM", 32u);
    const unsigned int bg_lines = env_u("NVHBI_BG_LINES", 1u);
    const unsigned int bg_local = env_u("NVHBI_BG_LOCAL", 0u);
    const double       buf_mult = (double)env_u("NVHBI_BUF_MULT", 8u);

    size_t msg_list[16];
    size_t bg_tmp[32];
    const int msg_n = parse_list_sz("NVHBI_NCCL_BYTES_LIST",
                                    "1048576,8388608,67108864", msg_list, 16);
    const int bg_n  = parse_list_sz("NVHBI_BG_SMS_LIST", "0,16,32,64", bg_tmp, 32);

    int ndev = 0;
    CHECK_CUDA(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { fprintf(stderr, "ERROR: need >= 2 GPUs, found %d\n", ndev); return 1; }
    const int nranks = 2;

    NvhbiTopo t;
    nvhbi_probe(t, 0, buf_mult);

    unsigned int bg_list[32];
    unsigned int bg_max = 0;
    for (int i = 0; i < bg_n; ++i) {
        bg_list[i] = (unsigned int)bg_tmp[i];
        if (bg_list[i] > t.sms_p0) bg_list[i] = t.sms_p0;
        if (bg_list[i] > bg_max) bg_max = bg_list[i];
    }
    const unsigned int bg_die = bg_local ? 0u : 1u;
    const unsigned int* d_bg_list = (bg_die == 1u) ? t.d_far_idx : t.d_near_idx;
    const unsigned int  bg_avail  = (bg_die == 1u) ? t.far_count : t.near_count;
    const unsigned int  bg_reserve = nvhbi_chunks_used(bg_max, bg_nbps, bg_block, bg_lines);
    if (bg_reserve > bg_avail) {
        fprintf(stderr, "ERROR: background needs %u die-%u chunks, have %u\n",
                bg_reserve, bg_die, bg_avail);
        return 1;
    }

    /* ---------------- NCCL setup ---------------- */
    std::vector<int> devs(nranks);
    for (int i = 0; i < nranks; ++i) devs[i] = i;
    std::vector<ncclComm_t> comms(nranks);
    CHECK_NCCL(ncclCommInitAll(comms.data(), nranks, devs.data()));

    size_t max_msg = 0;
    for (int i = 0; i < msg_n; ++i) if (msg_list[i] > max_msg) max_msg = msg_list[i];
    const size_t buf_bytes = max_msg * (size_t)nranks;

    std::vector<char*>        sbuf(nranks), rbuf(nranks);
    std::vector<cudaStream_t> str(nranks);
    std::vector<cudaEvent_t>  ev0(nranks), ev1(nranks);
    for (int i = 0; i < nranks; ++i) {
        CHECK_CUDA(cudaSetDevice(devs[i]));
        CHECK_CUDA(cudaMalloc(&sbuf[i], buf_bytes));
        CHECK_CUDA(cudaMalloc(&rbuf[i], buf_bytes));
        CHECK_CUDA(cudaMemset(sbuf[i], i + 1, buf_bytes));
        CHECK_CUDA(cudaStreamCreateWithFlags(&str[i], cudaStreamNonBlocking));
        CHECK_CUDA(cudaEventCreate(&ev0[i]));
        CHECK_CUDA(cudaEventCreate(&ev1[i]));
    }

    /* ---------------- background machinery ---------------- */
    CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t s_bg;
    CHECK_CUDA(cudaStreamCreateWithFlags(&s_bg, cudaStreamNonBlocking));
    unsigned long long* d_bg_prog = nullptr;
    CHECK_CUDA(cudaMalloc(&d_bg_prog, sizeof(unsigned long long)));
    NvhbiStopFlag stop;
    nvhbi_stop_flag_create(stop);

    printf("exp4: NCCL all-to-all, %d ranks\n", nranks);
    printf("  background: die%u (%s), up to %u SMs x %u blocks x %u thr, L=%u\n",
           bg_die, bg_local ? "OWN die = CONTROL" : "cross-die = loads the bisection",
           bg_max, bg_nbps, bg_block, bg_lines);
    printf("  NOTE: NCCL buffers are die-agnostic (~50/50 across both dies).\n");
    printf("  NOTE: NCCL also uses GPU0 SMs -- compare against NVHBI_BG_LOCAL=1.\n\n");

    auto run_alltoall = [&](size_t chunk, cudaStream_t* s) {
        CHECK_NCCL(ncclGroupStart());
        for (int i = 0; i < nranks; ++i) {
            CHECK_CUDA(cudaSetDevice(devs[i]));
            for (int r = 0; r < nranks; ++r) {
                CHECK_NCCL(ncclSend(sbuf[i] + (size_t)r * chunk, chunk, ncclChar,
                                    r, comms[i], s[i]));
                CHECK_NCCL(ncclRecv(rbuf[i] + (size_t)r * chunk, chunk, ncclChar,
                                    r, comms[i], s[i]));
            }
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int i = 0; i < nranks; ++i) {
            CHECK_CUDA(cudaSetDevice(devs[i]));
            CHECK_CUDA(cudaStreamSynchronize(s[i]));
        }
    };

    printf("# CFG,exp,bg_local,bg_sms,msg_bytes,rep,nccl_ms,algbw_GBps,busbw_GBps,bg_GBps\n");

    for (int bi = 0; bi < bg_n; ++bi) {
        const unsigned int bg_sms = bg_list[bi];
        const unsigned int bg_chunks = bg_sms
            ? nvhbi_chunks_used(bg_sms, bg_nbps, bg_block, bg_lines) : 0u;

        for (int mi = 0; mi < msg_n; ++mi) {
            const size_t chunk = msg_list[mi];

            /* ---- flush + warm the background region from its owning die ---- */
            CHECK_CUDA(cudaSetDevice(0));
            nvhbi_flush_l2(t);
            if (bg_chunks) {
                nvhbi_warm(t, d_bg_list, 0u, bg_chunks, bg_die);
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            /* ---- start background, settle ---- */
            CHECK_CUDA(cudaMemset(d_bg_prog, 0, sizeof(unsigned long long)));
            nvhbi_stop_flag_reset(stop);
            if (bg_sms) {
                const unsigned long long dl =
                    (unsigned long long)60000u * (unsigned long long)t.clock_khz;
                nvhbi_stress_write<<<t.sm_count * bg_nbps, bg_block, 0, s_bg>>>(
                    t.d_data, t.d_far_idx, t.d_near_idx, t.d_sm_side,
                    /*writer_partition=*/0u, bg_local, bg_sms, bg_nbps,
                    (unsigned int)t.sm_count, bg_lines, /*chunk_offset=*/0u,
                    /*iteration=*/0u, dl, stop.d, d_bg_prog, nullptr, nullptr, t.d_sink);
                CHECK_CUDA(cudaGetLastError());
                spin_ms(100.0);
            }

            run_alltoall(chunk, str.data());          // untimed warm-up

            for (unsigned int rep = 0; rep < repeat; ++rep) {
                unsigned long long bg_p0 = 0, bg_p1 = 0;
                CHECK_CUDA(cudaSetDevice(0));
                CHECK_CUDA(cudaMemcpy(&bg_p0, d_bg_prog, sizeof(bg_p0), cudaMemcpyDeviceToHost));

                const double wall0 = now_ms();
                for (int i = 0; i < nranks; ++i) {
                    CHECK_CUDA(cudaSetDevice(devs[i]));
                    CHECK_CUDA(cudaEventRecord(ev0[i], str[i]));
                }
                run_alltoall(chunk, str.data());
                for (int i = 0; i < nranks; ++i) {
                    CHECK_CUDA(cudaSetDevice(devs[i]));
                    CHECK_CUDA(cudaEventRecord(ev1[i], str[i]));
                    CHECK_CUDA(cudaStreamSynchronize(str[i]));
                }
                const double wall_ms = now_ms() - wall0;

                float nccl_ms = 0.f;
                for (int i = 0; i < nranks; ++i) {
                    CHECK_CUDA(cudaSetDevice(devs[i]));
                    float e = 0.f;
                    CHECK_CUDA(cudaEventElapsedTime(&e, ev0[i], ev1[i]));
                    if (e > nccl_ms) nccl_ms = e;      // slowest rank sets the time
                }

                CHECK_CUDA(cudaSetDevice(0));
                CHECK_CUDA(cudaMemcpy(&bg_p1, d_bg_prog, sizeof(bg_p1), cudaMemcpyDeviceToHost));

                // nccl-tests convention for all-to-all
                const double off_rank = (double)chunk * (nranks - 1);
                const double algbw = (nccl_ms > 0.f) ? off_rank / (nccl_ms * 1e-3) / 1e9 : 0.0;
                const double busbw = algbw * (double)(nranks - 1) / (double)nranks;
                const double bg_gbps = (bg_sms && wall_ms > 0.0)
                    ? (double)(bg_p1 - bg_p0) * 32.0 / (wall_ms * 1e-3) / 1e9 : 0.0;

                printf("CFG,4,%u,%u,%zu,%u,%.4f,%.2f,%.2f,%.2f\n",
                       bg_local, bg_sms, chunk, rep, nccl_ms, algbw, busbw, bg_gbps);
                fflush(stdout);
            }

            if (bg_sms) {
                nvhbi_stop_flag_set(stop);
                CHECK_CUDA(cudaSetDevice(0));
                CHECK_CUDA(cudaStreamSynchronize(s_bg));
            }
        }
    }

    for (int i = 0; i < nranks; ++i) {
        CHECK_CUDA(cudaSetDevice(devs[i]));
        CHECK_CUDA(cudaEventDestroy(ev0[i]));
        CHECK_CUDA(cudaEventDestroy(ev1[i]));
        CHECK_CUDA(cudaStreamDestroy(str[i]));
        CHECK_CUDA(cudaFree(sbuf[i]));
        CHECK_CUDA(cudaFree(rbuf[i]));
        CHECK_NCCL(ncclCommDestroy(comms[i]));
    }
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaStreamDestroy(s_bg));
    CHECK_CUDA(cudaFree(d_bg_prog));
    nvhbi_stop_flag_destroy(stop);
    nvhbi_free(t);
    return 0;
}
