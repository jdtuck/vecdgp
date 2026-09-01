// Straight C++/OpenMP transliteration of vecdgp.vecchia.u_entries,
// written the way the deepgp RcppArmadillo code is written: one small
// Cholesky per row, parallel over rows.
//
// Reads x_ord / NN / NN_len from binary files, writes Uvals back, prints
// the best wall time over several repeats.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline double matern25(double r2) {          // r2 = 5 * d2 / theta
    const double s = std::sqrt(r2);
    return (1.0 + s + r2 / 3.0) * std::exp(-s);
}

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : ".";
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    int sym  = argc > 3 ? atoi(argv[3]) : 0;   // 1 = exploit symmetry

    char path[512];
    auto slurp = [&](const char* name, void* dst, size_t bytes) {
        snprintf(path, sizeof path, "%s/%s", dir, name);
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "missing %s\n", path); exit(1); }
        if (fread(dst, 1, bytes, f) != bytes) { fprintf(stderr, "short read %s\n", name); exit(1); }
        fclose(f);
    };

    long long meta[4];
    slurp("meta.bin", meta, sizeof meta);
    const long long n = meta[0], d = meta[1], mp1 = meta[2];
    double theta, g;
    { double par[2]; slurp("par.bin", par, sizeof par); theta = par[0]; g = par[1]; }

    std::vector<double>  x(n * d);
    std::vector<long long> NN(n * mp1), NNlen(n);
    slurp("x.bin",  x.data(),  x.size()  * sizeof(double));
    slurp("nn.bin", NN.data(), NN.size() * sizeof(long long));
    slurp("nl.bin", NNlen.data(), NNlen.size() * sizeof(long long));

    std::vector<double> U(n * mp1, 0.0);

    double best = 1e30;
    for (int rep = 0; rep < reps; ++rep) {
        auto t0 = std::chrono::high_resolution_clock::now();

#pragma omp parallel
        {
            // per-thread scratch, allocated once (not once per row)
            std::vector<double> pts(mp1 * d), cov(mp1 * mp1), mvec(mp1);
#pragma omp for schedule(static)
            for (long long i = 0; i < n; ++i) {
                const long long n0 = NNlen[i];

                for (long long j = 0; j < n0; ++j) {
                    const long long idx = NN[i * mp1 + (n0 - 1 - j)];
                    for (long long k = 0; k < d; ++k) pts[j * d + k] = x[idx * d + k];
                }

                if (sym) {                       // fill lower triangle only
                    for (long long a = 0; a < n0; ++a) {
                        for (long long b = 0; b <= a; ++b) {
                            double r2 = 0.0;
                            for (long long k = 0; k < d; ++k) {
                                const double diff = pts[a * d + k] - pts[b * d + k];
                                r2 += diff * diff;
                            }
                            r2 = 5.0 * r2 / theta;
                            cov[a * n0 + b] = matern25(r2);
                        }
                        cov[a * n0 + a] += g;
                    }
                } else {                         // full square, as in the port
                    for (long long a = 0; a < n0; ++a) {
                        for (long long b = 0; b < n0; ++b) {
                            double r2 = 0.0;
                            for (long long k = 0; k < d; ++k) {
                                const double diff = pts[a * d + k] - pts[b * d + k];
                                r2 += diff * diff;
                            }
                            r2 = 5.0 * r2 / theta;
                            cov[a * n0 + b] = matern25(r2);
                        }
                        cov[a * n0 + a] += g;
                    }
                }

                // lower Cholesky, in place
                for (long long j = 0; j < n0; ++j) {
                    double s = cov[j * n0 + j];
                    for (long long k = 0; k < j; ++k) s -= cov[j * n0 + k] * cov[j * n0 + k];
                    cov[j * n0 + j] = std::sqrt(s);
                    const double djj = cov[j * n0 + j];
                    for (long long a = j + 1; a < n0; ++a) {
                        double t = cov[a * n0 + j];
                        for (long long k = 0; k < j; ++k) t -= cov[a * n0 + k] * cov[j * n0 + k];
                        cov[a * n0 + j] = t / djj;
                    }
                }

                // M = R^{-1} e_last, R = L^T  =>  back substitution
                const long long last = n0 - 1;
                mvec[last] = 1.0 / cov[last * n0 + last];
                for (long long k = last - 1; k >= 0; --k) {
                    double s = 0.0;
                    for (long long j = k + 1; j < n0; ++j) s += cov[j * n0 + k] * mvec[j];
                    mvec[k] = -s / cov[k * n0 + k];
                }
                for (long long k = 0; k < n0; ++k) U[i * mp1 + k] = mvec[last - k];
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }

    snprintf(path, sizeof path, "%s/u_cpp.bin", dir);
    FILE* f = fopen(path, "wb");
    fwrite(U.data(), sizeof(double), U.size(), f);
    fclose(f);

    printf("%.6f\n", best);
    return 0;
}
