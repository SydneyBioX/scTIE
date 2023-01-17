#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cstring>
#include <cmath>
#include <chrono>

#include "ot_ctx.hpp"

//#define DEBUG
//#define PROFILING
//#define PROFILING_STEP1

#define USE_BLOCKWISE
#define USE_SIMD
//#define USE_GEMV2
#define USE_GEMTV
#define USE_GEMTV2

typedef double v4df __attribute__ ((vector_size (32),  aligned (32)));

union v4d
{
  v4df v;
  double d[4];
};

template <typename data_t>
data_t nan_to_num(data_t num)
{
    if (std::isinf(num)) {
        if (std::signbit(num))
            return -MAXFLOAT;
        else
            return MAXFLOAT;
    } else {
        return num;
    }
}

template <typename data_t>
void gemv(
    data_t *out, data_t *mat, data_t *vec,
    int m, int n
) {
#ifdef USE_GEMV2
    //printf("%d %d\n", m, n);
    //fprintf(stderr, "done\n"); fflush(stderr);

    int i = 0;

    for (; i < (m - 3); i += 4) {
        //data_t r00 = 0, r01 = 0, r02 = 0, r03 = 0;
        //data_t r10 = 0, r11 = 0, r12 = 0, r13 = 0;
        //data_t r20 = 0, r21 = 0, r22 = 0, r23 = 0;
        //data_t r30 = 0, r31 = 0, r32 = 0, r33 = 0;
        v4d r0 = {0, 0, 0, 0};
        v4d r1 = {0, 0, 0, 0};
        v4d r2 = {0, 0, 0, 0};
        v4d r3 = {0, 0, 0, 0};

        for (int j = 0; j < (n - 3); j += 4) {
            //v4df v = *(v4df*)(&vec[j]);
            v4df v = { vec[j], vec[j+1], vec[j+2], vec[j+3] };

            //r00 += mat[(i    ) * n + j] * vec[j];
            //r01 += mat[(i    ) * n + j + 1] * vec[j + 1];
            //r02 += mat[(i    ) * n + j + 2] * vec[j + 2];
            //r03 += mat[(i    ) * n + j + 3] * vec[j + 3];
            //v4df m0 = *(v4df*)(&mat[(i    ) * n + j]);
            v4df m0 = { mat[(i    ) * n + j    ],
                        mat[(i    ) * n + j + 1],
                        mat[(i    ) * n + j + 2],
                        mat[(i    ) * n + j + 3] };
            r0.v += m0 * v;

            //r10 += mat[(i + 1) * n + j] * vec[j];
            //r11 += mat[(i + 1) * n + j + 1] * vec[j + 1];
            //r12 += mat[(i + 1) * n + j + 2] * vec[j + 2];
            //r13 += mat[(i + 1) * n + j + 3] * vec[j + 3];
            //v4df m1 = *(v4df*)(&mat[(i + 1) * n + j]);
            v4df m1 = { mat[(i + 1) * n + j    ],
                        mat[(i + 1) * n + j + 1],
                        mat[(i + 1) * n + j + 2],
                        mat[(i + 1) * n + j + 3] };
            r1.v += m1 * v;

            //r20 += mat[(i + 2) * n + j] * vec[j];
            //r21 += mat[(i + 2) * n + j + 1] * vec[j + 1];
            //r22 += mat[(i + 2) * n + j + 2] * vec[j + 2];
            //r23 += mat[(i + 2) * n + j + 3] * vec[j + 3];
            //v4df m2 = *(v4df*)(&mat[(i + 2) * n + j]);
            v4df m2 = { mat[(i + 2) * n + j    ],
                        mat[(i + 2) * n + j + 1],
                        mat[(i + 2) * n + j + 2],
                        mat[(i + 2) * n + j + 3] };
            r2.v += m2 * v;

            //r30 += mat[(i + 3) * n + j] * vec[j];
            //r31 += mat[(i + 3) * n + j + 1] * vec[j + 1];
            //r32 += mat[(i + 3) * n + j + 2] * vec[j + 2];
            //r33 += mat[(i + 3) * n + j + 3] * vec[j + 3];
            //v4df m3 = *(v4df*)(&mat[(i + 3) * n + j]);
            v4df m3 = { mat[(i + 3) * n + j    ],
                        mat[(i + 3) * n + j + 1],
                        mat[(i + 3) * n + j + 2],
                        mat[(i + 3) * n + j + 3] };
            r3.v += m3 * v;
        }

        //out[i    ] = r00 + r01 + r02 + r03;
        //out[i + 1] = r10 + r11 + r12 + r13;
        //out[i + 2] = r20 + r21 + r22 + r23;
        //out[i + 3] = r30 + r31 + r32 + r33;
        out[i    ] = r0.d[0] + r0.d[1] + r0.d[2] + r0.d[3];
        out[i + 1] = r1.d[0] + r1.d[1] + r1.d[2] + r1.d[3];
        out[i + 2] = r2.d[0] + r2.d[1] + r2.d[2] + r2.d[3];
        out[i + 3] = r3.d[0] + r3.d[1] + r3.d[2] + r3.d[3];
    }

    for (; i < m; i++) {
        data_t r = 0;
        for (int j = 0; j < n; j++) {
            r += mat[i * n + j] * vec[j];
        }
        out[i] = r;
    }

#else // USE_GEMV2
#ifdef USE_SIMD
    int i = 0;
    for (; i < (m - 3); i += 4) {
        data_t r0 = 0;
        data_t r1 = 0;
        data_t r2 = 0;
        data_t r3 = 0;

        #pragma unroll(4)
        for (int j = 0; j < n; j++) {
            r0 += mat[(i    ) * n + j] * vec[j];
            r1 += mat[(i + 1) * n + j] * vec[j];
            r2 += mat[(i + 2) * n + j] * vec[j];
            r3 += mat[(i + 3) * n + j] * vec[j];
        }

        out[i    ] = r0;
        out[i + 1] = r1;
        out[i + 2] = r2;
        out[i + 3] = r3;
    }

    for (; i < m; i++) {
        data_t r = 0;
        for (int j = 0; j < n; j++) {
            r += mat[i * n + j] * vec[j];
        }
        out[i] = r;
    }
#else
    for (int i = 0; i < m; i++) {
        data_t r = 0;
        for (int j = 0; j < n; j++) {
            r += mat[i * n + j] * vec[j];
        }
        out[i] = r;
    }
#endif
#endif // USE_GEMV2
}

template <typename data_t>
void gemtv(
    data_t *out, data_t *mat, data_t *vec,
    int m, int n
) {
#ifdef USE_GEMTV2
    for (int i = 0; i < m; i++)
        out[i] = 0;

    int j = 0;

    for (; j < (n - 3); j += 4) {
        int i = 0;

        #pragma unroll(4)
        for (; i < m; i++) {
            out[i] += mat[ j      * m + i] * vec[j    ];
            out[i] += mat[(j + 1) * m + i] * vec[j + 1];
            out[i] += mat[(j + 2) * m + i] * vec[j + 2];
            out[i] += mat[(j + 3) * m + i] * vec[j + 3];
        }
    }

    for (; j < n; j++) {
        int i = 0;

        for (; i < (m - 3); i += 4) {
            out[i    ] += mat[j * m + (i    )] * vec[j];
            out[i + 1] += mat[j * m + (i + 1)] * vec[j];
            out[i + 2] += mat[j * m + (i + 2)] * vec[j];
            out[i + 3] += mat[j * m + (i + 3)] * vec[j];
        }

        for (; i < m; i++) {
            out[i] += mat[j * m + i] * vec[j];
        }
    }

#else // GEMTV2
#ifdef USE_SIMD
    int i = 0;
    for (; i < (m - 3); i += 4) {
        data_t r0 = 0;
        data_t r1 = 0;
        data_t r2 = 0;
        data_t r3 = 0;

        for (int j = 0; j < n; j++) {
            r0 += mat[j * m + (i    )] * vec[j];
            r1 += mat[j * m + (i + 1)] * vec[j];
            r2 += mat[j * m + (i + 2)] * vec[j];
            r3 += mat[j * m + (i + 3)] * vec[j];
        }

        out[i    ] = r0;
        out[i + 1] = r1;
        out[i + 2] = r2;
        out[i + 3] = r3;
    }

    for (; i < m; i++) {
        data_t r = 0;
        for (int j = 0; j < n; j++) {
            r += mat[j * m + i] * vec[j];
        }
        out[i] = r;
    }
#else
    for (int i = 0; i < m; i++) {
        data_t r = 0;
        for (int j = 0; j < n; j++) {
            r += mat[j * m + i] * vec[j];
        }
        out[i] = r;
    }
#endif
#endif // GEMTV2
}

template <typename data_t>
void gemm(
    data_t *out, data_t *lhs, data_t *rhs,
    int m, int n, int k
) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            data_t r = 0;
            for (int l = 0; l < k; l++) {
                r += lhs[i * n + k] * rhs[k * m + j];
            }
            out[i * n + j] = r;
        }
    }
}

template <typename data_t>
void transpose(
    data_t *out, data_t *in,
    int m, int n
) {
#ifdef USE_BLOCKWISE
    int i = 0;

    for (; i < (m - 7); i += 8) {
        int j = 0;

        for (; j < (n - 7); j += 8) {
            for (int ii = 0; ii < 8; ii++) {
                for (int jj = 0; jj < 8; jj++) {
                    out[(j + jj) * m + (i + ii)] = in[(i + ii) * n + (j + jj)];
                }
            }
        }

        for (; j < n; j++) {
            for (int ii = 0; ii < 8; ii++) {
                out[j * m + (i + ii)] = in[(i + ii) * n + j];
            }
        }
    }

    for (; i < m; i++) {
        for (int j = 0; j < n; j++) {
            out[j * m + i] = in[i * n + j];
        }
    }

#else
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            out[j * m + i] = in[i * n + j];
        }
    }
#endif
}

template <typename data_t>
data_t fdiv(
    data_t l,
    data_t *x, // m
    data_t *p, // m
    data_t *dx, // m
    int m
) {
    // return l * np.sum(dx * (x * (np.log(x / p)) - x + p))
    data_t r = 0;
    for (int i = 0; i < m; i++) {
        r += dx[i] * (x[i] * std::log(x[i] / p[i]) - x[i] + p[i]);
    }
    return l * r;
}

template <typename data_t>
data_t fdivstar(
    data_t l,
    data_t *x,
    data_t *p,
    data_t *dx,
    int m
) {
    // return l * np.sum((p * dx) * (np.exp(x / l) - 1))
    data_t r = 0;
    for (int i = 0; i < m; i++) {
        r += (p[i] * dx[i]) * (std::exp(x[i] / l) - 1);
    }
    return l * r;
}

template <typename data_t>
data_t fdivstarexp(
    data_t l,
    data_t epsilon,
    data_t *x,
    data_t *p,
    data_t *dx,
    int m
) {
    // return l * np.sum((p * dx) * (np.exp(-epsilon * log(x) / l) - 1))
    data_t r = 0;
    for (int i = 0; i < m; i++) {
        r += (p[i] * dx[i]) * (std::exp((-epsilon * std::log(x[i])) / l) - 1);
    }
    return l * r;
}

template <typename data_t>
data_t primal(
    data_t *C, // (m, n)
    data_t *K, // (m, n)
    data_t *R, // (m, n)
    data_t *dx, // m
    data_t *dy, // n
    data_t *p, // m
    data_t *q, // n
    data_t *a, // m
    data_t *b, // n
    data_t epsilon,
    data_t lambda1,
    data_t lambda2,
    int m,
    int n
) {
    data_t *t1 = (data_t*)malloc(m * sizeof(data_t));
    data_t *t2 = (data_t*)malloc(n * sizeof(data_t));
#ifndef USE_GEMTV
    data_t *Rt = (data_t*)malloc(m * n * sizeof(data_t));
#endif

#ifdef PROFILING
    auto time_s0 = std::chrono::system_clock::now();
#endif

#ifndef USE_GEMTV
    transpose<data_t>(Rt, R, m, n);
#endif

#ifdef PROFILING
    auto time_s1 = std::chrono::system_clock::now();
#endif

    // dot(R, dy)
    gemv<data_t>(t1, R, dy, m, n);

#ifdef PROFILING
    auto time_s2 = std::chrono::system_clock::now();
#endif

    // dot(Rt, dx)
#ifdef USE_GEMTV
    gemtv<data_t>(t2, R, dx, n, m);
#else
    gemv<data_t>(t2, Rt, dx, n, m);
#endif

#ifdef PROFILING
    auto time_s3 = std::chrono::system_clock::now();
#endif

    // np.sum(R * np.nan_to_num(np.log(R)) - R + K)
    data_t t3 = 0;

    for (int i = 0; i < m * n; i++) {
        data_t a = R[i] * nan_to_num<data_t>(std::log(R[i])) - R[i] + K[i];
        //if (i % 1000 == 0 && !isnan(t3))
        //    printf("i %d t3 %f\n", i, t3);
        t3 += a;
    }

    //printf("final t3 %f\n", t3);

#ifdef PROFILING
    auto time_s4 = std::chrono::system_clock::now();
#endif

    // np.sum(R * C)
    data_t t4 = 0;
    for (int i = 0; i < m * n; i++) { t4 += R[i] * C[i]; }

    //printf("t4 %f\n", t4);

    data_t ret = fdiv<data_t>(lambda1, t1, p, dx, m)
               + fdiv<data_t>(lambda2, t2, q, dy, n)
               + (epsilon * t3 + t4) / (data_t)(m * n);

    //printf("ret %f\n", ret);

#ifdef PROFILING
    auto time_s5 = std::chrono::system_clock::now();
#endif

#ifdef PROFILING
    fprintf(stderr, "s0 %d ms\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(time_s1 - time_s0).count());
    fprintf(stderr, "s1 %d ms\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(time_s2 - time_s1).count());
    fprintf(stderr, "s2 %d ms\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(time_s3 - time_s2).count());
    fprintf(stderr, "s3 %d ms\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(time_s4 - time_s3).count());
    fprintf(stderr, "s4 %d ms\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(time_s5 - time_s4).count());
#endif

    free(t1);
    free(t2);
#ifndef USE_GEMTV
    free(Rt);
#endif

    return ret;
}

template <typename data_t>
data_t dual(
    data_t *C, // (m, n)
    data_t *K, // (m, n)
    data_t *R, // (m, n)
    data_t *dx, // m
    data_t *dy, // n
    data_t *p, // m
    data_t *q, // n
    data_t *a, // m
    data_t *b, // n
    data_t epsilon,
    data_t lambda1,
    data_t lambda2,
    int m,
    int n
) {
    // np.sum(R - K)
    data_t am = 0;
    for (int i = 0; i < m * n; i++) { am += R[i] - K[i]; }

    data_t t1 = - fdivstarexp<data_t>(lambda1, epsilon, a, p, dx, m);
    data_t t2 = - fdivstarexp<data_t>(lambda2, epsilon, b, q, dy, n);
    data_t t3 = - epsilon * am / (data_t)(m * n);

    return t1 + t2 + t3;
}

template <typename data_t>
data_t compute_duality_gap(
    data_t *C, // (m, n)
    data_t *K, // (m, n)
    data_t *R, // (m, n)
    data_t *dx, // m
    data_t *dy, // n
    data_t *p, // m
    data_t *q, // n
    data_t *a, // m
    data_t *b, // n
    data_t epsilon,
    data_t lambda1,
    data_t lambda2,
    int m,
    int n
) {
    data_t pri = primal<data_t>(
        C, // (m, n)
        K, // (m, n)
        R, // (m, n)
        dx, // m
        dy, // n
        p, // m
        q, // n
        a, // m
        b, // n
        epsilon,
        lambda1,
        lambda2,
        m,
        n
    );

    data_t dua = dual<data_t>(
        C, // (m, n)
        K, // (m, n)
        R, // (m, n)
        dx, // m
        dy, // n
        p, // m
        q, // n
        a, // m
        b, // n
        epsilon,
        lambda1,
        lambda2,
        m,
        n
    );

    return (pri - dua) / std::abs(pri);
}

template <typename data_t>
void update_k(
    data_t *K, // (m, n)
    data_t *K_, // (m, n)
    data_t *C, // (m, n)
    data_t *u, // (m)
    data_t *v, // (n)
    data_t epsilon,
    int m,
    int n
) {
    // _K = np.exp(-C / epsilon_i)
    for (int i = 0; i < m * n; i++) {
        K_[i] = std::exp(- C[i] / epsilon);
    }

    // K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            K[i * n + j] = std::exp((u[i] + v[j] - C[i * n + j]) / epsilon);
        }
    }
}

template <typename data_t>
void update_R(
    data_t *R, // (m, n)
    data_t *K, // (m, n)
    data_t *a, // (m)
    data_t *b, // (n)
    int m,
    int n
) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            R[i * n + j] = K[i * n + j] * a[i] * b[j];
        }
    }
}

template <typename data_t>
void update_a_b(
    data_t *a, // m
    data_t *b, // n
    data_t *K, // (m, n)
    data_t *dx, // m
    data_t *dy, // n
    data_t *p, // m
    data_t *q, // n
    data_t *u, // m
    data_t *v, // n
    data_t lambda1,
    data_t lambda2,
    data_t alpha1,
    data_t alpha2,
    data_t epsilon,
    int m,
    int n
) {
    int l = (m > n) ? m : n;

    data_t *t1 = (data_t*)malloc(l * sizeof(data_t));
    data_t *t2 = (data_t*)malloc(l * sizeof(data_t));

    // p / (K.dot(np.multiply(b, dy)))
    for (int i = 0; i < n; i++) {
        t1[i] = b[i] * dy[i];
    }

#ifdef PROFILING_STEP1
    auto time_s0 = std::chrono::system_clock::now();
#endif

    gemv<data_t>(t2, K, t1, m, n);

#ifdef PROFILING_STEP1
    auto time_s1 = std::chrono::system_clock::now();
#endif

    for (int i = 0; i < m; i++) {
        t1[i] = p[i] / t2[i];
    }

#ifdef PROFILING_STEP1
    auto time_s2 = std::chrono::system_clock::now();
#endif

    // a = (p / (K.dot(np.multiply(b, dy)))) ** alpha1 * np.exp(-u / (lambda1 + epsilon_i))
    for (int i = 0; i < m; i++) {
        a[i] = std::pow(t1[i], alpha1) * std::exp(-u[i] / (lambda1 + epsilon));
    }

#ifdef PROFILING_STEP1
    auto time_s3 = std::chrono::system_clock::now();
#endif

    // (q / (K.T.dot(np.multiply(a, dx))))
    for (int i = 0; i < m; i++) {
        t1[i] = a[i] * dx[i];
    }

#ifdef PROFILING_STEP1
    auto time_s4 = std::chrono::system_clock::now();
#endif

    gemtv<data_t>(t2, K, t1, n, m);

#ifdef PROFILING_STEP1
    auto time_s5 = std::chrono::system_clock::now();
#endif

    for (int i = 0; i < n; i++) {
        t1[i] = q[i] / t2[i];
    }

#ifdef PROFILING_STEP1
    auto time_s6 = std::chrono::system_clock::now();
#endif

    // (q / (K.T.dot(np.multiply(a, dx)))) ** alpha2 * np.exp(-v / (lambda2 + epsilon_i))
    for (int i = 0; i < n; i++) {
        b[i] = std::pow(t1[i], alpha2) * std::exp(-v[i] / (lambda2 + epsilon));
    }

#ifdef PROFILING_STEP1
    auto time_s7 = std::chrono::system_clock::now();
#endif

    free(t1);
    free(t2);

#ifdef PROFILING_STEP1
    fprintf(stderr, "gemv %f ms\n",
        1000.0 * std::chrono::duration_cast<std::chrono::duration<float>>(time_s1 - time_s0).count());
    fprintf(stderr, "powexp %f ms\n",
        1000.0 * std::chrono::duration_cast<std::chrono::duration<float>>(time_s3 - time_s2).count());
    fprintf(stderr, "gemtv %f ms\n",
        1000.0 * std::chrono::duration_cast<std::chrono::duration<float>>(time_s5 - time_s4).count());
    fprintf(stderr, "powexp %f ms\n",
        1000.0 * std::chrono::duration_cast<std::chrono::duration<float>>(time_s7 - time_s6).count());
#endif
}

template <typename data_t>
int step1_process(
    data_t *a,
    data_t *b,
    data_t *old_a,
    data_t *old_b,
    data_t *K,
    data_t *C,
    data_t *dx,
    data_t *dy,
    data_t *p,
    data_t *q,
    data_t *u,
    data_t *v,
    int cur_iter,
    int max_iter,
    int iters,
    data_t tau,
    data_t lambda1,
    data_t lambda2,
    data_t alpha1,
    data_t alpha2,
    data_t epsilon,
    int m,
    int n
) {
#ifdef DEBUG
    data_t a_sum = 0;
    for (int i = 0; i < m; i++)
        a_sum += a[i];
    data_t b_sum = 0;
    for (int i = 0; i < n; i++)
        b_sum += b[i];
    printf("a %f\n", a_sum);
    printf("b %f\n", b_sum);
#endif

    for (int i = 0; i < iters; i++) {
        cur_iter += 1;

        for (int i = 0; i < m; i++)
            old_a[i] = a[i];
        for (int i = 0; i < n; i++)
            old_b[i] = b[i];

        //memcpy(old_a, a, m * sizeof(data_t));
        //memcpy(old_b, b, n * sizeof(data_t));

#ifdef PROFILING_STEP1
    auto time_s0 = std::chrono::system_clock::now();
#endif
        update_a_b(
            a, // m
            b, // n
            K, // (m, n)
            dx, // m
            dy, // n
            p, // m
            q, // n
            u, // m
            v, // n
            lambda1,
            lambda2,
            alpha1,
            alpha2,
            epsilon,
            m,
            n
        );
#ifdef PROFILING_STEP1
    auto time_s1 = std::chrono::system_clock::now();
#endif

#ifdef PROFILING_STEP1
    fprintf(stderr, "step1 %f ms\n",
        1000.0 * std::chrono::duration_cast<std::chrono::duration<float>>(time_s1 - time_s0).count());
#endif

#ifdef DEBUG
        data_t a_sum = 0;
        for (int i = 0; i < m; i++)
            a_sum += a[i];
        data_t b_sum = 0;
        for (int i = 0; i < n; i++)
            b_sum += b[i];
        printf("a %f\n", a_sum);
        printf("b %f\n", b_sum);
#endif

        int to_stablize = 0;
        for (int j = 0; j < m; j++) {
            if (a[j] > tau) {
                to_stablize = 1;
                break;
            }
        }
        for (int j = 0; j < n; j++) {
            if (b[j] > tau) {
                to_stablize = 1;
                break;
            }
        }

        if (to_stablize) {
            // u = u + epsilon_i * np.log(a)
            for (int j = 0; j < m; j++) {
                u[j] = u[j] + epsilon * std::log(a[j]);
            }
            // v = v + epsilon_i * np.log(b)
            for (int j = 0; j < n; j++) {
                v[j] = v[j] + epsilon * std::log(b[j]);
            }
            // K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    K[i * n + j] = std::exp((u[i] + v[j] - C[i * n + j]) / epsilon);
                }
            }

            // a, b = np.ones(I), np.ones(J)
            for (int j = 0; j < m; j++) {
                a[j] = 1;
            }
            for (int j = 0; j < n; j++) {
                b[j] = 1;
            }

#ifdef DEBUG
            printf("stabilization\n");
#endif
        }

        if (cur_iter >= max_iter) {
            printf("Reached max_iter with duality gap still above threshold. Returning");
            return -1;
        }
    }

    return cur_iter;
}

template <typename data_t>
data_t update_process(
    data_t *R,
    data_t *a,
    data_t *b,
    data_t *old_a,
    data_t *old_b,
    data_t *K,
    data_t *_K,
    data_t *C,
    data_t *dx,
    data_t *dy,
    data_t *p,
    data_t *q,
    data_t *u,
    data_t *v,
    int epsilon_scalings,
    int cur_epsilon_scaling,
    int batch_size,
    data_t epsilon,
    data_t threshold,
    data_t tau,
    data_t lambda1,
    data_t lambda2,
    data_t alpha1,
    data_t alpha2,
    int cur_iter,
    int max_iter,
    int m,
    int n
) {
    data_t duality_gap = 1e100;

    data_t *_a = (data_t*)malloc(m * sizeof(data_t));
    data_t *_b = (data_t*)malloc(n * sizeof(data_t));

    while (duality_gap > threshold) {
        int iters = (cur_epsilon_scaling == epsilon_scalings) ? batch_size : 5;

        cur_iter =
            step1_process(
                a, b, old_a, old_b, K, C, dx, dy, p, q, u, v,
                cur_iter, max_iter, iters,
                tau, lambda1, lambda2, alpha1, alpha2, epsilon, m, n
            );

        // The real dual variables. a and b are only the stabilized variables
        //_a = a * np.exp(u / epsilon_i)
        for (int i = 0; i < m; i++) {
            _a[i] = a[i] * std::exp(u[i] / epsilon);
        }
        //_b = b * np.exp(v / epsilon_i)
        for (int i = 0; i < n; i++) {
            _b[i] = b[i] * std::exp(v[i] / epsilon);
        }

        // Skip duality gap computation for the first epsilon scalings, use dual variables evolution instead
        if (cur_epsilon_scaling == epsilon_scalings) {
            update_R(R, K, a, b, m, n);

            duality_gap =
                compute_duality_gap(
                    C, _K, R, dx, dy, p, q, _a, _b,
                    epsilon, lambda1, lambda2, m, n
                );
        }
        else {
            // v1 = np.linalg.norm(_a - old_a * np.exp(u / epsilon_i)) / (1 + np.linalg.norm(_a))
            data_t v11 = 0;
            for (int i = 0; i < m; i++) {
                data_t t = _a[i] - old_a[i] * std::exp(u[i] / epsilon);
                v11 += t * t;
            }
            data_t v12 = 0;
            for (int i = 0; i < m; i++) {
                v12 += _a[i] * _a[i];
            }
            data_t v1 = std::sqrt(v11) / (1 + std::sqrt(v12));

            // v2 = np.linalg.norm(_b - old_b * np.exp(v / epsilon_i)) / (1 + np.linalg.norm(_b))
            data_t v21 = 0;
            for (int i = 0; i < n; i++) {
                data_t t = _b[i] - old_b[i] * std::exp(v[i] / epsilon);
                v21 += t * t;
            }
            data_t v22 = 0;
            for (int i = 0; i < n; i++) {
                v22 += _b[i] * _b[i];
            }
            data_t v2 = std::sqrt(v21) / (1 + std::sqrt(v22));

            // duality_gap = max(v1, v2)
            duality_gap = std::max(v1, v2);
        }
    }

    free(_a);
    free(_b);

    return duality_gap;
}

// ****************************************************************************
// ****************************************************************************
// ****************************************************************************
// ****************************************************************************
// ****************************************************************************

extern "C" float dummy_float(
    float *C, // (m, n)
    float *K, // (m, n)
    float *R, // (m, n)
    float *dx, // m
    float *dy, // n
    float *p, // m
    float *q, // n
    float *a, // m
    float *b, // n
    float epsilon,
    float lambda1,
    float lambda2,
    int m,
    int n
) {
    return 0;
}

extern "C" double dummy_double(
    double *C, // (m, n)
    double *K, // (m, n)
    double *R, // (m, n)
    double *dx, // m
    double *dy, // n
    double *p, // m
    double *q, // n
    double *a, // m
    double *b, // n
    double epsilon,
    double lambda1,
    double lambda2,
    int m,
    int n
) {
    return 0;
}


extern "C" float primal_float(
    float *C, // (m, n)
    float *K, // (m, n)
    float *R, // (m, n)
    float *dx, // m
    float *dy, // n
    float *p, // m
    float *q, // n
    float *a, // m
    float *b, // n
    float epsilon,
    float lambda1,
    float lambda2,
    int m,
    int n
) {
    return primal<float>(
        C, // (m, n)
        K, // (m, n)
        R, // (m, n)
        dx, // m
        dy, // n
        p, // m
        q, // n
        a, // m
        b, // n
        epsilon,
        lambda1,
        lambda2,
        m,
        n
    );
}

extern "C" double primal_double(
    double *C, // (m, n)
    double *K, // (m, n)
    double *R, // (m, n)
    double *dx, // m
    double *dy, // n
    double *p, // m
    double *q, // n
    double *a, // m
    double *b, // n
    double epsilon,
    double lambda1,
    double lambda2,
    int m,
    int n
) {
    return primal<double>(
        C, // (m, n)
        K, // (m, n)
        R, // (m, n)
        dx, // m
        dy, // n
        p, // m
        q, // n
        a, // m
        b, // n
        epsilon,
        lambda1,
        lambda2,
        m,
        n
    );
}

extern "C" float dual_float(
    float *C, // (m, n)
    float *K, // (m, n)
    float *R, // (m, n)
    float *dx, // m
    float *dy, // n
    float *p, // m
    float *q, // n
    float *a, // m
    float *b, // n
    float epsilon,
    float lambda1,
    float lambda2,
    int m,
    int n
) {
    return dual<float>(
        C, // (m, n)
        K, // (m, n)
        R, // (m, n)
        dx, // m
        dy, // n
        p, // m
        q, // n
        a, // m
        b, // n
        epsilon,
        lambda1,
        lambda2,
        m,
        n
    );
}

extern "C" double dual_double(
    double *C, // (m, n)
    double *K, // (m, n)
    double *R, // (m, n)
    double *dx, // m
    double *dy, // n
    double *p, // m
    double *q, // n
    double *a, // m
    double *b, // n
    double epsilon,
    double lambda1,
    double lambda2,
    int m,
    int n
) {
    return dual<double>(
        C, // (m, n)
        K, // (m, n)
        R, // (m, n)
        dx, // m
        dy, // n
        p, // m
        q, // n
        a, // m
        b, // n
        epsilon,
        lambda1,
        lambda2,
        m,
        n
    );
}

extern "C" float compute_duality_gap_float(
    float *C, // (m, n)
    float *K, // (m, n)
    float *R, // (m, n)
    float *dx, // m
    float *dy, // n
    float *p, // m
    float *q, // n
    float *a, // m
    float *b, // n
    float epsilon,
    float lambda1,
    float lambda2,
    int m,
    int n
) {
    return compute_duality_gap<float>(
        C, // (m, n)
        K, // (m, n)
        R, // (m, n)
        dx, // m
        dy, // n
        p, // m
        q, // n
        a, // m
        b, // n
        epsilon,
        lambda1,
        lambda2,
        m,
        n
    );
}

extern "C" double compute_duality_gap_double(
    double *C, // (m, n)
    double *K, // (m, n)
    double *R, // (m, n)
    double *dx, // m
    double *dy, // n
    double *p, // m
    double *q, // n
    double *a, // m
    double *b, // n
    double epsilon,
    double lambda1,
    double lambda2,
    int m,
    int n
) {
    return compute_duality_gap<double>(
        C, // (m, n)
        K, // (m, n)
        R, // (m, n)
        dx, // m
        dy, // n
        p, // m
        q, // n
        a, // m
        b, // n
        epsilon,
        lambda1,
        lambda2,
        m,
        n
    );
}

extern "C" void update_k_float(
    float *K, // (m, n)
    float *K_, // (m, n)
    float *C, // (m, n)
    float *u, // (m)
    float *v, // (n)
    float epsilon,
    int m,
    int n
) {
    update_k<float>(
        K, // (m, n)
        K_, // (m, n)
        C, // (m, n)
        u, // (m)
        v, // (n)
        epsilon,
        m,
        n
    );
}

extern "C" void update_k_double(
    double *K, // (m, n)
    double *K_, // (m, n)
    double *C, // (m, n)
    double *u, // (m)
    double *v, // (n)
    double epsilon,
    int m,
    int n
) {
    update_k<double>(
        K, // (m, n)
        K_, // (m, n)
        C, // (m, n)
        u, // (m)
        v, // (n)
        epsilon,
        m,
        n
    );
}

extern "C" void update_R_float(
    float *R, // (m, n)
    float *K, // (m, n)
    float *a, // (m)
    float *b, // (n)
    int m,
    int n
) {
    update_R<float>(
        R,
        K,
        a,
        b,
        m,
        n
    );
}

extern "C" void update_R_double(
    double *R, // (m, n)
    double *K, // (m, n)
    double *a, // (m)
    double *b, // (n)
    int m,
    int n
) {
    update_R<double>(
        R,
        K,
        a,
        b,
        m,
        n
    );
}

extern "C" int step1_process_double(
    double *a,
    double *b,
    double *old_a,
    double *old_b,
    double *K,
    double *C,
    double *dx,
    double *dy,
    double *p,
    double *q,
    double *u,
    double *v,
    int cur_iter,
    int max_iter,
    int iters,
    double tau,
    double lambda1,
    double lambda2,
    double alpha1,
    double alpha2,
    double epsilon,
    int m,
    int n
) {
    return step1_process<double>(
        a,
        b,
        old_a,
        old_b,
        K,
        C,
        dx,
        dy,
        p,
        q,
        u,
        v,
        cur_iter,
        max_iter,
        iters,
        tau,
        lambda1,
        lambda2,
        alpha1,
        alpha2,
        epsilon,
        m,
        n
    );
}

extern "C" double update_process_double(
    double *R,
    double *a,
    double *b,
    double *old_a,
    double *old_b,
    double *K,
    double *_K,
    double *C,
    double *dx,
    double *dy,
    double *p,
    double *q,
    double *u,
    double *v,
    int epsilon_scalings,
    int cur_epsilon_scaling,
    int batch_size,
    double epsilon,
    double threshold,
    double tau,
    double lambda1,
    double lambda2,
    double alpha1,
    double alpha2,
    int cur_iter,
    int max_iter,
    int m,
    int n
) {
    return update_process<double>(
        R,
        a,
        b,
        old_a,
        old_b,
        K,
        _K,
        C,
        dx,
        dy,
        p,
        q,
        u,
        v,
        epsilon_scalings,
        cur_epsilon_scaling,
        batch_size,
        epsilon,
        threshold,
        tau,
        lambda1,
        lambda2,
        alpha1,
        alpha2,
        cur_iter,
        max_iter,
        m,
        n
    );
}
