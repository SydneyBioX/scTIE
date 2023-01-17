
void add_op(
    float *out, float *lhs, float *rhs, int size
) {
    for (int i = 0; i < size; i++) {
        out[i] = lhs[i] + rhs[i];
    }
}

void sub_op(
    float *out, float *lhs, float *rhs, int size
) {
    for (int i = 0; i < size; i++) {
        out[i] = lhs[i] - rhs[i];
    }
}

void mul_op(
    float *out, float *lhs, float *rhs, int size
) {
    for (int i = 0; i < size; i++) {
        out[i] = lhs[i] * rhs[i];
    }
}

void log_op(
    float *out, float *in, int size
) {
    for (int i = 0; i < size; i++) {
        out[i] = logf(in[i]);
    }
}

float sum_op(
    float *in, int size
) {
    float r = 0;
    for (int i = 0; i < size; i++) {
        r += in[i];
    }
    return r;
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
    //I = len(p)
    //J = len(q)
    //F1 = lambda x, y: fdiv(lambda1, x, p, y)
    //F2 = lambda x, y: fdiv(lambda2, x, q, y)
    //with np.errstate(divide="ignore"):
    //    return (
    //        F1(np.dot(R, dy), dx)
    //        + F2(np.dot(R.T, dx), dy)
    //        + (epsilon * np.sum(R * np.nan_to_num(np.log(R)) - R + K) + np.sum(R * C)) / (I * J)
    //    )

    float *t1 = (float*)malloc(m * sizeof(float));
    float *t2 = (float*)malloc(n * sizeof(float));
    float* Rt = (float*)malloc(m * n * sizeof(float));

    transpose(Rt, R, m, n);

    // dot(R, dy)
    gemv(t1, R, dy, m, n);
    // dot(Rt, dx)
    gemv(t2, Rt, dx, n, m);

    // np.sum(R * np.nan_to_num(np.log(R)) - R + K)
    float t3 = 0;
    for (int i = 0; i < m * n; i++) {
        t3 += R[i] * nan_to_num(logf(R[i])) - R[i] + K[i];
    }

    // np.sum(R * C)
    float t4 = 0;
    for (int i = 0; i < m * n; i++) {
        t4 += R[i] * C[i];
    }


    float ret = fdiv(lambda1, t1, p, dx, m)
              + fdiv(lambda2, t2, q, dy, n)
              + (epsilon * t3 + t4) / (float)(m * n);

    free(t1);
    free(t2);
    free(Rt);

    return ret;
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
    //I = len(p)
    //J = len(q)
    //F1 = lambda x, y: fdiv(lambda1, x, p, y)
    //F2 = lambda x, y: fdiv(lambda2, x, q, y)
    //with np.errstate(divide="ignore"):
    //    return (
    //        F1(np.dot(R, dy), dx)
    //        + F2(np.dot(R.T, dx), dy)
    //        + (epsilon * np.sum(R * np.nan_to_num(np.log(R)) - R + K) + np.sum(R * C)) / (I * J)
    //    )

    double *t1 = (double*)malloc(m * sizeof(double));
    double *t2 = (double*)malloc(n * sizeof(double));
    double* Rt = (double*)malloc(m * n * sizeof(double));

    transpose(Rt, R, m, n);

    // dot(R, dy)
    gemv(t1, R, dy, m, n);
    // dot(Rt, dx)
    gemv(t2, Rt, dx, n, m);

    // np.sum(R * np.nan_to_num(np.log(R)) - R + K)
    double t3 = 0;
    for (int i = 0; i < m * n; i++) {
        t3 += R[i] * nan_to_num(std::log(R[i])) - R[i] + K[i];
    }

    // np.sum(R * C)
    double t4 = 0;
    for (int i = 0; i < m * n; i++) {
        t4 += R[i] * C[i];
    }


    double ret = fdiv(lambda1, t1, p, dx, m)
               + fdiv(lambda2, t2, q, dy, n)
               + (epsilon * t3 + t4) / (double)(m * n);

    free(t1);
    free(t2);
    free(Rt);

    return ret;
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
    //I = len(p)
    //J = len(q)
    //F1c = lambda u, v: fdivstar(lambda1, u, p, v)
    //F2c = lambda u, v: fdivstar(lambda2, u, q, v)
    //return -F1c(-epsilon * np.log(a), dx) - F2c(-epsilon * np.log(b), dy) - epsilon * np.sum(R - K) / (I * J)

    float *t1 = (float*)malloc(m * sizeof(float));
    float *t2 = (float*)malloc(n * sizeof(float));

    //-epsilon * np.log(a)
    for (int i = 0; i < m; i++) {
        t1[i] = -epsilon * logf(a[i]);
    }

    //-epsilon * np.log(b)
    for (int i = 0; i < n; i++) {
        t2[i] = -epsilon * logf(b[i]);
    }

    // np.sum(R - K)
    float t3 = 0;
    for (int i = 0; i < m * n; i++) {
        t3 += R[i] - K[i];
    }

    //float ret = - fdivstar(lambda1, t1, p, dx, m)
    //            - fdivstar(lambda2, t2, q, dy, n)
    //            - epsilon * t3 / (float)(m * n);

    float term1 = - fdivstar(lambda1, t1, p, dx, m);
    float term2 = - fdivstar(lambda2, t2, q, dy, n);
    float term3 = - epsilon * t3 / (float)(m * n);

    //fprintf(stdout, "C %d %d %f %f %f %f %f\n", m, n, lambda1, lambda2, term1, term2, term3);
    //fflush(stdout);

    float ret = term1 + term2 + term3;

    free(t1);
    free(t2);

    return ret;
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
    //I = len(p)
    //J = len(q)
    //F1c = lambda u, v: fdivstar(lambda1, u, p, v)
    //F2c = lambda u, v: fdivstar(lambda2, u, q, v)
    //return -F1c(-epsilon * np.log(a), dx) - F2c(-epsilon * np.log(b), dy) - epsilon * np.sum(R - K) / (I * J)

    //double *t1 = (double*)malloc(m * sizeof(double));
    //double *t2 = (double*)malloc(n * sizeof(double));

    ////-epsilon * np.log(a)
    //for (int i = 0; i < m; i++) {
    //    t1[i] = -epsilon * std::log(a[i]);
    //}

    ////-epsilon * np.log(b)
    //for (int i = 0; i < n; i++) {
    //    t2[i] = -epsilon * std::log(b[i]);
    //}

    // np.sum(R - K)
    double t3 = 0;
    for (int i = 0; i < m * n; i++) {
        t3 += R[i] - K[i];
    }

    //double term1 = - fdivstar<double>(lambda1, t1, p, dx, m);
    //double term2 = - fdivstar<double>(lambda2, t2, q, dy, n);
    //double term3 = - epsilon * t3 / (double)(m * n);

    double term1 = - fdivstarexp(lambda1, epsilon, a, p, dx, m);
    double term2 = - fdivstarexp(lambda2, epsilon, b, q, dy, n);
    double term3 = - epsilon * t3 / (double)(m * n);

    //fprintf(stdout, "Cdouble %d %d %f %f %f %f %f\n", m, n, lambda1, lambda2, term1, term2, term3);
    //fflush(stdout);

    double ret = term1 + term2 + term3;

    //free(t1);
    //free(t2);

    return ret;
}
