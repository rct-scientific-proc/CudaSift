#include <math.h>
#include <string.h>
#include <immintrin.h>
#include "cudaSift.h"

/* Portable 32-byte alignment for AVX2 */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  #define ALIGN32 _Alignas(32)
#elif defined(_MSC_VER)
  #define ALIGN32 __declspec(align(32))
#elif defined(__GNUC__)
  #define ALIGN32 __attribute__((aligned(32)))
#else
  #define ALIGN32
#endif

/*
 * Fast approximate 1/sqrt(x) using SSE rsqrt + one Newton-Raphson step.
 * ~23 bits of mantissa accuracy, much faster than 1.0f / sqrtf(x).
 */
static inline float fast_rsqrtf(float x)
{
    __m128 vx  = _mm_set_ss(x);
    __m128 est = _mm_rsqrt_ss(vx);
    /* Newton-Raphson: est *= 1.5 - 0.5 * x * est^2 */
    __m128 half_x = _mm_mul_ss(_mm_set_ss(0.5f), vx);
    __m128 est_sq = _mm_mul_ss(est, est);
    __m128 nr     = _mm_sub_ss(_mm_set_ss(1.5f), _mm_mul_ss(half_x, est_sq));
    return _mm_cvtss_f32(_mm_mul_ss(est, nr));
}

/*
 * Accumulate the rank-1 outer product  M += Y * Y^T * w  using AVX2 FMA.
 * M is 8x8 row-major (32-byte aligned rows), Y is 8-element (32-byte aligned).
 */
static inline void accumulate_outer_avx2(float M[8][8],
                                         const float *Y, float w)
{
    __m256 y_vec = _mm256_load_ps(Y);
    for (int r = 0; r < 8; r++)
    {
        __m256 yr_w  = _mm256_set1_ps(Y[r] * w);
        __m256 m_row = _mm256_load_ps(&M[r][0]);
        m_row = _mm256_fmadd_ps(y_vec, yr_w, m_row);
        _mm256_store_ps(&M[r][0], m_row);
    }
}

/*
 * Accumulate  X += Y * scalar  using AVX2 FMA.
 * Both X and Y are 8-element, 32-byte aligned.
 */
static inline void accumulate_vec_avx2(float *X, const float *Y, float s)
{
    __m256 x_vec = _mm256_load_ps(X);
    __m256 y_vec = _mm256_load_ps(Y);
    __m256 s_vec = _mm256_set1_ps(s);
    x_vec = _mm256_fmadd_ps(y_vec, s_vec, x_vec);
    _mm256_store_ps(X, x_vec);
}

/*
 * Solve the 8x8 linear system M * x = b in-place using Cholesky
 * decomposition (M = L * L^T). M must be symmetric positive-definite.
 *
 * Returns 0 on success, -1 if the matrix is not positive-definite.
 */
static int solve_cholesky_8x8(float M[8][8], float b[8])
{
    float L[8][8];
    memset(L, 0, sizeof(L));

    /* Cholesky factorisation: M = L * L^T */
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < j; k++)
                sum += L[i][k] * L[j][k];

            if (i == j)
            {
                float val = M[i][i] - sum;
                if (val <= 0.0f)
                    return -1; /* not positive-definite */
                L[i][j] = sqrtf(val);
            }
            else
            {
                L[i][j] = (M[i][j] - sum) / L[j][j];
            }
        }
    }

    /* Forward substitution: L * y = b  (y stored in b) */
    for (int i = 0; i < 8; i++)
    {
        float sum = 0.0f;
        for (int k = 0; k < i; k++)
            sum += L[i][k] * b[k];
        b[i] = (b[i] - sum) / L[i][i];
    }

    /* Back substitution: L^T * x = y  (x stored in b) */
    for (int i = 7; i >= 0; i--)
    {
        float sum = 0.0f;
        for (int k = i + 1; k < 8; k++)
            sum += L[k][i] * b[k];
        b[i] = (b[i] - sum) / L[i][i];
    }

    return 0;
}

int ImproveHomography(SiftData *data, float *homography, int numLoops,
                      float minScore, float maxAmbiguity, float thresh)
{
    if (data->h_data == NULL)
        return 0;

    SiftPoint *mpts = data->h_data;
    float limit = thresh * thresh;
    int numPts = data->numPts;

    float A[8]; /* current homography parameters (h9 = 1) */
    float inv_h8 = 1.0f / homography[8];
    for (int i = 0; i < 8; i++)
        A[i] = homography[i] * inv_h8;

    for (int loop = 0; loop < numLoops; loop++)
    {
        ALIGN32 float M[8][8];
        ALIGN32 float X[8];
        ALIGN32 float Y[8];

        /* Zero M and X with AVX2 stores */
        __m256 zero = _mm256_setzero_ps();
        for (int r = 0; r < 8; r++)
            _mm256_store_ps(&M[r][0], zero);
        _mm256_store_ps(X, zero);

        for (int i = 0; i < numPts; i++)
        {
            /* Prefetch next SiftPoint into L1 cache */
            if (i + 1 < numPts)
                _mm_prefetch((const char *)&mpts[i + 1], _MM_HINT_T0);

            SiftPoint *pt = &mpts[i];
            if (pt->score < minScore || pt->ambiguity > maxAmbiguity)
                continue;

            float xp = pt->xpos;
            float yp = pt->ypos;
            float mx = pt->match_xpos;
            float my = pt->match_ypos;

            float den = A[6] * xp + A[7] * yp + 1.0f;
            float inv_den = 1.0f / den;
            float dx = (A[0] * xp + A[1] * yp + A[2]) * inv_den - mx;
            float dy = (A[3] * xp + A[4] * yp + A[5]) * inv_den - my;
            float err_sq = dx * dx + dy * dy;

            /* Huber weight: 1 for inliers, thresh/err for outliers */
            float wei = (err_sq <= limit) ? 1.0f
                                          : thresh * fast_rsqrtf(err_sq);

            /* --- x-equation contribution --- */
            Y[0] = xp;  Y[1] = yp;  Y[2] = 1.0f;
            Y[3] = 0.0f; Y[4] = 0.0f; Y[5] = 0.0f;
            Y[6] = -xp * mx; Y[7] = -yp * mx;

            accumulate_outer_avx2(M, Y, wei);
            accumulate_vec_avx2(X, Y, mx * wei);

            /* --- y-equation contribution --- */
            Y[0] = 0.0f; Y[1] = 0.0f; Y[2] = 0.0f;
            Y[3] = xp;  Y[4] = yp;  Y[5] = 1.0f;
            Y[6] = -xp * my; Y[7] = -yp * my;

            accumulate_outer_avx2(M, Y, wei);
            accumulate_vec_avx2(X, Y, my * wei);
        }

        /* Solve M * A = X via Cholesky */
        if (solve_cholesky_8x8(M, X) != 0)
            break; /* degenerate — keep current A */

        for (int i = 0; i < 8; i++)
            A[i] = X[i];
    }

    /* Count inliers and set per-point match_error */
    int numfit = 0;
    for (int i = 0; i < numPts; i++)
    {
        SiftPoint *pt = &mpts[i];
        float den = A[6] * pt->xpos + A[7] * pt->ypos + 1.0f;
        float dx = (A[0] * pt->xpos + A[1] * pt->ypos + A[2]) / den - pt->match_xpos;
        float dy = (A[3] * pt->xpos + A[4] * pt->ypos + A[5]) / den - pt->match_ypos;
        float err = dx * dx + dy * dy;
        if (err < limit)
            numfit++;
        pt->match_error = sqrtf(err);
    }

    for (int i = 0; i < 8; i++)
        homography[i] = A[i];
    homography[8] = 1.0f;

    return numfit;
}
