#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "cudaSift.h"

/* ────────────────────────────────────────────────────────────────────────
 * AVX2 + FMA gating
 *
 * Both AVX2 and FMA must be advertised by the compiler for the SIMD path
 * to compile.  Otherwise we fall back to a portable scalar implementation.
 *   - MSVC: `/arch:AVX2` defines `__AVX2__` and implicitly enables FMA
 *           intrinsics, so we don't require `__FMA__` there.
 *   - GCC/Clang: require both `__AVX2__` and `__FMA__` (i.e. `-mavx2 -mfma`).
 * ──────────────────────────────────────────────────────────────────────── */
#if defined(__AVX2__) && (defined(_MSC_VER) || defined(__FMA__))
  #define CUSIFT_HAVE_AVX2 1
  #include <immintrin.h>
#else
  #define CUSIFT_HAVE_AVX2 0
#endif

#if CUSIFT_HAVE_AVX2
  /* Portable 32-byte alignment for AVX2 loads/stores */
  #if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #define ALIGN32 _Alignas(32)
  #elif defined(_MSC_VER)
    #define ALIGN32 __declspec(align(32))
  #elif defined(__GNUC__)
    #define ALIGN32 __attribute__((aligned(32)))
  #else
    #define ALIGN32
  #endif
#else
  #define ALIGN32
#endif

/* ────────────────────────────────────────────────────────────────────────
 * Runtime CPU feature detection
 *
 * When the binary was built with the AVX2 path, the entire translation
 * unit may emit AVX2 instructions (the compiler is free to auto-vectorise
 * scalar code under `/arch:AVX2` or `-mavx2`).  Running such a binary on
 * a CPU without AVX2/FMA will fault with an illegal instruction.  This
 * helper returns non-zero when the host CPU advertises AVX2 + FMA + the
 * required OSXSAVE/XGETBV bits, so the caller can abort with a clear
 * diagnostic instead of crashing.
 * ──────────────────────────────────────────────────────────────────────── */
#if CUSIFT_HAVE_AVX2
  #if defined(_MSC_VER)
    #include <intrin.h>
    static int cusift_cpu_supports_avx2_fma(void)
    {
        int regs[4];
        __cpuid(regs, 0);
        if (regs[0] < 7) return 0;

        __cpuid(regs, 1);
        const int has_osxsave = (regs[2] >> 27) & 1;
        const int has_fma     = (regs[2] >> 12) & 1;
        const int has_avx     = (regs[2] >> 28) & 1;
        if (!has_osxsave || !has_fma || !has_avx) return 0;

        /* OS must have enabled XMM (bit 1) and YMM (bit 2) state saving. */
        const unsigned long long xcr0 = _xgetbv(0);
        if ((xcr0 & 0x6ULL) != 0x6ULL) return 0;

        __cpuidex(regs, 7, 0);
        const int has_avx2 = (regs[1] >> 5) & 1;
        return has_avx2;
    }
  #elif defined(__GNUC__) || defined(__clang__)
    static int cusift_cpu_supports_avx2_fma(void)
    {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }
  #else
    static int cusift_cpu_supports_avx2_fma(void) { return 1; }
  #endif

  static int cusift_check_simd_or_warn(void)
  {
      /* Check once; cache the result. */
      static int checked = 0;
      static int ok      = 0;
      if (!checked) {
          ok      = cusift_cpu_supports_avx2_fma();
          checked = 1;
          if (!ok) {
              fprintf(stderr,
                  "cusift: this build of geomFuncs requires AVX2 + FMA, "
                  "but the host CPU does not advertise them. "
                  "Rebuild with -DCUSIFT_ENABLE_AVX2=OFF for a portable "
                  "scalar fallback.\n");
          }
      }
      return ok;
  }
#endif /* CUSIFT_HAVE_AVX2 */

/* ────────────────────────────────────────────────────────────────────────
 * Inner kernels (AVX2 path or scalar fallback)
 * ──────────────────────────────────────────────────────────────────────── */
#if CUSIFT_HAVE_AVX2

/*
 * Accumulate the rank-1 outer product  M += Y * Y^T * w  using AVX2 FMA.
 * M is 8x8 row-major (32-byte aligned rows), Y is 8-element (32-byte aligned).
 */
static inline void accumulate_outer(float M[8][8], const float *Y, float w)
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
static inline void accumulate_vec(float *X, const float *Y, float s)
{
    __m256 x_vec = _mm256_load_ps(X);
    __m256 y_vec = _mm256_load_ps(Y);
    __m256 s_vec = _mm256_set1_ps(s);
    x_vec = _mm256_fmadd_ps(y_vec, s_vec, x_vec);
    _mm256_store_ps(X, x_vec);
}

#else /* !CUSIFT_HAVE_AVX2 — portable scalar fallbacks */

static inline void accumulate_outer(float M[8][8], const float *Y, float w)
{
    for (int r = 0; r < 8; r++) {
        const float yr_w = Y[r] * w;
        for (int c = 0; c < 8; c++)
            M[r][c] += Y[c] * yr_w;
    }
}

static inline void accumulate_vec(float *X, const float *Y, float s)
{
    for (int i = 0; i < 8; i++)
        X[i] += Y[i] * s;
}

#endif /* CUSIFT_HAVE_AVX2 */

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

#if CUSIFT_HAVE_AVX2
    /* Refuse to execute AVX2-compiled code on a CPU that lacks AVX2/FMA
     * rather than letting the program die with an illegal instruction. */
    if (!cusift_check_simd_or_warn())
        return 0;
#endif

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

#if CUSIFT_HAVE_AVX2
        /* Zero M and X with AVX2 stores */
        __m256 zero = _mm256_setzero_ps();
        for (int r = 0; r < 8; r++)
            _mm256_store_ps(&M[r][0], zero);
        _mm256_store_ps(X, zero);
#else
        memset(M, 0, sizeof(M));
        memset(X, 0, sizeof(X));
#endif

        for (int i = 0; i < numPts; i++)
        {
#if CUSIFT_HAVE_AVX2
            /* Prefetch next SiftPoint into L1 cache */
            if (i + 1 < numPts)
                _mm_prefetch((const char *)&mpts[i + 1], _MM_HINT_T0);
#endif

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

            /* Binary weight */
            float wei = (err_sq <= limit) ? 1.0f : 0.0f;

            /* Outliers contribute nothing (0 * Y * Y^T == 0); skip the
             * 16 outer-product FMAs and 4 vector FMAs entirely.  This is
             * numerically identical to the previous behaviour but cheaper,
             * and after the first iteration with a marginal initial
             * homography most points are outliers. */
            if (wei == 0.0f)
                continue;

            /* --- x-equation contribution --- */
            Y[0] = xp;  Y[1] = yp;  Y[2] = 1.0f;
            Y[3] = 0.0f; Y[4] = 0.0f; Y[5] = 0.0f;
            Y[6] = -xp * mx; Y[7] = -yp * mx;

            accumulate_outer(M, Y, wei);
            accumulate_vec(X, Y, mx * wei);

            /* --- y-equation contribution --- */
            Y[0] = 0.0f; Y[1] = 0.0f; Y[2] = 0.0f;
            Y[3] = xp;  Y[4] = yp;  Y[5] = 1.0f;
            Y[6] = -xp * my; Y[7] = -yp * my;

            accumulate_outer(M, Y, wei);
            accumulate_vec(X, Y, my * wei);
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
        float inv_den = 1.0f / den;
        float dx = (A[0] * pt->xpos + A[1] * pt->ypos + A[2]) * inv_den - pt->match_xpos;
        float dy = (A[3] * pt->xpos + A[4] * pt->ypos + A[5]) * inv_den - pt->match_ypos;
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
