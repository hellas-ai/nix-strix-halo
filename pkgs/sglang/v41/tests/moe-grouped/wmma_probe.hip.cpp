// Hardware facts about v_wmma_f32_16x16x16_bf16 on gfx1151 that the grouped MoE kernel relies on or could exploit.
//   (1) slot order: is D unchanged, bit for bit, when the 16 K slots of A and B are permuted the same way?
//       (the kernel feeds each WMMA the even-K half then the odd-K half of 16 consecutive K)
//   (2) upper half-wave: are lanes 16..31 of A and B ignored (so they may hold garbage / other data)?
//   (3) rounding: how does D compare with the correctly rounded fp32 sum of the 16 exact products plus C?  (informational)
//   (4) chaining: is D(chain of two WMMAs over K 0..15, 16..31) equal to sequential WMMA with the same operands? (always; sanity)
// Build: therock-hip-clang++ -x hip --offload-arch=gfx1151 -O2 wmma_probe.hip.cpp -o wmma_probe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <cstdint>

typedef unsigned int u32x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)

static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

// Each lane l gets A row (l & 15) and B column (l & 15) from the arrays; mode selects what lanes 16..31 hold.
//   mode 0: duplicates of lanes 0..15; mode 1: garbage in lanes 16..31 (A and B)
// perm: slot order applied to both A and B: slot s holds K index perm[s].
__global__ void probe(const uint16_t* A, const uint16_t* B, const float* C, float* D, const int* perm, int mode, int trials) {
  const int lane = threadIdx.x;
  const int t = blockIdx.x;
  if (t >= trials) return;
  const int r = lane & 15, upper = lane >> 4;
  u32x8 a, b;
  for (int i = 0; i < 8; ++i) {
    uint16_t a0, a1, b0, b1;
    const int k0 = perm[2 * i], k1 = perm[2 * i + 1];
    a0 = A[((size_t)t * 16 + r) * 16 + k0]; a1 = A[((size_t)t * 16 + r) * 16 + k1];
    b0 = B[((size_t)t * 16 + r) * 16 + k0]; b1 = B[((size_t)t * 16 + r) * 16 + k1];
    if (upper && (mode == 1 || mode == 2)) { a0 = 0x7F00 ^ (uint16_t)(lane * 977 + i); a1 = 0x5F13 ^ (uint16_t)(lane * 31); }   // garbage A in lanes 16..31
    if (upper && (mode == 1 || mode == 3)) { b0 = 0x6E01 ^ (uint16_t)(lane * 7); b1 = 0xFEED; }                                  // garbage B in lanes 16..31
    a[i] = (unsigned)a0 | ((unsigned)a1 << 16);
    b[i] = (unsigned)b0 | ((unsigned)b1 << 16);
  }
  f32x8 c;
  for (int i = 0; i < 8; ++i) c[i] = C[((size_t)t * 16 + 2 * i + upper) * 16 + r];
  f32x8 d = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(bf16x16, a), __builtin_bit_cast(bf16x16, b), c);
  for (int i = 0; i < 8; ++i) D[((size_t)t * 16 + 2 * i + upper) * 16 + r] = d[i];
}

int main(int argc, char** argv) {
  const int trials = argc > 1 ? atoi(argv[1]) : 4000;
  std::mt19937 rng(7);
  std::vector<uint16_t> A((size_t)trials * 256), B((size_t)trials * 256);
  std::vector<float> C((size_t)trials * 256);
  // data with wide exponent spread and cancellation: values +-1.xx * 2^e, e in [-20, 20]; sometimes exactly cancelling pairs
  auto rnd_bf16 = [&]() {
    const int e = (int)(rng() % 41) - 20 + 127;
    const uint16_t sign = (rng() & 1) ? 0x8000 : 0;
    return (uint16_t)(sign | (e << 7) | (rng() & 0x7F));
  };
  for (size_t i = 0; i < A.size(); ++i) { A[i] = rnd_bf16(); B[i] = rnd_bf16(); }
  for (size_t i = 0; i < C.size(); ++i) { const float v = bf2f(rnd_bf16()); C[i] = (rng() % 4 == 0) ? 0.f : v * (float)(1 + rng() % 5); }
  uint16_t *dA, *dB; float *dC, *dD0, *dD1, *dD2, *dD3; int *dp0, *dp1;
  CK(hipMalloc(&dA, A.size() * 2)); CK(hipMalloc(&dB, B.size() * 2)); CK(hipMalloc(&dC, C.size() * 4));
  CK(hipMalloc(&dD0, C.size() * 4)); CK(hipMalloc(&dD1, C.size() * 4)); CK(hipMalloc(&dD2, C.size() * 4)); CK(hipMalloc(&dD3, C.size() * 4));
  CK(hipMalloc(&dp0, 64)); CK(hipMalloc(&dp1, 64));
  CK(hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dB, B.data(), B.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(dC, C.data(), C.size() * 4, hipMemcpyHostToDevice));
  // slot permutations (slot s holds K index perm[s]) applied identically to A and B
  const int NPERM = 6;
  const char* pname[NPERM] = {"identity", "swap within pairs (s^1)", "even-K then odd-K (the kernel's KPERM)", "reverse", "pairs reversed (pair order, not pair content)", "rotate by 2 slots"};
  int perms[NPERM][16];
  for (int s = 0; s < 16; ++s) {
    perms[0][s] = s;
    perms[1][s] = s ^ 1;
    perms[2][s] = s < 8 ? 2 * s : 2 * (s - 8) + 1;
    perms[3][s] = 15 - s;
    perms[4][s] = (14 - (s & ~1)) | (s & 1);
    perms[5][s] = (s + 2) & 15;
  }
  int* dp[NPERM];
  for (int q = 0; q < NPERM; ++q) { CK(hipMalloc(&dp[q], 64)); CK(hipMemcpy(dp[q], perms[q], 64, hipMemcpyHostToDevice)); }
  float* dD[NPERM + 3];
  for (int q = 0; q < NPERM + 3; ++q) CK(hipMalloc(&dD[q], C.size() * 4));
  for (int q = 0; q < NPERM; ++q) hipLaunchKernelGGL(probe, dim3(trials), dim3(32), 0, 0, dA, dB, dC, dD[q], dp[q], 0, trials);
  for (int m = 1; m <= 3; ++m) hipLaunchKernelGGL(probe, dim3(trials), dim3(32), 0, 0, dA, dB, dC, dD[NPERM + m - 1], dp[0], m, trials);
  CK(hipDeviceSynchronize());
  std::vector<std::vector<float>> D(NPERM + 3, std::vector<float>(C.size()));
  for (int q = 0; q < NPERM + 3; ++q) CK(hipMemcpy(D[q].data(), dD[q], C.size() * 4, hipMemcpyDeviceToHost));
  long diff_exact = 0, diff_seq = 0, diff_pairseq = 0;
  std::vector<long> dperm(NPERM, 0), dupper(3, 0), dupper_even(3, 0), dupper_odd(3, 0);
  for (int t = 0; t < trials; ++t)
    for (int row = 0; row < 16; ++row)
      for (int col = 0; col < 16; ++col) {
        const size_t i = ((size_t)t * 16 + row) * 16 + col;
        for (int q = 1; q < NPERM; ++q) if (memcmp(&D[0][i], &D[q][i], 4)) ++dperm[q];
        for (int m = 0; m < 3; ++m) if (memcmp(&D[0][i], &D[NPERM + m][i], 4)) { ++dupper[m]; if (row & 1) ++dupper_odd[m]; else ++dupper_even[m]; }
        // reference: sum_k A[t][row][k] * B[t][col][k] + C (the kernel's operands: A row = output row, B row = output column)
        double acc = C[i];
        float seq = C[i], pseq = C[i];
        for (int k = 0; k < 16; ++k) {
          const float x = bf2f(A[((size_t)t * 16 + row) * 16 + k]), y = bf2f(B[((size_t)t * 16 + col) * 16 + k]);
          acc += (double)x * (double)y;
          seq = fmaf(x, y, seq);
        }
        for (int p2 = 0; p2 < 8; ++p2) {   // pairwise: acc = (acc + x0*y0 + x1*y1) with the pair summed exactly, one rounding per pair
          const float x0 = bf2f(A[((size_t)t * 16 + row) * 16 + 2 * p2]), y0 = bf2f(B[((size_t)t * 16 + col) * 16 + 2 * p2]);
          const float x1 = bf2f(A[((size_t)t * 16 + row) * 16 + 2 * p2 + 1]), y1 = bf2f(B[((size_t)t * 16 + col) * 16 + 2 * p2 + 1]);
          pseq = (float)((double)pseq + (double)x0 * (double)y0 + (double)x1 * (double)y1);
        }
        const float exact = (float)acc;
        if (memcmp(&D[0][i], &exact, 4)) ++diff_exact;
        if (memcmp(&D[0][i], &seq, 4)) ++diff_seq;
        if (memcmp(&D[0][i], &pseq, 4)) ++diff_pairseq;
      }
  const long total = (long)trials * 256;
  printf("trials=%d outputs=%ld\n", trials, total);
  for (int q = 1; q < NPERM; ++q)
    printf("(1) slot permutation '%s' changes D bits in %ld outputs (%.3f%%)\n", pname[q], dperm[q], 100.0 * dperm[q] / total);
  const char* un[3] = {"A and B", "A only", "B only"};
  for (int m = 0; m < 3; ++m)
    printf("(2) garbage %s in lanes 16..31 changes D bits in %ld outputs (even rows %ld, odd rows %ld)  -> %s\n", un[m], dupper[m], dupper_even[m], dupper_odd[m],
           dupper[m] ? "upper half-wave IS read" : "ignored");
  printf("(3) D differs from the correctly rounded exact sum in %ld outputs (%.2f%%)\n", diff_exact, 100.0 * diff_exact / total);
  printf("(3b) D differs from a sequential fp32 fma chain in %ld outputs (%.2f%%)\n", diff_seq, 100.0 * diff_seq / total);
  printf("(3c) D differs from 8 sequential exact-pair sums (one rounding per K pair) in %ld outputs (%.2f%%)\n", diff_pairseq, 100.0 * diff_pairseq / total);
  return 0;
}
