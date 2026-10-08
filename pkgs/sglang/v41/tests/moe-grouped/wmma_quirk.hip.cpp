// Which operand patterns in the OTHER K slots (A zero there) disturb a single exact product? A one-hot at `slot`, B pattern selected by `pat`, A zeros +0 or -0.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
typedef unsigned int u32x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)
__global__ void k(const unsigned short* av, const unsigned short* bd, float* out, int slot, unsigned azero) {
  const int lane = threadIdx.x, r = lane & 15, upper = lane >> 4;
  u32x8 a, b;
  for (int j = 0; j < 8; ++j) { a[j] = azero | (azero << 16); b[j] = (unsigned)bd[r * 16 + 2 * j] | ((unsigned)bd[r * 16 + 2 * j + 1] << 16); }
  { unsigned v = a[slot >> 1]; if (slot & 1) v = (v & 0xFFFFu) | ((unsigned)av[r] << 16); else v = (v & 0xFFFF0000u) | av[r]; a[slot >> 1] = v; }
  f32x8 c = {0, 0, 0, 0, 0, 0, 0, 0};
  c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(bf16x16, a), __builtin_bit_cast(bf16x16, b), c);
  for (int i = 0; i < 8; ++i) out[(2 * i + upper) * 16 + r] = c[i];
}
static float bf2f(unsigned short b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static unsigned short f2bfx(float f) { unsigned u; memcpy(&u, &f, 4); return (unsigned short)(u >> 16); }
int main() {
  std::mt19937 rng(5);
  unsigned short *dav, *dbd; float* dout; CK(hipMalloc(&dav, 32)); CK(hipMalloc(&dbd, 512)); CK(hipMalloc(&dout, 1024));
  const char* pn[] = {"other weights +0", "other weights -0", "other weights positive", "other weights negative", "other weights mixed sign", "other weights mixed incl zeros"};
  for (unsigned azero : {0x0000u, 0x8000u}) {
    for (int pat = 0; pat < 6; ++pat) {
      long bad = 0, total = 0, badup = 0, badlow = 0;
      for (int trial = 0; trial < 1500; ++trial) {
        unsigned short av[16], bd[256]; const int slot = (int)(rng() % 16);
        for (int i = 0; i < 16; ++i) av[i] = (unsigned short)(0x3B80 + rng() % 0x0200);
        float prod_w[16];
        for (int c = 0; c < 16; ++c) {
          for (int j = 0; j < 16; ++j) {
            static const float mm[7] = {0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
            float w = mm[rng() % 7] * std::ldexp(1.0f, (int)(rng() % 11) - 10);
            if (j != slot) switch (pat) { case 0: w = 0.f; break; case 1: w = -0.f; break; case 2: break; case 3: w = -w; break; case 4: if (rng() & 1) w = -w; break;
              case 5: { const unsigned q = rng() % 4; if (q == 0) w = 0.f; else if (q == 1) w = -0.f; else if (rng() & 1) w = -w; } break; }
            else if (rng() & 1) w = -w;   // the product slot: random sign
            bd[c * 16 + j] = f2bfx(w);
            if (j == slot) prod_w[c] = w;
          }
        }
        CK(hipMemcpy(dav, av, 32, hipMemcpyHostToDevice)); CK(hipMemcpy(dbd, bd, 512, hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k, dim3(1), dim3(32), 0, 0, dav, dbd, dout, slot, azero);
        float out[256]; CK(hipMemcpy(out, dout, 1024, hipMemcpyDeviceToHost));
        for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) {
          const float exact = bf2f(av[r]) * prod_w[c];
          ++total;
          if (memcmp(&out[r * 16 + c], &exact, 4)) { ++bad; if (r & 1) ++badup; else ++badlow; }
        }
      }
      printf("A zeros %s, %-36s: %6ld of %ld differ (%.3f%%; even rows %ld, odd rows %ld)\n", azero ? "-0" : "+0", pn[pat], bad, total, 100.0 * bad / total, badlow, badup);
    }
  }
}
