{
  amd-aiter,
  lib,
  ...
}:

# Slim override of `nixpkgs.python3Packages.amd-aiter`. Goal is to layer on
# top, not fork the whole derivation, so we keep upstream improvements and only
# carry our own deltas.
#
# What we override:
#
#   1. `patches`: extension point for the gfx1151 (RDNA 3.5) JIT-compile
#      fixes paudley's tree carries — `vec_convert.h` scalar fallback
#      for the CDNA-only `v_pk_mul_f32` / `v_cvt_pk_fp8_f32` inline asm,
#      and `hip_reduce.h` `ds_swizzle` fallback for DPP `row_bcast`.
#      Initially empty; drop .patch files into `pkgs/amd-aiter/patches/`
#      and they get picked up here.
#
#   2. `passthru.tests` JIT arch: upstream defaults to `gfx942` (CDNA3),
#      but our hardware is `gfx1151`. Building the JIT smoke-test wheel
#      against gfx942 doesn't tell us anything useful for Strix Halo;
#      retarget it. (Test only fires when torch.rocmSupport is on, so
#      this is a no-op for CPU-only torch builds.)

amd-aiter.overridePythonAttrs (old: {
  patches =
    (old.patches or [ ])
    ++ (
      let
        patchDir = ./patches;
        entries = builtins.attrNames (builtins.readDir patchDir);
        patches = builtins.filter (n: lib.hasSuffix ".patch" n) entries;
      in
      map (p: patchDir + "/${p}") patches
    );

  postInstall = (old.postInstall or "") + ''
    # AITER's JIT code still looks for the pre-ROCm-7 Composable Kernel
    # layout directly under 3rdparty/composable_kernel:
    #
    #   3rdparty/composable_kernel/include/ck_tile/core.hpp
    #   3rdparty/composable_kernel/library/include
    #   3rdparty/composable_kernel/example/ck_tile/...
    #
    # ROCm 7 packages CK under projects/composablekernel instead. Keep the
    # upstream source tree intact and add compatibility links so runtime JIT
    # builds do not fail looking for ck_tile headers or generator scripts.
    for site in "$out"/lib/python*/site-packages; do
      ck="$site/aiter_meta/3rdparty/composable_kernel"
      ck_project="$ck/projects/composablekernel"
      if [[ -d "$ck_project" ]]; then
        ln -sfn "$ck_project/include" "$ck/include"
        ln -sfn "$ck_project/library" "$ck/library"
        ln -sfn "$ck_project/example" "$ck/example"
      fi
    done
  '';

  passthru = (old.passthru or { }) // {
    # Re-anchor the JIT smoke tests on gfx1151. Falls through to
    # upstream's other passthru attrs (vllm-flash-attn etc.) if any.
    tests = lib.mapAttrs (
      _: drv:
      drv.overrideAttrs (oldDrv: {
        env = (oldDrv.env or { }) // {
          GPU_ARCHS = "gfx1151";
          PYTORCH_ROCM_ARCH = "gfx1151";
        };
      })
    ) (old.passthru.tests or { });
  };
})
