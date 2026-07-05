{
  amd-aiter,
  lib,
  ...
}:

# Slim override of nixpkgs' ROCm/aiter packaging. Keep the upstream
# derivation and only add local gfx1151 fixes plus runtime CK layout
# compatibility for AITER's JIT code.
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
    # layout directly under 3rdparty/composable_kernel. TheRock sources use
    # the projects/composablekernel layout, so add compatibility links.
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
