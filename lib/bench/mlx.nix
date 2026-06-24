{
  pkgs,
  package,
  target ? null,
  accelerator ? if pkgs.stdenv.isDarwin then "metal" else "rocm",
  hostProfile ? if accelerator == "metal" then "darwin-metal" else "linux-amd-kfd",
  matrixMetadata ? { },
  extraSystemFeatures ? [ ],
  extraSandboxPaths ? [ ],
  n ? 256,
  warmup ? 2,
  iterations ? 5,
}:

assert pkgs.lib.assertMsg (
  accelerator == "metal" || accelerator == "rocm"
) "unsupported MLX benchmark accelerator: ${accelerator}";
assert pkgs.lib.assertMsg (
  accelerator == "metal" || target != null
) "MLX ROCm benchmarks require a target record";

let
  inherit (pkgs) lib;
  benchLib = import ./lib.nix { inherit lib; };

  isMetal = accelerator == "metal";
  backend = accelerator;
  packageRole = if isMetal then "mlx" else "mlx-rocm-${target.packageSuffix}";
  python = pkgs.python3.withPackages (_: [ package ]);
  caseName = if isMetal then "metal-gemm-smoke" else "rocm-${target.packageSuffix}-gemm-smoke";
  derivationName = "mlx-${caseName}";
  metaPlatforms = if isMetal then [ "aarch64-darwin" ] else [ "x86_64-linux" ];
  systemFeatures =
    (
      if isMetal then
        [
          "metal"
          "benchmark"
        ]
      else
        [ target.systemFeature ]
    )
    ++ extraSystemFeatures;
  sandboxPaths =
    (
      if isMetal then
        [ ]
      else
        [
          "/dev/dri"
          "/dev/kfd"
          "/sys/class/drm"
          "/sys/class/kfd"
        ]
    )
    ++ extraSandboxPaths;
  targetMetadata =
    if isMetal then
      {
        packageSuffix = "metal";
        runtimeArch = "metal";
        systemFeature = "metal";
      }
    else
      {
        inherit (target)
          packageSuffix
          runtimeArch
          systemFeature
          ;
      };

  runner = pkgs.writeShellScript "${derivationName}-runner" ''
    set -euo pipefail

    export HOME="$TMPDIR"
    export XDG_CACHE_HOME="$TMPDIR/cache"
    export PYTHONNOUSERSITE=1

    ${python}/bin/python - <<'PY'
    import json
    import math
    import os

    import mlx.core as mx

    if not mx.is_available(mx.gpu):
        raise SystemExit("MLX GPU device is not available")

    mx.set_default_device(mx.gpu)

    n = ${toString n}
    warmup = ${toString warmup}
    iterations = ${toString iterations}
    expected = float(2 * n)

    a = mx.ones((n, n), dtype=mx.float32)
    b = mx.full((n, n), 2.0, dtype=mx.float32)

    for _ in range(warmup):
        mx.eval(a @ b)

    c = None
    for _ in range(iterations):
        c = a @ b
        mx.eval(c)

    value = float(c[0, 0].item())
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-3):
        raise SystemExit(f"bad GEMM result: {value} != {expected}")

    try:
        device_info = mx.device_info(mx.gpu)
    except Exception as exc:
        device_info = {"error": str(exc)}

    print(json.dumps({
        "device": str(mx.default_device()),
        "device_info": device_info,
        "dtype": "float32",
        "expected": expected,
        "iterations": iterations,
        "n": n,
        "op": "matmul",
        "shape": list(c.shape),
        "value": value,
        "verified": True,
    }, default=str, sort_keys=True), flush=True)

    # ROCm can leave non-daemon runtime threads alive after the work is done.
    # The smoke has already synchronized and verified the result, so exit
    # without waiting for backend teardown.
    os._exit(0)
    PY
  '';
in
{
  benchmarks = {
    mlx = {
      ${caseName} = benchLib.mkBenchmark {
        inherit
          pkgs
          package
          ;
        name = derivationName;
        command = [ runner ];
        requirements = {
          inherit
            systemFeatures
            sandboxPaths
            ;
          hostProfiles = [ hostProfile ];
        };
        metadata = lib.recursiveUpdate {
          kind = "mlx-gpu-smoke";
          smokeRevision = 3;
          suite = "mlx";
          inherit
            accelerator
            backend
            ;
          scenario = "gemm-smoke";
          params = {
            inherit
              iterations
              n
              warmup
              ;
            dtype = "float32";
            expected = 2 * n;
            op = "matmul";
          };
          target = targetMetadata;
          tool = {
            inherit
              backend
              packageRole
              ;
            executable = "python";
          };
        } matrixMetadata;
        meta.platforms = metaPlatforms;
        description = "Run MLX ${accelerator} GEMM smoke benchmark";
      };
    };
  };
}
