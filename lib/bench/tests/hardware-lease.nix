{ pkgs, benchLib }:
let
  workload = pkgs.writeText "lease-workload.py" ''
    import os
    import pathlib
    import sys

    pathlib.Path(os.environ["out"], "started").touch()
    assert sys.stdin.buffer.read(1) == b"x"
  '';
  benchmark =
    targetPkgs: features:
    benchLib.mkBenchmark {
      pkgs = targetPkgs;
      name = "hardware-lease-fixture";
      command = [
        "${pkgs.python3}/bin/python3"
        workload
      ];
      requirements.systemFeatures = features;
    };
  command = (benchmark pkgs [ "gfx1151" ]).buildCommand;
  lockPath = "/run/benchmark-gpu.lock";
  # Execute the actual wrapper, substituting only the fixture path and wait.
  script = pkgs.writeText "hardware-lease-fixture.sh" (
    pkgs.lib.replaceStrings [ lockPath "--timeout 600" ] [ "$LEASE_FIXTURE_LOCK" "--timeout 0.2" ]
      command
  );
in
assert pkgs.lib.hasInfix lockPath (benchmark pkgs [ "xdna2" ]).buildCommand;
assert !(pkgs.lib.hasInfix lockPath (benchmark pkgs [ "gfx1030" ]).buildCommand);
assert !(pkgs.lib.hasInfix lockPath (benchmark pkgs [ "rtx4090" ]).buildCommand);
assert !(pkgs.lib.hasInfix lockPath (benchmark pkgs [ ]).buildCommand);
assert
  !(pkgs.lib.hasInfix lockPath
    (benchmark (pkgs // { stdenv.hostPlatform.isLinux = false; }) [ "gfx1151" ]).buildCommand
  );
pkgs.runCommandLocal "ci-benchmark-hardware-lease"
  {
    nativeBuildInputs = [
      pkgs.python3
      pkgs.util-linux
      pkgs.bash
    ];
    meta.maintainers = with pkgs.lib.maintainers; [ georgewhewell ];
  }
  ''
    python3 ${./hardware-lease.py} ${script} > "$out"
  ''
