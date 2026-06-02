# NixOS module for machines that execute benchmark derivations.
#
# Benchmarks define their workload inputs. This module only advertises
# host capabilities to Nix and exposes the matching devices in the sandbox.
{
  config,
  lib,
  pkgs,
  ...
}:

let
  inherit (lib)
    any
    concatMap
    concatStringsSep
    filter
    getBin
    hasPrefix
    mapAttrsToList
    mkIf
    mkOption
    optionals
    stringAfter
    types
    unique
    ;

  common = import ./benchmark-common.nix { inherit lib pkgs; };
  cfg = config.benchmark;

  setfacl = "${getBin pkgs.acl}/bin/setfacl";

  # Grant rw to the Nix build sandbox via a `g:nixbld:rw` POSIX ACL.
  # The build user's primary gid is `nixbld`, which survives the user
  # namespace, so this ACL match works without touching the device's
  # default ownership/mode — upstream `video`/`render` groups and
  # seat-ACL flows for interactive sessions remain intact.
  mkUdevAcl = devicePath: ''RUN+="${setfacl} -m g:nixbld:rw ${devicePath}"'';
  mkShellAcl = pathVar: ''${setfacl} -m g:nixbld:rw -- "${pathVar}" 2>/dev/null || true'';
  enabledRunnerEntries = filter (entry: entry.value.enable) (
    mapAttrsToList (name: value: { inherit name value; }) cfg.runners
  );
  enabledRunners = map (entry: entry.value) enabledRunnerEntries;
  hasEnabledRunner = enabledRunners != [ ];

  runnerFeatures =
    runner:
    runner.systemFeatures
    ++ map common.gpuFeature runner.gpus
    ++ map common.npuFeature runner.npus
    ++ optionals runner.rdma.enable runner.rdma.systemFeatures;

  modelsPath = toString cfg.modelsPath;

  gpus = concatMap (runner: runner.gpus) enabledRunners;
  npus = concatMap (runner: runner.npus) enabledRunners;

  hasGpu = gpus != [ ];
  hasAmdGpu = any (gpu: gpu.type == "amd") gpus;
  hasNvidiaGpu = any (gpu: gpu.type == "nvidia") gpus;
  hasNpu = npus != [ ];
  hasAmdNpu = any (npu: npu.type == "amd") npus;
  hasRdma = any (runner: runner.rdma.enable) enabledRunners;

  optionalSandboxPaths = paths: map (path: "${path}?") paths;

  # Pair with lib/bench/lib.nix: every benchmark derivation requires the
  # `benchmark` system feature. Hosts that enable any runner advertise it
  # so the scheduler can find them.
  systemFeatures = unique ([ "benchmark" ] ++ concatMap runnerFeatures enabledRunners);
  extraSandboxPaths = unique (
    [
      modelsPath
    ]
    ++ concatMap (runner: runner.extraSandboxPaths) enabledRunners
    ++ optionalSandboxPaths (
      optionals (hasGpu || hasNpu || hasRdma) [
        "/dev/shm"
        "/proc"
        "/sys/dev"
        "/sys/devices"
      ]
    )
    ++ optionalSandboxPaths (
      optionals hasGpu [
        "/dev/dri"
        "/sys/class/drm"
      ]
    )
    ++ optionalSandboxPaths (
      optionals hasAmdGpu [
        "/dev/kfd"
        "/sys/bus/pci/devices"
        "/sys/class/hwmon"
        "/sys/class/kfd"
        "/sys/class/net"
        "/sys/class/scsi_host"
      ]
    )
    ++ optionalSandboxPaths (
      optionals hasAmdNpu [
        "/dev/accel"
        "/sys/class/accel"
      ]
    )
    ++ optionalSandboxPaths (
      optionals hasRdma [
        "/dev/infiniband"
        "/sys/class/infiniband"
        "/sys/class/infiniband_verbs"
        "/sys/class/net"
      ]
    )
    ++ optionalSandboxPaths (
      optionals hasNvidiaGpu [
        "/dev/nvidia0"
        "/dev/nvidiactl"
        "/dev/nvidia-uvm"
        "/dev/nvidia-uvm-tools"
        "/dev/nvidia-caps"
      ]
    )
  );

  relaxSandbox = any (runner: runner.relaxSandbox) enabledRunners;
  deviceAclGlobs =
    optionals hasAmdGpu [
      "/dev/kfd"
    ]
    ++ optionals hasGpu [
      "/dev/dri/card*"
      "/dev/dri/renderD*"
    ]
    ++ optionals hasAmdNpu [
      "/dev/accel/accel*"
    ]
    ++ optionals hasRdma [
      "/dev/infiniband/*"
    ]
    ++ optionals hasNvidiaGpu [
      "/dev/nvidia[0-9]*"
      "/dev/nvidiactl"
      "/dev/nvidia-uvm"
      "/dev/nvidia-uvm-tools"
    ];
  applyDeviceAcls = concatStringsSep "\n" (
    map (glob: ''
      for path in ${glob}; do
        [ -e "$path" ] || continue
        ${mkShellAcl "$path"}
      done
    '') deviceAclGlobs
  );

  isIommuParam =
    param:
    hasPrefix "iommu=" param
    || hasPrefix "amd_iommu=" param
    || hasPrefix "intel_iommu=" param
    || hasPrefix "iommu.passthrough=" param;
  iommuParams = filter isIommuParam config.boot.kernelParams;
  iommuOff = iommuParams == [ "iommu=off" ];

  runnerModule = types.submodule (
    { name, ... }:
    {
      options = {
        enable = mkOption {
          type = types.bool;
          default = true;
          description = "Whether to include this benchmark runner in host setup.";
        };

        gpus = mkOption {
          type = types.listOf common.gpuModule;
          default = [ ];
          example = [
            {
              type = "amd";
              arch = "1151";
            }
          ];
          description = "GPU capabilities available to benchmark derivations.";
        };

        npus = mkOption {
          type = types.listOf common.npuModule;
          default = [ ];
          example = [
            {
              type = "amd";
              arch = "xdna2";
            }
          ];
          description = "NPU capabilities available to benchmark derivations.";
        };

        systemFeatures = mkOption {
          type = types.listOf types.str;
          default = [ ];
          example = [ "big-memory" ];
          description = "Additional Nix system features advertised by this runner.";
        };

        rdma = {
          enable = mkOption {
            type = types.bool;
            default = false;
            description = "Whether this runner exposes RDMA devices to benchmark derivations.";
          };

          systemFeatures = mkOption {
            type = types.listOf types.str;
            default = [
              "rdma"
              "rdma-usb4"
            ];
            description = "Nix system features advertised when RDMA support is enabled.";
          };
        };

        extraSandboxPaths = mkOption {
          type = types.listOf types.str;
          default = [ ];
          example = [ "/mnt/bench-data" ];
          description = "Additional paths bind-mounted into benchmark build sandboxes.";
        };

        relaxSandbox = mkOption {
          type = types.bool;
          default = false;
          description = "Whether this runner requires `nix.settings.sandbox = relaxed`.";
        };

        requireIommuOff = mkOption {
          type = types.bool;
          default = true;
          description = ''
            Assert that the host boots with `iommu=off`. Strix Halo GPU
            benchmark hosts use this by default because IOMMU translation
            measurably slows GPU-memory workloads. Set this to false only
            for runners that intentionally need IOMMU, such as NPU hosts.
          '';
        };

        description = mkOption {
          type = types.str;
          default = name;
          description = "Human-readable runner description.";
        };
      };
    }
  );
in
{
  imports = [
    ./benchmark-executor.nix
  ];

  options.benchmark = {
    modelsPath = mkOption {
      type = types.path;
      default = "/models";
      description = ''
        Host path containing benchmark model files. Benchmark runners create
        this directory if needed and bind it read-only into Nix build
        sandboxes. Individual benchmark derivations still define the model
        files they expect beneath this root.
      '';
    };

    runners = mkOption {
      type = types.attrsOf runnerModule;
      default = { };
      description = "Named local benchmark runner capabilities.";
    };
  };

  config = mkIf hasEnabledRunner {
    assertions =
      map (entry: {
        assertion = !entry.value.requireIommuOff || iommuOff;
        message = ''
          benchmark.runners.${entry.name} requires IOMMU disabled, but boot.kernelParams has IOMMU settings: ${concatStringsSep " " iommuParams}
          Use boot.kernelParams = [ "iommu=off" ] for Strix Halo benchmark runners, or set benchmark.runners.${entry.name}.requireIommuOff = false for an IOMMU-dependent runner.
        '';
      }) enabledRunnerEntries
      ++ map (entry: {
        assertion = !entry.value.requireIommuOff || entry.value.npus == [ ];
        message = ''
          benchmark.runners.${entry.name} declares NPUs while requireIommuOff=true.
          AMD NPU support requires IOMMU passthrough; do not advertise NPUs from an IOMMU-off benchmark runner.
        '';
      }) enabledRunnerEntries;

    nix.settings = {
      "system-features" = systemFeatures;
      "extra-sandbox-paths" = extraSandboxPaths;
      sandbox = mkIf relaxSandbox "relaxed";
    };

    systemd.tmpfiles.rules = [
      "d ${modelsPath} 0755 root root -"
    ];

    # At boot, the udev rules below fire `RUN+= setfacl` when each device
    # appears. On `nixos-rebuild switch` (no reboot), existing device nodes
    # were created under the previous udev rules; this activation script
    # re-applies the ACL to those. `users` is the dependency because
    # setfacl resolves `nixbld` against /etc/group.
    system.activationScripts.benchmark-runner-device-acls = mkIf (deviceAclGlobs != [ ]) (
      stringAfter [ "users" ] applyDeviceAcls
    );

    services.udev.extraRules = concatStringsSep "\n" (
      optionals hasAmdGpu [
        ''KERNEL=="kfd", ${mkUdevAcl "/dev/$kernel"}''
      ]
      ++ optionals hasAmdNpu [
        ''SUBSYSTEM=="accel", KERNEL=="accel*", ${mkUdevAcl "/dev/accel/$kernel"}''
      ]
      ++ optionals hasGpu [
        ''SUBSYSTEM=="drm", KERNEL=="card*", ${mkUdevAcl "/dev/dri/$kernel"}''
        ''SUBSYSTEM=="drm", KERNEL=="renderD*", ${mkUdevAcl "/dev/dri/$kernel"}''
      ]
      ++ optionals hasRdma [
        ''SUBSYSTEM=="infiniband", ${mkUdevAcl "/dev/infiniband/$kernel"}''
        ''SUBSYSTEM=="infiniband_verbs", ${mkUdevAcl "/dev/infiniband/$kernel"}''
        ''SUBSYSTEM=="infiniband_cm", ${mkUdevAcl "/dev/infiniband/$kernel"}''
      ]
      ++ optionals hasNvidiaGpu [
        ''KERNEL=="nvidia[0-9]*", ${mkUdevAcl "/dev/$kernel"}''
        ''KERNEL=="nvidiactl", ${mkUdevAcl "/dev/$kernel"}''
        ''KERNEL=="nvidia-uvm", ${mkUdevAcl "/dev/$kernel"}''
        ''KERNEL=="nvidia-uvm-tools", ${mkUdevAcl "/dev/$kernel"}''
      ]
    );
  };
}
