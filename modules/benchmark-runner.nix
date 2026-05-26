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
    hasPrefix
    mapAttrsToList
    mkIf
    mkOption
    optionals
    types
    unique
    ;

  common = import ./benchmark-common.nix { inherit lib pkgs; };
  cfg = config.benchmark;
  enabledRunnerEntries = filter (entry: entry.value.enable) (
    mapAttrsToList (name: value: { inherit name value; }) cfg.runners
  );
  enabledRunners = map (entry: entry.value) enabledRunnerEntries;
  hasEnabledRunner = enabledRunners != [ ];

  runnerFeatures =
    runner:
    runner.systemFeatures ++ map common.gpuFeature runner.gpus ++ map common.npuFeature runner.npus;

  modelsPath = toString cfg.modelsPath;

  gpus = concatMap (runner: runner.gpus) enabledRunners;
  npus = concatMap (runner: runner.npus) enabledRunners;

  hasGpu = gpus != [ ];
  hasAmdGpu = any (gpu: gpu.type == "amd") gpus;
  hasNvidiaGpu = any (gpu: gpu.type == "nvidia") gpus;
  hasNpu = npus != [ ];
  hasAmdNpu = any (npu: npu.type == "amd") npus;

  systemFeatures = unique (concatMap runnerFeatures enabledRunners);
  extraSandboxPaths = unique (
    [
      modelsPath
    ]
    ++ concatMap (runner: runner.extraSandboxPaths) enabledRunners
    ++ optionals (hasGpu || hasNpu) [
      "/dev/shm"
      "/proc"
      "/sys/dev"
      "/sys/devices"
    ]
    ++ optionals hasGpu [
      "/dev/dri"
      "/sys/class/drm"
    ]
    ++ optionals hasAmdGpu [
      "/dev/kfd"
      "/sys/bus/pci/devices"
      "/sys/class/hwmon"
      "/sys/class/kfd"
      "/sys/class/net"
      "/sys/class/scsi_host"
    ]
    ++ optionals hasAmdNpu [
      "/dev/accel"
      "/sys/class/accel"
    ]
    ++ optionals hasNvidiaGpu [
      "/dev/nvidia0"
      "/dev/nvidiactl"
      "/dev/nvidia-uvm"
      "/dev/nvidia-uvm-tools"
      "/dev/nvidia-caps"
      # autoAddDriverRunpath points binaries at /run/opengl-driver.
      # Mount the symlink target too, otherwise the sandbox sees a
      # dangling link instead of the NVIDIA userspace driver package.
      "/run/opengl-driver=${config.hardware.nvidia.package}"
    ]
  );

  relaxSandbox = any (runner: runner.relaxSandbox) enabledRunners;

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
    ]
    ++ optionals hasAmdGpu [
      "z /dev/kfd 0660 root nixbld -"
    ]
    ++ optionals hasGpu [
      "z /dev/dri/card* 0660 root nixbld -"
      "z /dev/dri/renderD* 0660 root nixbld -"
    ]
    ++ optionals hasAmdNpu [
      "z /dev/accel/accel* 0660 root nixbld -"
    ]
    ++ optionals hasNvidiaGpu [
      "z /dev/nvidia[0-9]* 0660 root nixbld -"
      "z /dev/nvidiactl 0660 root nixbld -"
      "z /dev/nvidia-uvm 0660 root nixbld -"
      "z /dev/nvidia-uvm-tools 0660 root nixbld -"
    ];

    services.udev.extraRules = concatStringsSep "\n" (
      optionals hasAmdGpu [
        ''KERNEL=="kfd", GROUP:="nixbld", MODE:="0660"''
      ]
      ++ optionals hasAmdNpu [
        ''SUBSYSTEM=="accel", KERNEL=="accel*", GROUP:="nixbld", MODE:="0660"''
      ]
      ++ optionals hasGpu [
        ''SUBSYSTEM=="drm", KERNEL=="card*", GROUP:="nixbld", MODE:="0660"''
        ''SUBSYSTEM=="drm", KERNEL=="renderD*", GROUP:="nixbld", MODE:="0660"''
      ]
      ++ optionals hasNvidiaGpu [
        ''KERNEL=="nvidia[0-9]*", GROUP:="nixbld", MODE:="0660"''
        ''KERNEL=="nvidiactl", GROUP:="nixbld", MODE:="0660"''
        ''KERNEL=="nvidia-uvm", GROUP:="nixbld", MODE:="0660"''
        ''KERNEL=="nvidia-uvm-tools", GROUP:="nixbld", MODE:="0660"''
      ]
    );
  };
}
