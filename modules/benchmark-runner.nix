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
    concatMapStringsSep
    concatStringsSep
    filter
    getBin
    hasPrefix
    mapAttrsToList
    mkIf
    mkOption
    optionals
    optionalString
    stringAfter
    types
    unique
    ;

  common = import ./benchmark-common.nix { inherit lib pkgs; };
  cfg = config.benchmark;

  setfacl = "${getBin pkgs.acl}/bin/setfacl";

  # ACL rw entries for cfg.extraUsers. Devices keep group=nixbld so Nix
  # build sandboxes (whose user namespace only maps the primary nixbld
  # group) retain access; ACLs grant interactive users a separate rw
  # entry without competing for the device's group ownership.
  mkUdevAclSuffix =
    devicePath:
    optionalString (cfg.extraUsers != [ ]) (
      ", "
      + concatMapStringsSep ", " (
        user: ''RUN+="${setfacl} -m u:${user}:rw ${devicePath}"''
      ) cfg.extraUsers
    );

  mkShellAclLines =
    pathVar:
    concatMapStringsSep "\n        " (
      user: ''${setfacl} -m u:${user}:rw -- "${pathVar}" 2>/dev/null || true''
    ) cfg.extraUsers;
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

  systemFeatures = unique (concatMap runnerFeatures enabledRunners);
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
  devicePermissionGlobs =
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
  applyDevicePermissions = concatStringsSep "\n" (
    map (glob: ''
      for path in ${glob}; do
        [ -e "$path" ] || continue
        ${pkgs.coreutils}/bin/chgrp nixbld -- "$path" 2>/dev/null || true
        ${pkgs.coreutils}/bin/chmod 0660 -- "$path" 2>/dev/null || true
        ${mkShellAclLines "$path"}
      done
    '') devicePermissionGlobs
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

    extraUsers = mkOption {
      type = types.listOf types.str;
      default = [ ];
      example = [ "grw" ];
      description = ''
        Users granted rw access to benchmark accelerator devices via POSIX
        ACLs, in addition to the `nixbld` group. Lets a host's interactive
        users share `/dev/kfd`, `/dev/dri/*`, `/dev/accel/*`,
        `/dev/infiniband/*`, and `/dev/nvidia*` with Nix build sandboxes
        on the same host without changing the devices' group ownership.

        Sandbox builds must use group=nixbld because Nix runs each build
        in a user namespace where only the primary group is mapped;
        supplementary group memberships set on nixbld users on the host
        are dropped inside the sandbox. ACLs work because the kernel
        evaluates them against the real host UID regardless of the
        namespace, so an entry like `u:grw:rw` reaches grw directly
        without touching the device group.
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
    ++ optionals hasRdma [
      "z /dev/infiniband/* 0660 root nixbld -"
    ]
    ++ optionals hasNvidiaGpu [
      "z /dev/nvidia[0-9]* 0660 root nixbld -"
      "z /dev/nvidiactl 0660 root nixbld -"
      "z /dev/nvidia-uvm 0660 root nixbld -"
      "z /dev/nvidia-uvm-tools 0660 root nixbld -"
    ];

    system.activationScripts.benchmark-runner-device-permissions = mkIf (devicePermissionGlobs != [ ]) (
      stringAfter [ "users" ] applyDevicePermissions
    );

    systemd.services = mkIf (devicePermissionGlobs != [ ]) {
      benchmark-runner-device-permissions = {
        description = "Apply benchmark runner device permissions";
        wantedBy = [ "multi-user.target" ];
        wants = [
          "systemd-udev-trigger.service"
          "systemd-udev-settle.service"
        ];
        after = [
          "systemd-udev-trigger.service"
          "systemd-udev-settle.service"
          "systemd-tmpfiles-setup-dev.service"
        ];
        serviceConfig = {
          Type = "oneshot";
          RemainAfterExit = true;
        };
        script = applyDevicePermissions;
      };
    };

    services.udev.extraRules = concatStringsSep "\n" (
      optionals hasAmdGpu [
        ''KERNEL=="kfd", GROUP:="nixbld", MODE:="0660"${mkUdevAclSuffix "/dev/$kernel"}''
      ]
      ++ optionals hasAmdNpu [
        ''SUBSYSTEM=="accel", KERNEL=="accel*", GROUP:="nixbld", MODE:="0660"${mkUdevAclSuffix "/dev/accel/$kernel"}''
      ]
      ++ optionals hasGpu [
        ''SUBSYSTEM=="drm", KERNEL=="card*", GROUP:="nixbld", MODE:="0660"${mkUdevAclSuffix "/dev/dri/$kernel"}''
        ''SUBSYSTEM=="drm", KERNEL=="renderD*", GROUP:="nixbld", MODE:="0660"${mkUdevAclSuffix "/dev/dri/$kernel"}''
      ]
      ++ optionals hasRdma [
        ''SUBSYSTEM=="infiniband", GROUP:="nixbld", MODE:="0660"${mkUdevAclSuffix "/dev/infiniband/$kernel"}''
        ''SUBSYSTEM=="infiniband_verbs", GROUP:="nixbld", MODE:="0660"${mkUdevAclSuffix "/dev/infiniband/$kernel"}''
        ''SUBSYSTEM=="infiniband_cm", GROUP:="nixbld", MODE:="0660"${mkUdevAclSuffix "/dev/infiniband/$kernel"}''
      ]
      ++ optionals hasNvidiaGpu [
        ''KERNEL=="nvidia[0-9]*", GROUP:="nixbld", MODE:="0660"${mkUdevAclSuffix "/dev/$kernel"}''
        ''KERNEL=="nvidiactl", GROUP:="nixbld", MODE:="0660"${mkUdevAclSuffix "/dev/$kernel"}''
        ''KERNEL=="nvidia-uvm", GROUP:="nixbld", MODE:="0660"${mkUdevAclSuffix "/dev/$kernel"}''
        ''KERNEL=="nvidia-uvm-tools", GROUP:="nixbld", MODE:="0660"${mkUdevAclSuffix "/dev/$kernel"}''
      ]
    );
  };
}
