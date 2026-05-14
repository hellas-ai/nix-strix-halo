{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.hardware.strixHalo;
  pagesPerGiB = 262144; # 1 GiB / 4 KiB
  mesFirmwareRev = "3d5c8135206cef364e7d353711b3e7358a90d152";
  fetchMesFirmware =
    file: hash:
    pkgs.fetchurl {
      url = "https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/plain/amdgpu/${file}?id=${mesFirmwareRev}";
      inherit hash;
    };
  mesFirmwarePackage = pkgs.runCommand "strix-halo-mes-firmware-0x80" { } ''
    install -Dm644 ${fetchMesFirmware "gc_11_5_0_mes_2.bin" "sha256-XdxUTOMcScfvDxQQyo7oi3KmrARRS9Ec+v/gBJQ5ce0="} \
      "$out/lib/firmware/amdgpu/gc_11_5_0_mes_2.bin"
    install -Dm644 ${fetchMesFirmware "gc_11_5_1_mes_2.bin" "sha256-jgeDLBjYe3ZD/CIWM9hBpfo7rg59Kq0fKyEuGQ+WOgo="} \
      "$out/lib/firmware/amdgpu/gc_11_5_1_mes_2.bin"
  '';
in
{
  options.hardware.strixHalo = {
    enable = lib.mkEnableOption "AMD Ryzen AI Max+ 395 (Strix Halo / gfx1151) tuning";

    gpuMemoryGiB = lib.mkOption {
      type = lib.types.ints.positive;
      default = 110;
      example = 62;
      description = ''
        Upper bound on GPU-addressable GTT memory, in GiB. Sets
        `ttm.pages_limit` on the kernel cmdline.

        Default 110: tuned for dedicated-inference Strix Halo boxes with
        128 GiB total, leaving ~18 GiB for the OS. Required to fit large
        quantised models (e.g. 92 GiB DS4 Flash Q2_K) in
        device-allocated HIP tensors.
      '';
    };

    iommuPassthrough = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable IOMMU pass-through. Required for zero-overhead GTT access from the GPU.";
    };

    mesFirmware0x80 = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Install uncompressed Strix Halo MES 0x80 firmware blobs ahead of
        distro linux-firmware. MES 0x83 regresses ROCm queue creation on
        gfx1151 with gfxhub CPF page faults.
      '';
    };

    disableCwsr = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Disable amdgpu compute wave save/restore. Leave off by default; enable
        only as a workaround for ROCm compute preemption instability.
      '';
    };

    tuned = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Enable the tuned daemon with an inference-tuned profile.";
      };
      profile = lib.mkOption {
        type = lib.types.str;
        default = "accelerator-performance";
        example = "throughput-performance";
        description = "Active tuned profile.";
      };
    };

    hugePages = lib.mkOption {
      type = lib.types.enum [
        "never"
        "madvise"
        "always"
      ];
      default = "always";
      description = "Transparent huge pages mode. `always` cuts IOMMU TLB misses on huge model mmaps ~512x.";
    };

    vmMaxMapCount = lib.mkOption {
      type = lib.types.ints.positive;
      default = 1048576;
      description = "vm.max_map_count. Default (65530) is too low for huge model mmaps.";
    };
  };

  config = lib.mkIf cfg.enable {
    boot = {
      kernelParams = [
        "ttm.pages_limit=${toString (cfg.gpuMemoryGiB * pagesPerGiB)}"
        "transparent_hugepage=${cfg.hugePages}"
      ]
      ++ lib.optional cfg.disableCwsr "amdgpu.cwsr_enable=0"
      ++ lib.optionals cfg.iommuPassthrough [
        "amd_iommu=pt"
        "iommu=pt"
        "iommu.passthrough=1"
      ];

      tmp.useTmpfs = true;

      kernel.sysctl = {
        "vm.max_map_count" = cfg.vmMaxMapCount;
        "kernel.numa_balancing" = 0;
        "vm.swappiness" = 1;
      };
    };

    hardware.firmware = lib.optional cfg.mesFirmware0x80 mesFirmwarePackage;

    # `hipHostRegister` pins large pages; default RLIMIT_MEMLOCK is 64 KiB.
    security.pam.loginLimits = [
      {
        domain = "*";
        type = "soft";
        item = "memlock";
        value = "unlimited";
      }
      {
        domain = "*";
        type = "hard";
        item = "memlock";
        value = "unlimited";
      }
    ];

    services.tuned = lib.mkIf cfg.tuned.enable {
      enable = true;
      profiles.strix-halo.main.include = cfg.tuned.profile;
    };
    services.power-profiles-daemon.enable = lib.mkIf cfg.tuned.enable false;

    systemd.services.tuned-set-profile = lib.mkIf cfg.tuned.enable {
      description = "Set TuneD profile";
      after = [ "tuned.service" ];
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        Type = "oneshot";
        ExecStart = "${pkgs.tuned}/bin/tuned-adm profile ${cfg.tuned.profile}";
      };
    };

    assertions = [
      {
        assertion = cfg.gpuMemoryGiB <= 120;
        message = "hardware.strixHalo.gpuMemoryGiB=${toString cfg.gpuMemoryGiB} leaves under 8 GiB for the OS on a 128 GiB box.";
      }
    ];
  };
}
