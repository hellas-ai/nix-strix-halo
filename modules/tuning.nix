{
  config,
  lib,
  pkgs,
  ...
}: {
  boot = {
    kernelParams = [
      "amd_iommu=off"
      "amdgpu.gttsize=131072"
      "ttm.pages_limit=33554432"
    ];
    tmp.useTmpfs = true;
    kernelPackages = pkgs.linuxPackages_cachyos-rc.cachyOverride {
      mArch = "ZEN4";
    };
  };

  services.tuned = {
    enable = true;
    profiles = {
      strix-halo = {
        main = {
          include = "accelerator-performance";
        };
      };
    };
  };

  systemd.services.tuned-set-profile = {
    description = "Set TuneD profile";
    after = ["tuned.service"];
    wantedBy = ["multi-user.target"];
    serviceConfig = {
      Type = "oneshot";
      ExecStart = "${pkgs.tuned}/bin/tuned-adm profile accelerator-performance";
    };
  };
}
