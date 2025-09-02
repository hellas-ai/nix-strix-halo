{
  config,
  lib,
  pkgs,
  ...
}: {
  boot = {
    kernelParams = [
      "amd_iommu=off"
    ];
    tmp.useTmpfs = true;
    extraModprobeConfig = ''
      # From: https://strixhalo-homelab.d7.wtf/AI/AI-Capabilities-Overview#memory-limits
      ## This specifies GTT by # of 4KB pages:
      ##   31457280 * 4KB / 1024 / 1024 = 120 GiB
      ## We leave a buffer of 8GiB on the limit to try to keep your system from crashing if it runs out of memory
      options ttm pages_limit=31457280

      ## Optionally we can pre-allocate any amount of memory. This pool is never accessible to the system.
      ## You might want to do this to reduce GTT fragmentation, and it might have a perf improvement.
      ## If you are using your system exclusively to run AI models, just max this out to match your pages_limit.
      ## This example specifies 60GiB pre-allocated.
      # options ttm page_pool_size=15728640
    '';
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
