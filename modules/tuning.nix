{ pkgs, ... }:
{
  boot = {
    kernelParams = [
      # GTT: 80GB (leaves ~30GB system RAM for OS/apps)
      "ttm.pages_limit=20971520"
    ];
    tmp.useTmpfs = true;
  };

  hardware.firmware = [ pkgs.strix-halo-mes-firmware ];

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
    after = [ "tuned.service" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "oneshot";
      ExecStart = "${pkgs.tuned}/bin/tuned-adm profile accelerator-performance";
    };
  };
}
