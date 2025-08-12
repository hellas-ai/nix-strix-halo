{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.ec-su-axb35;
in {
  options.services.ec-su-axb35 = {
    enable = lib.mkEnableOption "EC-SU_AXB35 embedded controller kernel module";

    monitor = {
      enable = lib.mkEnableOption "EC-SU_AXB35 monitor script";
    };
  };

  config = lib.mkIf cfg.enable {
    # Add kernel module to boot.extraModulePackages
    boot.extraModulePackages = [
      (pkgs.ec-su-axb35 {
        kernel = config.boot.kernelPackages.kernel;
      })
    ];

    # Load the module at boot
    boot.kernelModules = ["ec_su_axb35"];

    # Add monitor script if enabled
    environment.systemPackages = lib.mkIf cfg.monitor.enable [
      pkgs.ec-su-axb35-monitor
    ];
  };
}
