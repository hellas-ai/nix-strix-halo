{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.ec-su-axb35;
  ec-su-axb35-src = config.lib.inputs.ec-su-axb35 or (throw "ec-su-axb35 input not found");

  mkEcSuAxb35 = import ../pkgs/ec-su-axb35.nix {
    inherit pkgs;
    ec-su-axb35 = ec-su-axb35-src;
  };
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
      (mkEcSuAxb35.ec-su-axb35 {
        kernel = config.boot.kernelPackages.kernel;
      })
    ];

    # Load the module at boot
    boot.kernelModules = ["ec_su_axb35"];

    # Add monitor script if enabled
    environment.systemPackages = lib.mkIf cfg.monitor.enable [
      mkEcSuAxb35.ec-su-axb35-monitor
    ];
  };
}
