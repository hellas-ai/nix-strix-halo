# NixOS configuration example for bootable usb stick
{
  config,
  pkgs,
  lib,
  modulesPath,
  inputs,
  ...
}: {
  imports = [
    # Import our custom modules
    inputs.self.nixosModules.default
    inputs.self.nixosModules.rpc-server
    inputs.self.nixosModules.benchmark-runner
    inputs.self.nixosModules.ec-su-axb35
  ];

  system.stateVersion = "25.03";

  environment.systemPackages = with pkgs; [
    llamacpp-rocm.gfx1151-rocwmma
  ];

  # open-webui is unfree?!
  nixpkgs.config.allowUnfreePredicate = pkg:
    builtins.elem (lib.getName pkg) [
      "open-webui"
    ];

  services.open-webui = {
    enable = true;
    environment = {
      ANONYMIZED_TELEMETRY = "False";
      DO_NOT_TRACK = "True";
      SCARF_NO_ANALYTICS = "True";
    };
  };

  # Time zone and locale
  time.timeZone = "UTC";
  i18n.defaultLocale = "en_US.UTF-8";

  # Help text for the live environment
  services.getty.helpLine = ''
    Welcome to NixOS LlamaCPP ROCm Live USB
    Log in as "nixos" with password "nixos"
  '';
}
