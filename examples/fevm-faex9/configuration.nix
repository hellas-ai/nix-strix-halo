# NixOS configuration example for FEVM-FAEX9 system with AMD GPU
{
  lib,
  inputs,
  ...
}:
{
  imports = [
    inputs.self.nixosModules.default
    inputs.self.nixosModules.rpc-server
    inputs.self.nixosModules.benchmark-runner
    inputs.self.nixosModules.ec-su-axb35
    inputs.self.nixosModules.disko-raid0
  ];

  # This value determines the NixOS release from which the default
  # settings for stateful data, like file locations and database versions
  # on your system were taken.
  system.stateVersion = "25.03";

  # Boot configuration
  boot = {
    loader = {
      systemd-boot.enable = true;
      efi.canTouchEfiVariables = true;
      timeout = 3;
    };
  };

  # High-performance profile
  powerManagement.cpuFreqGovernor = lib.mkDefault "performance";

  # Networking
  networking = {
    hostName = "fevm-faex9";
    useDHCP = true;
    useNetworkd = true;
    firewall.enable = true;
  };

  # Time zone and locale
  time.timeZone = "UTC";
  i18n.defaultLocale = "en_US.UTF-8";

  services.benchmark-runner = {
    enable = true;
    gpuTarget = "gfx1151";
    cpuTarget = "aimax395";
    modelsPath = "/models";
  };
}
