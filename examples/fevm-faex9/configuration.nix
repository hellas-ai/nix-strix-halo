# NixOS configuration example for FEVM-FAEX9 system with AMD GPU
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

  # Time zone and locale
  time.timeZone = "UTC";
  i18n.defaultLocale = "en_US.UTF-8";

  # Networking
  networking = {
    hostName = "fevm-faex9";
    useDHCP = lib.mkDefault true;
    useNetworkd = lib.mkDefault true;
    firewall = {
      enable = true;
      allowedTCPPorts = [22 8080 50052];
    };
  };

  # Enable SSH
  # services.openssh = {
  #   enable = true;
  #   settings = {
  #     PermitRootLogin = "no";
  #     PasswordAuthentication = false;
  #   };
  # };

  # Llama.cpp RPC server service (from our module)
  # services.llamacpp-rpc-server = {
  #   enable = true;
  #   package = pkgs.llamacpp-rocm.gfx1151-rocwmma;
  #   host = "0.0.0.0";
  #   port = 50052;
  # };
}
