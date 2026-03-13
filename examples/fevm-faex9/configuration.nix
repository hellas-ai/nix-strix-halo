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
    firewall = {
      enable = true;
      allowedTCPPorts = [22 8080 50052];
    };
  };

  # Time zone and locale
  time.timeZone = "UTC";
  i18n.defaultLocale = "en_US.UTF-8";

  # Enable SSH
  # services.openssh = {
  #   enable = true;
  #   settings = {
  #     PermitRootLogin = "no";
  #     PasswordAuthentication = false;
  #   };
  # };

  # Llama.cpp RPC server instances (from our module)
  # services.llamacpp-rpc-servers = {
  #   gpu = {
  #     enable = true;
  #     device = "0";
  #     host = "0.0.0.0";
  #     port = 50052;
  #     memory = 32768;
  #   };
  #   cpu = {
  #     enable = true;
  #     host = "0.0.0.0";
  #     port = 50053;
  #     threads = 64;
  #     memory = 65536;
  #   };
  # };

  # # Benchmark runner service (from our module)
  services.benchmark-runner = {
    enable = true;
    gpuTarget = "gfx1151";
    modelsPath = "/models";
  };
}
