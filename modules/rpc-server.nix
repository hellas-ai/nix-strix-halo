{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.llamacpp-rpc-server;
in {
  options.services.llamacpp-rpc-server = {
    enable = mkEnableOption "llama.cpp RPC server";

    package = mkOption {
      type = types.package;
      default = pkgs.llamacpp-rocm.gfx1151;
      defaultText = literalExpression "pkgs.llamacpp-rocm.gfx1151";
      description = "The llama.cpp package to use (must include rpc-server binary)";
    };

    threads = mkOption {
      type = types.int;
      default = 64;
      description = "Number of threads for the CPU backend";
    };

    device = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "0";
      description = "Device to use (e.g., GPU device ID)";
    };

    host = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = "Host to bind to";
    };

    port = mkOption {
      type = types.port;
      default = 50052;
      description = "Port to bind to";
    };

    memory = mkOption {
      type = types.nullOr types.int;
      default = null;
      example = 8192;
      description = "Backend memory size in MB";
    };

    enableCache = mkOption {
      type = types.bool;
      default = false;
      description = "Enable local file cache";
    };

    cacheDirectory = mkOption {
      type = types.path;
      default = "/var/cache/llamacpp-rpc";
      description = "Directory for the file cache (when enableCache is true)";
    };

    extraArgs = mkOption {
      type = types.listOf types.str;
      default = [];
      example = [ "--verbose" ];
      description = "Extra arguments to pass to rpc-server";
    };

    user = mkOption {
      type = types.str;
      default = "llamacpp-rpc";
      description = "User under which the rpc-server runs";
    };

    group = mkOption {
      type = types.str;
      default = "llamacpp-rpc";
      description = "Group under which the rpc-server runs";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Whether to open the firewall for the RPC port";
    };
  };

  config = mkIf cfg.enable {
    # Create user and group
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      description = "llama.cpp RPC server user";
      home = cfg.cacheDirectory;
      createHome = cfg.enableCache;
    };

    users.groups.${cfg.group} = {};

    # Create cache directory if needed
    systemd.tmpfiles.rules = mkIf cfg.enableCache [
      "d ${cfg.cacheDirectory} 0755 ${cfg.user} ${cfg.group} - -"
    ];

    # Systemd service
    systemd.services.llamacpp-rpc-server = {
      description = "llama.cpp RPC server";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = let
          args = [
            "${cfg.package}/bin/rpc-server"
            "--threads" (toString cfg.threads)
            "--host" cfg.host
            "--port" (toString cfg.port)
          ]
          ++ optionals (cfg.device != null) [ "--device" cfg.device ]
          ++ optionals (cfg.memory != null) [ "--mem" (toString cfg.memory) ]
          ++ optionals cfg.enableCache [ "--cache" ]
          ++ cfg.extraArgs;
        in escapeShellArgs args;
        
        Restart = "on-failure";
        RestartSec = 5;
        
        # Security hardening
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        NoNewPrivileges = true;
        PrivateDevices = false; # Need access to GPU devices
        ReadWritePaths = mkIf cfg.enableCache [ cfg.cacheDirectory ];
        
        # Environment for ROCm
        Environment = [
          "HSA_OVERRIDE_GFX_VERSION=11.5.1"
          "ROCM_PATH=${cfg.package}/rocm"
        ];
      };
    };

    # Open firewall if requested
    networking.firewall.allowedTCPPorts = mkIf cfg.openFirewall [ cfg.port ];

    # Add package to system packages for debugging
    environment.systemPackages = [ cfg.package ];
  };
}