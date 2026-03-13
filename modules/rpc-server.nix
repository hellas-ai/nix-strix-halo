{
  config,
  lib,
  pkgs,
  ...
}:
with lib; let
  cfg = config.services.llamacpp-rpc-servers;

  instanceModule = {
    name,
    config,
    ...
  }: {
    options = {
      enable = mkEnableOption "this llama.cpp RPC server instance";

      package = mkOption {
        type = types.package;
        default = pkgs.llamacpp-rocm;
        defaultText = literalExpression "pkgs.llamacpp-rocm";
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

      enableCache = mkOption {
        type = types.bool;
        default = true;
        description = "Enable local file cache";
      };

      cacheDirectory = mkOption {
        type = types.path;
        default = "/var/cache/llamacpp-rpc/${name}";
        description = "Directory for the file cache (when enableCache is true)";
      };

      extraArgs = mkOption {
        type = types.listOf types.str;
        default = [];
        example = ["--verbose"];
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
  };

  enabledInstances = filterAttrs (_: inst: inst.enable) cfg;
in {
  options.services.llamacpp-rpc-servers = mkOption {
    type = types.attrsOf (types.submodule instanceModule);
    default = {};
    description = "Named llama.cpp RPC server instances";
    example = literalExpression ''
      {
        gpu = {
          enable = true;
          device = "0";
          port = 50052;
        };
        cpu = {
          enable = true;
          port = 50053;
          threads = 64;
        };
      }
    '';
  };

  config = mkIf (enabledInstances != {}) {
    # Collect all unique user/group pairs
    users.users = mapAttrs' (_: inst:
      nameValuePair inst.user {
        isSystemUser = true;
        group = inst.group;
        extraGroups = ["render" "video"];
        description = "llama.cpp RPC server user";
      })
    enabledInstances;

    users.groups = mapAttrs' (_: inst:
      nameValuePair inst.group {})
    enabledInstances;

    # Create cache directories
    systemd.tmpfiles.rules = concatLists (mapAttrsToList (_: inst:
      optional inst.enableCache
      "d ${inst.cacheDirectory} 0755 ${inst.user} ${inst.group} - -")
    enabledInstances);

    # Systemd services — one per instance
    systemd.services = mapAttrs' (name: inst:
      nameValuePair "llamacpp-rpc-server-${name}" {
        description = "llama.cpp RPC server (${name})";
        after = ["network.target"];
        wantedBy = ["multi-user.target"];

        serviceConfig = {
          Type = "simple";
          User = inst.user;
          Group = inst.group;
          ExecStart = let
            args =
              [
                "${inst.package}/bin/llama-rpc-server"
                "--threads"
                (toString inst.threads)
                "--host"
                inst.host
                "--port"
                (toString inst.port)
              ]
              ++ optionals (inst.device != null) ["--device" inst.device]
              ++ optionals inst.enableCache ["--cache"]
              ++ inst.extraArgs;
          in
            escapeShellArgs args;

          WorkingDirectory = mkIf inst.enableCache inst.cacheDirectory;
          Environment = mkIf inst.enableCache "HOME=${inst.cacheDirectory}";

          Restart = "on-failure";
          RestartSec = 5;

          # Security hardening
          PrivateTmp = true;
          ProtectHome = true;
          NoNewPrivileges = true;
          PrivateDevices = false; # Need access to GPU devices
        };
      })
    enabledInstances;

    # Open firewall for instances that request it
    networking.firewall.allowedTCPPorts =
      mapAttrsToList (_: inst: inst.port)
      (filterAttrs (_: inst: inst.openFirewall) enabledInstances);

    # Add packages to system packages for debugging
    environment.systemPackages =
      unique (mapAttrsToList (_: inst: inst.package) enabledInstances);
  };
}
