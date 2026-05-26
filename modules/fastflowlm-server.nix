# NixOS module for the FastFlowLM (FLM) OpenAI-compatible HTTP server
# running on the AMD Ryzen AI NPU.
#
# Models are not auto-downloaded: pull them once with
#
#   FLM_MODEL_PATH=/models/flm flm pull <tag>
#
# before enabling the service. The path is configurable via
# `services.fastflowlm-server.modelsPath`.
{
  config,
  lib,
  pkgs,
  ...
}:

with lib;

let
  cfg = config.services.fastflowlm-server;
in
{
  options.services.fastflowlm-server = {
    enable = mkEnableOption "FastFlowLM NPU inference server";

    package = mkOption {
      type = types.package;
      default = pkgs.fastflowlm;
      defaultText = literalExpression "pkgs.fastflowlm";
      description = "FastFlowLM package providing the `flm` binary.";
    };

    model = mkOption {
      type = types.str;
      default = "llama3.2:1b";
      example = "gemma4-it:e4b";
      description = ''
        Initial model tag (matches an `flm list` entry). The model
        directory must be pre-pulled to `''${modelsPath}/models/<NAME>-NPU2/`.
        FLM auto-switches to a different model if a request specifies one.
      '';
    };

    host = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = "Address to bind. Use 0.0.0.0 to listen on all interfaces.";
    };

    port = mkOption {
      type = types.port;
      default = 52625;
      description = "TCP port for the OpenAI-compatible HTTP API.";
    };

    pmode = mkOption {
      type = types.enum [
        "powersaver"
        "balanced"
        "performance"
        "turbo"
      ];
      default = "performance";
      description = "NPU power mode.";
    };

    sockets = mkOption {
      type = types.ints.positive;
      default = 10;
      description = "Maximum concurrent socket connections.";
    };

    qLen = mkOption {
      type = types.ints.positive;
      default = 10;
      description = "Maximum NPU queue length.";
    };

    cors = mkOption {
      type = types.bool;
      default = true;
      description = "Enable CORS for the HTTP API.";
    };

    ctxLen = mkOption {
      type = types.nullOr types.int;
      default = null;
      description = "Override the model's context length (tokens). null = model default.";
    };

    prefillChunkLen = mkOption {
      type = types.nullOr types.int;
      default = null;
      description = "Prefill chunk length (tokens). null = model default.";
    };

    modelsPath = mkOption {
      type = types.path;
      default = "/models/flm";
      description = ''
        FLM_MODEL_PATH for the service. Pre-pulled model directories
        live under `''${modelsPath}/models/<NAME>-NPU2/`.
      '';
    };

    extraArgs = mkOption {
      type = types.listOf types.str;
      default = [ ];
      example = [ "--verbose" ];
      description = "Extra args appended to the `flm serve` command line.";
    };

    user = mkOption {
      type = types.str;
      default = "fastflowlm";
      description = "Service user.";
    };

    group = mkOption {
      type = types.str;
      default = "fastflowlm";
      description = "Service group.";
    };

    udevMode = mkOption {
      type = types.str;
      default = "0666";
      example = "0660";
      description = ''
        Permissions applied to /dev/accel/accel* by the installed udev
        rule. The default 0666 is intentional: the NPU is a single-user
        device with no security-sensitive state, and 0666 means both
        this service and ad-hoc users (incl. nixbld builders for the
        bench-flm-* derivations) can open it without per-user group
        plumbing. Tighten to 0660 if you want group-scoped access.
      '';
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Whether to open the firewall for the configured port.";
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = elem "iommu=pt" config.boot.kernelParams;
        message = ''
          services.fastflowlm-server requires boot.kernelParams = [ "iommu=pt" ].
          The AMD XDNA NPU runtime does not work correctly with IOMMU disabled.
        '';
      }
    ];

    users.users.${cfg.user} = {
      isSystemUser = true;
      inherit (cfg) group;
      description = "FastFlowLM server user";
    };

    users.groups.${cfg.group} = { };

    # The accel subsystem is new (Linux 6.14+). Default udev rules don't
    # touch it, so a non-root service can't open /dev/accel/accel0
    # without an explicit rule.
    services.udev.extraRules = ''
      SUBSYSTEM=="accel", KERNEL=="accel*", GROUP="${cfg.group}", MODE="${cfg.udevMode}"
    '';

    systemd.services.fastflowlm-server = {
      description = "FastFlowLM NPU inference server";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        FLM_MODEL_PATH = toString cfg.modelsPath;
        FLM_DISABLE_UPDATE_CHECK = "1";
      };

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;

        ExecStart =
          let
            args = [
              (lib.getExe cfg.package)
              "serve"
              "--host"
              cfg.host
              "--port"
              (toString cfg.port)
              "--pmode"
              cfg.pmode
              "--socket"
              (toString cfg.sockets)
              "--q-len"
              (toString cfg.qLen)
              "--cors"
              (if cfg.cors then "1" else "0")
            ]
            ++ optionals (cfg.ctxLen != null) [
              "--ctx-len"
              (toString cfg.ctxLen)
            ]
            ++ optionals (cfg.prefillChunkLen != null) [
              "--prefill-chunk-len"
              (toString cfg.prefillChunkLen)
            ]
            ++ cfg.extraArgs
            ++ [ cfg.model ];
          in
          escapeShellArgs args;

        Restart = "on-failure";
        RestartSec = 5;

        # FLM mlocks NPU input/output buffers; the default 8 MB cap
        # forces fallback to pageable memory and tanks throughput.
        LimitMEMLOCK = "infinity";

        # Hardening: flm only needs the NPU device and the models tree.
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ProtectHostname = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectKernelLogs = true;
        ProtectControlGroups = true;
        ProtectClock = true;
        RestrictNamespaces = true;
        RestrictRealtime = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = false; # flm JITs / dlopens its NPU .so blobs
        NoNewPrivileges = true;
        SystemCallArchitectures = "native";
        # /dev/accel is the AMD NPU; mount it read-write into the service
        # namespace. DRM accel devices live in their own subsystem (not
        # /dev/dri).
        DeviceAllow = [ "/dev/accel/accel0 rw" ];
        DevicePolicy = "closed";
        # Read-only model tree; writable HOME-like state if flm needs to
        # cache anything between requests.
        ReadOnlyPaths = [ (toString cfg.modelsPath) ];
        StateDirectory = "fastflowlm";
        StateDirectoryMode = "0750";
        Environment = "HOME=/var/lib/fastflowlm";
      };
    };

    networking.firewall.allowedTCPPorts = mkIf cfg.openFirewall [ cfg.port ];

    environment.systemPackages = [ cfg.package ];
  };
}
