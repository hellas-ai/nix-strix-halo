{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.services.vllm-metal;

  arguments = [
    "serve"
    cfg.model
    "--served-model-name"
    cfg.servedModelName
    "--host"
    cfg.host
    "--port"
    (toString cfg.port)
    "--max-model-len"
    (toString cfg.maxModelLen)
    "--max-num-seqs"
    (toString cfg.maxNumSeqs)
    "--gpu-memory-utilization"
    (toString cfg.gpuMemoryUtilization)
  ]
  ++ lib.optional cfg.enablePrefixCaching "--enable-prefix-caching"
  ++ lib.optionals (cfg.reasoningParser != null) [
    "--reasoning-parser"
    cfg.reasoningParser
  ]
  ++ lib.optional cfg.enableAutoToolChoice "--enable-auto-tool-choice"
  ++ lib.optionals (cfg.toolCallParser != null) [
    "--tool-call-parser"
    cfg.toolCallParser
  ]
  ++ lib.optional cfg.trustRemoteCode "--trust-remote-code"
  ++ lib.optionals (cfg.revision != null) [
    "--revision"
    cfg.revision
  ]
  ++ lib.optionals (cfg.speculativeConfig != null) [
    "--speculative-config"
    (builtins.toJSON cfg.speculativeConfig)
  ]
  ++ cfg.extraArgs;

  runScript = pkgs.writeShellScript "vllm-metal-run" ''
    exec ${cfg.package}/bin/vllm ${lib.escapeShellArgs arguments}
  '';
in
{
  options.services.vllm-metal = {
    enable = lib.mkEnableOption "the vLLM OpenAI-compatible server with the Metal plugin";

    package = lib.mkOption {
      type = lib.types.package;
      description = "vLLM environment containing the vLLM-Metal plugin.";
    };

    model = lib.mkOption {
      type = lib.types.nonEmptyStr;
      example = "Qwen/Qwen3.8-27B";
      description = ''
        Local model directory or Hugging Face model identifier. This is a
        string rather than a Nix path so local model weights are not copied
        into the Nix store.
      '';
    };

    revision = lib.mkOption {
      type = lib.types.nullOr lib.types.nonEmptyStr;
      default = null;
      example = "706cebd746c4b6f2b1d1f892630867acfdfd3df8";
      description = "Model repository revision; null selects the repository default.";
    };

    servedModelName = lib.mkOption {
      type = lib.types.nonEmptyStr;
      default = "default";
      description = "Model name exposed by the OpenAI-compatible API.";
    };

    host = lib.mkOption {
      type = lib.types.nonEmptyStr;
      default = "127.0.0.1";
      description = "Address on which the HTTP server listens.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 11500;
      description = "Port on which the HTTP server listens.";
    };

    maxModelLen = lib.mkOption {
      type = lib.types.ints.positive;
      default = 262144;
      description = "Maximum combined prompt and completion length in tokens.";
    };

    maxNumSeqs = lib.mkOption {
      type = lib.types.ints.positive;
      default = 1;
      description = "Maximum number of sequences scheduled concurrently.";
    };

    gpuMemoryUtilization = lib.mkOption {
      type = lib.types.numbers.between 0.0 1.0;
      default = 0.55;
      description = "Fraction of unified memory made available to vLLM-Metal.";
    };

    enablePrefixCaching = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable automatic prefix caching.";
    };

    reasoningParser = lib.mkOption {
      type = lib.types.nullOr lib.types.nonEmptyStr;
      default = null;
      example = "qwen3";
      description = "Reasoning parser name, or null to leave reasoning unparsed.";
    };

    enableAutoToolChoice = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Allow the model to choose tools automatically.";
    };

    toolCallParser = lib.mkOption {
      type = lib.types.nullOr lib.types.nonEmptyStr;
      default = null;
      example = "qwen3_coder";
      description = "Tool-call parser name, or null to disable parsed tool calls.";
    };

    trustRemoteCode = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Allow model repositories to execute custom Python code.";
    };

    speculativeConfig = lib.mkOption {
      type = lib.types.nullOr (lib.types.attrsOf lib.types.anything);
      default = null;
      example = {
        method = "mtp";
        num_speculative_tokens = 1;
      };
      description = ''
        Configuration passed as JSON to --speculative-config. Speculation is
        disabled by default because its speed and exactness are model-specific.
      '';
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "--disable-log-requests" ];
      description = "Additional arguments appended to the vLLM invocation.";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      example = {
        HOME = "/Users/alice";
        HF_HOME = "/Users/alice/.cache/huggingface";
      };
      description = "Environment variables for the launchd service.";
    };

    user = lib.mkOption {
      type = lib.types.nonEmptyStr;
      example = "alice";
      description = "Unprivileged account under which the system daemon runs.";
    };

    workingDirectory = lib.mkOption {
      type = lib.types.nullOr lib.types.nonEmptyStr;
      default = null;
      example = "/Users/alice";
      description = "Working directory for the daemon, or null for launchd's default.";
    };

  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = pkgs.stdenv.hostPlatform.isDarwin && pkgs.stdenv.hostPlatform.isAarch64;
        message = "services.vllm-metal is supported only on Apple Silicon Darwin";
      }
      {
        assertion = cfg.gpuMemoryUtilization > 0.0;
        message = "services.vllm-metal.gpuMemoryUtilization must be greater than zero";
      }
      {
        assertion = !cfg.enableAutoToolChoice || cfg.toolCallParser != null;
        message = "services.vllm-metal.enableAutoToolChoice requires toolCallParser";
      }
      {
        assertion =
          cfg.speculativeConfig == null
          || (cfg.speculativeConfig.method or null) != "mtp"
          || cfg.maxNumSeqs == 1;
        message = "services.vllm-metal MTP speculation requires maxNumSeqs = 1";
      }
    ];

    launchd.daemons.vllm-metal = {
      command = "${runScript}";
      inherit (cfg) environment;
      serviceConfig = {
        KeepAlive = {
          SuccessfulExit = false;
        };
        RunAtLoad = true;
        ExitTimeOut = 60;
        ThrottleInterval = 60;
        UserName = cfg.user;
      }
      // lib.optionalAttrs (cfg.workingDirectory != null) {
        WorkingDirectory = cfg.workingDirectory;
      };
    };
  };
}
