# NixOS module for systems that run benchmarks
{
  config,
  lib,
  pkgs,
  ...
}:
with lib; let
  cfg = config.services.benchmark-runner;
in {
  options.services.benchmark-runner = {
    enable = mkEnableOption "benchmark runner configuration";

    gpuTarget = mkOption {
      type = types.enum ["gfx110X" "gfx1151" "gfx120X"];
      default = "gfx1151";
      description = "GPU architecture target";
    };

    modelsPath = mkOption {
      type = types.path;
      default = "/models";
      description = "Path where benchmark models are stored";
    };

    enableSandboxRelaxation = mkOption {
      type = types.bool;
      default = true;
      description = "Whether to relax sandbox restrictions for GPU access";
    };

    ensureModels = mkOption {
      type = types.listOf (types.submodule {
        options = {
          repo = mkOption {
            type = types.str;
            description = "HuggingFace repository ID (e.g., 'meta-llama/Llama-2-7b')";
          };
          files = mkOption {
            type = types.listOf types.str;
            default = [];
            description = "Specific files to download (empty means all files)";
          };
          name = mkOption {
            type = types.str;
            description = "Local directory name for the model";
          };
        };
      });
      default = [];
      description = "Models to ensure are downloaded from HuggingFace";
    };
  };

  config = mkIf cfg.enable {
    # Add ROCm and GPU features to system
    nix.settings.system-features = [
      "benchmark"
      "big-parallel"
      "ca-derivations"
      "rocm"
      cfg.gpuTarget
      "kvm"
    ];

    # Allow GPU access in sandbox
    nix.settings.extra-sandbox-paths = [
      "/dev/kfd"
      "/dev/dri"
      "/sys/devices/virtual/kfd"
      "/sys/class/kfd"
      "/sys/class/drm"
      cfg.modelsPath
    ];

    # Optionally relax sandbox for benchmark builds
    nix.settings.sandbox = mkIf cfg.enableSandboxRelaxation "relaxed";

    # Limit concurrent builds for benchmarks
    nix.settings.max-jobs = mkDefault 1;

    # Create models directory with proper permissions
    systemd.tmpfiles.rules = [
      "d ${cfg.modelsPath} 0755 root root -"
      # Allow nixbld group to read models
      "A ${cfg.modelsPath} - - - - group:nixbld:r-x"
      "A ${cfg.modelsPath} - - - - default:group:nixbld:r-x"
    ];

    # allow nixbld group to access GPU devices
    services.udev.extraRules = ''
      # Allow nixbld group access to GPU devices
      KERNEL=="kfd", GROUP="nixbld", MODE="0660"
      SUBSYSTEM=="drm", KERNEL=="card*", GROUP="nixbld", MODE="0660"
      SUBSYSTEM=="drm", KERNEL=="renderD*", GROUP="nixbld", MODE="0660"
    '';

    # Systemd services to ensure models are downloaded
    systemd.services = let
      # Create individual download services for each file
      fileServices = flatten (map (model:
        map (file: {
          name = "ensure-model-${model.name}-${builtins.replaceStrings ["." "/"] ["-" "-"] file}";
          value = {
            description = "Download ${file} for model ${model.name}";
            after = ["network.target"];
            path = with pkgs; [
              (python3.withPackages (ps: with ps; [huggingface-hub hf-transfer]))
              coreutils
            ];
            environment = {
              HF_HUB_ENABLE_HF_TRANSFER = "1";
              HF_HOME = "${cfg.modelsPath}/.cache/huggingface";
            };
            serviceConfig = {
              Type = "oneshot";
              RemainAfterExit = true;
            };
            script = ''
              MODEL_DIR="${cfg.modelsPath}/${model.name}"
              FILE_PATH="$MODEL_DIR/${file}"

              if [ -f "$FILE_PATH" ]; then
                echo "File ${file} already exists"
                exit 0
              fi

              echo "Downloading ${file} from ${model.repo}"
              mkdir -p "$MODEL_DIR"

              huggingface-cli download "${model.repo}" "${file}" \
                --local-dir "$MODEL_DIR" \
                --local-dir-use-symlinks False

              chmod -R a+r "$MODEL_DIR"
              echo "File ${file} downloaded successfully"
            '';
          };
        }) model.files
      ) cfg.ensureModels);

      # Create orchestrator services to coordinate downloads
      orchestratorServices = map (model: {
        name = "ensure-model-${model.name}";
        value = {
          description = "Orchestrate downloads for model ${model.name}";
          wantedBy = ["multi-user.target"];
          after = ["network.target"];
          wants = map (
            file: "ensure-model-${model.name}-${builtins.replaceStrings ["." "/"] ["-" "-"] file}.service"
          ) model.files;
          serviceConfig = {
            Type = "oneshot";
            RemainAfterExit = true;
            ExecStart = "${pkgs.coreutils}/bin/true";
          };
        };
      }) cfg.ensureModels;
    in
      builtins.listToAttrs (fileServices ++ orchestratorServices);
  };
}
