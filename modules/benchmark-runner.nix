# NixOS module for systems that run benchmarks
{
  config,
  lib,
  pkgs,
  ...
}:
with lib; let
  cfg = config.services.benchmark-runner;
  isNvidia = hasPrefix "rtx" cfg.gpuTarget;
  isAmd = hasPrefix "gfx" cfg.gpuTarget;
in {
  options.services.benchmark-runner = {
    enable = mkEnableOption "benchmark runner configuration";

    gpuTarget = mkOption {
      type = types.enum ["gfx1010" "gfx1151" "rtx4090"];
      default = "gfx1151";
      description = "GPU architecture target";
    };

    cpuTarget = mkOption {
      type = types.str;
      description = "CPU identifier for benchmark dispatch (e.g., '9950x3d', 'aimax395')";
    };

    modelsPath = mkOption {
      type = types.path;
      default = "/models";
      description = "Path where benchmark models are stored";
    };

    enableSandboxRelaxation = mkOption {
      type = types.bool;
      default = false;
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
    # Add GPU and CPU features to system
    nix.settings.system-features = [cfg.gpuTarget cfg.cpuTarget];

    # Allow GPU access in sandbox
    nix.settings.extra-sandbox-paths =
      [
        "/dev/dri"
        "/dev/shm"
        "/sys/dev"
        "/sys/devices"
        "/proc"
        cfg.modelsPath
      ]
      ++ optionals isAmd [
        "/dev/kfd"
        "/sys/class/kfd"
        "/sys/class/drm"
      ]
      ++ optionals isNvidia [
        "/dev/nvidia0"
        "/dev/nvidiactl"
        "/dev/nvidia-uvm"
        "/dev/nvidia-uvm-tools"
        "/dev/nvidia-caps"
        # Bind-mount the NVIDIA driver package to /run/opengl-driver so
        # autoAddDriverRunpath RUNPATH entries resolve inside the sandbox.
        # A plain "/run/opengl-driver" only mounts the symlink; the target
        # store path is not a build input so it won't be available.
        "/run/opengl-driver=${config.hardware.nvidia.package}"
      ];

    # Optionally relax sandbox for benchmark builds
    nix.settings.sandbox = mkIf cfg.enableSandboxRelaxation "relaxed";

    # Limit concurrent builds for benchmarks
    # nix.settings.max-jobs = mkDefault 1;

    # Create models directory with proper permissions
    systemd.tmpfiles.rules = [
      "d ${cfg.modelsPath} 0755 root root -"
      # Allow nixbld group to read models
      "A ${cfg.modelsPath} - - - - group:nixbld:r-x"
      "A ${cfg.modelsPath} - - - - default:group:nixbld:r-x"
    ];

    # allow nixbld group to access GPU devices
    services.udev.extraRules = concatStringsSep "\n" (
      optionals isAmd [
        ''KERNEL=="kfd", GROUP="nixbld", MODE="0660"''
      ]
      ++ [
        ''SUBSYSTEM=="drm", KERNEL=="card*", GROUP="nixbld", MODE="0660"''
        ''SUBSYSTEM=="drm", KERNEL=="renderD*", GROUP="nixbld", MODE="0660"''
      ]
      ++ optionals isNvidia [
        ''KERNEL=="nvidia[0-9]*", GROUP="nixbld", MODE="0660"''
        ''KERNEL=="nvidiactl", GROUP="nixbld", MODE="0660"''
        ''KERNEL=="nvidia-uvm", GROUP="nixbld", MODE="0660"''
        ''KERNEL=="nvidia-uvm-tools", GROUP="nixbld", MODE="0660"''
      ]
    );

    # Systemd services to ensure models are downloaded
    systemd.services = let
      # Create individual download services for each file
      fileServices = flatten (map (
          model:
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
                    --local-dir "$MODEL_DIR"

                  chmod -R a+r "$MODEL_DIR"
                  echo "File ${file} downloaded successfully"
                '';
              };
            })
            model.files
        )
        cfg.ensureModels);

      # Create orchestrator services to coordinate downloads
      orchestratorServices =
        map (model: {
          name = "ensure-model-${model.name}";
          value = {
            description = "Orchestrate downloads for model ${model.name}";
            wantedBy = ["multi-user.target"];
            after = ["network.target"];
            wants =
              map (
                file: "ensure-model-${model.name}-${builtins.replaceStrings ["." "/"] ["-" "-"] file}.service"
              )
              model.files;
            serviceConfig = {
              Type = "oneshot";
              RemainAfterExit = true;
              ExecStart = "${pkgs.coreutils}/bin/true";
            };
          };
        })
        cfg.ensureModels;
    in
      builtins.listToAttrs (fileServices ++ orchestratorServices);
  };
}
