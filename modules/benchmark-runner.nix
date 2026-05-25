# NixOS adapter for hosts that run benchmark derivations.
{
  config,
  lib,
  pkgs,
  ...
}:
with lib;
let
  cfg = config.services.benchmark-runner;

  builtinProfiles = {
    linux-drm-render = {
      sandboxPaths = [
        "/dev/dri"
        "/sys/class/drm"
        "/sys/dev"
        "/sys/devices"
      ];
      udevRules = [
        ''SUBSYSTEM=="drm", KERNEL=="card*", GROUP="nixbld", MODE="0660"''
        ''SUBSYSTEM=="drm", KERNEL=="renderD*", GROUP="nixbld", MODE="0660"''
      ];
    };

    linux-amd-kfd = {
      includes = [ "linux-drm-render" ];
      sandboxPaths = [
        "/dev/kfd"
        "/sys/class/kfd"
      ];
      udevRules = [
        ''KERNEL=="kfd", GROUP="nixbld", MODE="0660"''
      ];
    };

    linux-nvidia = {
      sandboxPaths = [
        "/dev/nvidia0"
        "/dev/nvidiactl"
        "/dev/nvidia-uvm"
        "/dev/nvidia-uvm-tools"
        "/dev/nvidia-caps"
        # Bind-mount the NVIDIA driver package to /run/opengl-driver so
        # autoAddDriverRunpath RUNPATH entries resolve inside the sandbox.
        "/run/opengl-driver=${config.hardware.nvidia.package}"
      ];
      udevRules = [
        ''KERNEL=="nvidia[0-9]*", GROUP="nixbld", MODE="0660"''
        ''KERNEL=="nvidiactl", GROUP="nixbld", MODE="0660"''
        ''KERNEL=="nvidia-uvm", GROUP="nixbld", MODE="0660"''
        ''KERNEL=="nvidia-uvm-tools", GROUP="nixbld", MODE="0660"''
      ];
    };
  };

  emptyProfile = {
    includes = [ ];
    systemFeatures = [ ];
    sandboxPaths = [ ];
    udevRules = [ ];
  };

  profileSet = builtinProfiles // cfg.profiles;
  profileOrEmpty = name: emptyProfile // (profileSet.${name} or { });

  collectProfileNames =
    seen: names:
    concatMap (
      name:
      if elem name seen then
        [ ]
      else
        let
          profile = profileOrEmpty name;
        in
        collectProfileNames (seen ++ [ name ]) profile.includes ++ [ name ]
    ) names;

  activeProfileNames = unique (collectProfileNames [ ] cfg.enabledProfiles);
  activeProfiles = map profileOrEmpty activeProfileNames;

  featureList = unique (
    cfg.systemFeatures ++ concatMap (profile: profile.systemFeatures) activeProfiles
  );

  sandboxPaths = unique (
    [
      "/dev/shm"
      "/proc"
    ]
    ++ optional (cfg.modelsPath != null) cfg.modelsPath
    ++ concatMap (profile: profile.sandboxPaths) activeProfiles
    ++ cfg.extraSandboxPaths
  );

  udevRules = concatMap (profile: profile.udevRules) activeProfiles ++ cfg.extraUdevRules;
in
{
  options.services.benchmark-runner = {
    enable = mkEnableOption "benchmark host configuration";

    systemFeatures = mkOption {
      type = types.listOf types.str;
      default = [ ];
      example = [
        "gfx1151"
        "aimax395"
        "cuda-sm_89"
      ];
      description = "Nix system features required by benchmark derivations that this host may run.";
    };

    enabledProfiles = mkOption {
      type = types.listOf types.str;
      default = [ ];
      example = [
        "linux-amd-kfd"
        "my-accelerator"
      ];
      description = "Named host profiles to apply for sandbox paths and device permissions.";
    };

    profiles = mkOption {
      type = types.attrsOf (
        types.submodule {
          options = {
            includes = mkOption {
              type = types.listOf types.str;
              default = [ ];
              description = "Other benchmark-runner profiles included by this profile.";
            };
            systemFeatures = mkOption {
              type = types.listOf types.str;
              default = [ ];
              description = "Nix system features contributed by this profile.";
            };
            sandboxPaths = mkOption {
              type = types.listOf types.str;
              default = [ ];
              description = "Sandbox paths contributed by this profile.";
            };
            udevRules = mkOption {
              type = types.listOf types.lines;
              default = [ ];
              description = "udev rules contributed by this profile.";
            };
          };
        }
      );
      default = { };
      description = "Additional benchmark host profiles. These are merged over the built-in profiles.";
    };

    extraSandboxPaths = mkOption {
      type = types.listOf types.str;
      default = [ ];
      example = [ "/sys/class/infiniband" ];
      description = "Additional sandbox paths required by caller-provided benchmark profiles.";
    };

    extraUdevRules = mkOption {
      type = types.listOf types.lines;
      default = [ ];
      description = "Additional udev rules required by caller-provided benchmark profiles.";
    };

    modelsPath = mkOption {
      type = types.nullOr types.str;
      default = "/models";
      description = "Path where benchmark models are stored, or null to skip model path setup.";
    };

    enableSandboxRelaxation = mkOption {
      type = types.bool;
      default = false;
      description = "Whether to relax sandbox restrictions for benchmark builds.";
    };

    ensureModels = mkOption {
      type = types.listOf (
        types.submodule {
          options = {
            repo = mkOption {
              type = types.str;
              description = "Hugging Face repository ID.";
            };
            files = mkOption {
              type = types.listOf types.str;
              default = [ ];
              example = [ "model.gguf" ];
              description = "Specific files to download from the repository.";
            };
            name = mkOption {
              type = types.str;
              description = "Local directory name for the model.";
            };
          };
        }
      );
      default = [ ];
      description = "Models to ensure are downloaded from Hugging Face.";
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = all (name: hasAttr name profileSet) cfg.enabledProfiles;
        message = "services.benchmark-runner.enabledProfiles contains an unknown profile";
      }
      {
        assertion = all (profile: all (name: hasAttr name profileSet) profile.includes) (
          attrValues profileSet
        );
        message = "services.benchmark-runner.profiles contains an unknown included profile";
      }
      {
        assertion = cfg.modelsPath == null || hasPrefix "/" cfg.modelsPath;
        message = "services.benchmark-runner.modelsPath must be null or an absolute path";
      }
      {
        assertion = cfg.modelsPath != null || cfg.ensureModels == [ ];
        message = "services.benchmark-runner.ensureModels requires services.benchmark-runner.modelsPath";
      }
      {
        assertion = all (model: model.files != [ ]) cfg.ensureModels;
        message = "services.benchmark-runner.ensureModels entries must include at least one file";
      }
    ];

    nix.settings = {
      system-features = featureList;
      extra-sandbox-paths = sandboxPaths;
    }
    // optionalAttrs cfg.enableSandboxRelaxation {
      sandbox = "relaxed";
    };

    systemd.tmpfiles.rules =
      optional (cfg.modelsPath != null) "d ${cfg.modelsPath} 0755 root root -"
      ++ optional (cfg.modelsPath != null) "A ${cfg.modelsPath} - - - - group:nixbld:r-x"
      ++ optional (cfg.modelsPath != null) "A ${cfg.modelsPath} - - - - default:group:nixbld:r-x";

    services.udev.extraRules = mkIf (udevRules != [ ]) (concatStringsSep "\n" udevRules);

    systemd.services =
      let
        fileServices = flatten (
          map (
            model:
            map (file: {
              name = "ensure-model-${model.name}-${builtins.replaceStrings [ "." "/" ] [ "-" "-" ] file}";
              value = {
                description = "Download ${file} for model ${model.name}";
                after = [ "network.target" ];
                path = with pkgs; [
                  (python3.withPackages (
                    ps: with ps; [
                      hf-transfer
                      huggingface-hub
                    ]
                  ))
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
            }) model.files
          ) cfg.ensureModels
        );

        orchestratorServices = map (model: {
          name = "ensure-model-${model.name}";
          value = {
            description = "Orchestrate downloads for model ${model.name}";
            wantedBy = [ "multi-user.target" ];
            after = [ "network.target" ];
            wants = map (
              file: "ensure-model-${model.name}-${builtins.replaceStrings [ "." "/" ] [ "-" "-" ] file}.service"
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
