{ lib
, pkgs
,
}:

let
  inherit (lib)
    hasPrefix
    mkOption
    optionalAttrs
    types
    unique
    ;

  normalizeAmdGpuArch = arch: if hasPrefix "gfx" arch then arch else "gfx${arch}";
  normalizeAmdNpuArch = arch: if hasPrefix "xdna" arch then arch else "xdna${arch}";

  gpuFeature =
    gpu:
    if gpu.systemFeature != null then
      gpu.systemFeature
    else if gpu.type == "amd" then
      normalizeAmdGpuArch gpu.arch
    else
      gpu.arch;

  npuFeature =
    npu:
    if npu.systemFeature != null then
      npu.systemFeature
    else if npu.type == "amd" then
      normalizeAmdNpuArch npu.arch
    else
      npu.arch;

  gpuModule = types.submodule {
    options = {
      type = mkOption {
        type = types.enum [
          "amd"
          "nvidia"
          "intel"
        ];
        example = "amd";
        description = "GPU vendor or driver family.";
      };

      arch = mkOption {
        type = types.str;
        example = "1151";
        description = ''
          GPU architecture identifier. AMD values may be either `1151`
          or `gfx1151`; non-AMD values are used as-is.
        '';
      };

      systemFeature = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "gfx1151";
        description = ''
          Nix system feature to advertise for this GPU. Defaults to
          `gfx<arch>` for AMD GPUs and `arch` for other GPU types.
        '';
      };
    };
  };

  npuModule = types.submodule {
    options = {
      type = mkOption {
        type = types.enum [ "amd" ];
        default = "amd";
        description = "NPU vendor or driver family.";
      };

      arch = mkOption {
        type = types.str;
        example = "xdna2";
        description = ''
          NPU architecture identifier. AMD values may be either `2`
          or `xdna2`.
        '';
      };

      systemFeature = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "xdna2";
        description = ''
          Nix system feature to advertise for this NPU. Defaults to
          `xdna<arch>` for AMD NPUs unless `arch` already starts with
          `xdna`.
        '';
      };
    };
  };

  builderModule = types.submodule (
    { name, ... }:
    {
      options = {
        enable = mkOption {
          type = types.bool;
          default = true;
          description = "Whether to include this remote benchmark runner as a Nix builder.";
        };

        hostName = mkOption {
          type = types.str;
          default = name;
          description = "SSH hostname for the remote Nix builder.";
        };

        sshUser = mkOption {
          type = types.str;
          default = "root";
          description = "SSH user for remote builds.";
        };

        protocol = mkOption {
          type = types.enum [
            "ssh"
            "ssh-ng"
          ];
          default = "ssh-ng";
          description = "Nix remote build protocol.";
        };

        systems = mkOption {
          type = types.listOf types.str;
          default = [ pkgs.stdenv.hostPlatform.system ];
          description = "Nix systems supported by the remote builder.";
        };

        maxJobs = mkOption {
          type = types.ints.positive;
          default = 1;
          description = "Maximum concurrent builds on this remote builder.";
        };

        speedFactor = mkOption {
          type = types.ints.positive;
          default = 1;
          description = "Relative speed factor for Nix builder scheduling.";
        };

        sshKey = mkOption {
          type = types.nullOr types.path;
          default = null;
          description = "Optional private key used by Nix to connect to this builder.";
        };

        publicKey = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "Optional SSH host public key to add to programs.ssh.knownHosts.";
        };

        gpus = mkOption {
          type = types.listOf gpuModule;
          default = [ ];
          example = [
            {
              type = "amd";
              arch = "1151";
            }
          ];
          description = "GPU capabilities advertised by the remote runner.";
        };

        npus = mkOption {
          type = types.listOf npuModule;
          default = [ ];
          example = [
            {
              type = "amd";
              arch = "xdna2";
            }
          ];
          description = "NPU capabilities advertised by the remote runner.";
        };

        systemFeatures = mkOption {
          type = types.listOf types.str;
          default = [ ];
          example = [
            "rocm"
            "kvm"
            "big-parallel"
          ];
          description = "Additional Nix system features advertised by the remote runner.";
        };

        supportedFeatures = mkOption {
          type = types.nullOr (types.listOf types.str);
          default = null;
          description = ''
            Complete supportedFeatures list for this builder. When null,
            features are derived from systemFeatures, gpus, and npus.
          '';
        };
      };
    }
  );

  builderFeatures =
    builder:
    if builder.supportedFeatures != null then
      builder.supportedFeatures
    else
      unique (builder.systemFeatures ++ map gpuFeature builder.gpus ++ map npuFeature builder.npus);

  mkBuildMachine =
    builder:
    {
      inherit (builder)
        hostName
        sshUser
        protocol
        systems
        maxJobs
        speedFactor
        ;
      supportedFeatures = builderFeatures builder;
    }
    // optionalAttrs (builder.sshKey != null) {
      inherit (builder) sshKey;
    };
in
{
  inherit
    builderFeatures
    builderModule
    gpuFeature
    gpuModule
    mkBuildMachine
    npuFeature
    npuModule
    ;
}
