{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.programs.amduprof;
  powerProfilerModule = pkgs.amduprof-power-profiler {
    inherit (config.boot.kernelPackages) kernel;
  };
  loadPowerProfiler = pkgs.writeShellScript "amduprof-load-power-profiler" ''
    exec ${pkgs.kmod}/bin/modprobe \
      -d /run/current-system/kernel-modules \
      AMDPowerProfiler
  '';
  createDevice = pkgs.writeShellScript "amduprof-create-power-profiler-device" ''
    set -eu

    procDevice=/proc/AMDPowerProfiler/device
    for _ in $(${pkgs.coreutils}/bin/seq 1 50); do
      [ -r "$procDevice" ] && break
      ${pkgs.coreutils}/bin/sleep 0.1
    done

    if [ ! -r "$procDevice" ]; then
      echo "AMDPowerProfiler did not publish its device number" >&2
      exit 1
    fi

    major=$(${pkgs.coreutils}/bin/cat "$procDevice")
    case "$major" in
      *[!0-9]*)
        echo "invalid AMDPowerProfiler major device number: $major" >&2
        exit 1
        ;;
    esac
    if [ -z "$major" ]; then
      echo "AMDPowerProfiler returned an empty major device number" >&2
      exit 1
    fi

    ${pkgs.coreutils}/bin/rm -f /dev/AMDPowerProfiler
    ${pkgs.coreutils}/bin/mknod /dev/AMDPowerProfiler c "$major" 0
    ${pkgs.coreutils}/bin/chmod ${cfg.powerProfiling.deviceMode} /dev/AMDPowerProfiler
  '';
in
{
  options.programs.amduprof = {
    enable = lib.mkEnableOption "AMD uProf performance analysis tools";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.amduprof;
      defaultText = lib.literalExpression "pkgs.amduprof";
      description = "AMD uProf user-space package to install.";
    };

    powerProfiling = {
      enable = lib.mkEnableOption "the AMD uProf power-profiling kernel module";

      deviceMode = lib.mkOption {
        type = lib.types.strMatching "0[0-7]{3}";
        default = "0666";
        description = "Permissions for /dev/AMDPowerProfiler (upstream defaults to 0666).";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ cfg.package ];

    boot.extraModulePackages = lib.mkIf cfg.powerProfiling.enable [ powerProfilerModule ];
    boot.kernelModules = lib.mkIf cfg.powerProfiling.enable [ "AMDPowerProfiler" ];

    systemd.services.amduprof-power-profiler-device = lib.mkIf cfg.powerProfiling.enable {
      description = "Create the AMD uProf power profiler device node";
      after = [ "systemd-modules-load.service" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStartPre = loadPowerProfiler;
        ExecStart = createDevice;
      };
    };
  };
}
