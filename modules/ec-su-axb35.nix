{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.ec-su-axb35;

  fanOptions = {
    options = {
      mode = lib.mkOption {
        type = lib.types.nullOr (lib.types.enum ["auto" "fixed" "curve"]);
        default = null;
        description = "Fan operating mode";
      };

      level = lib.mkOption {
        type = lib.types.nullOr (lib.types.ints.between 0 5);
        default = null;
        description = "Fan speed level (0=0%, 1=20%, 2=40%, 3=60%, 4=80%, 5=100%). Only used when mode is 'fixed'.";
      };

      rampupCurve = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "60,70,83,95,97";
        description = "Five comma-separated temperature thresholds (°C) for fan ramp-up at levels 1-5.";
      };

      rampdownCurve = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "40,50,80,94,96";
        description = "Five comma-separated temperature thresholds (°C) for fan ramp-down at levels 1-5.";
      };
    };
  };

  sysfsBase = "/sys/class/ec_su_axb35";

  writeSysfs = path: value: ''
    echo "${value}" > ${path}
  '';

  fanConfig = name: fanCfg: lib.optionalString (fanCfg.mode != null) ''
    # Configure ${name}
    ${writeSysfs "${sysfsBase}/${name}/mode" fanCfg.mode}
    ${lib.optionalString (fanCfg.level != null) (writeSysfs "${sysfsBase}/${name}/level" (toString fanCfg.level))}
    ${lib.optionalString (fanCfg.rampupCurve != null) (writeSysfs "${sysfsBase}/${name}/rampup_curve" fanCfg.rampupCurve)}
    ${lib.optionalString (fanCfg.rampdownCurve != null) (writeSysfs "${sysfsBase}/${name}/rampdown_curve" fanCfg.rampdownCurve)}
  '';

  configScript = pkgs.writeShellScript "ec-su-axb35-config" ''
    set -euo pipefail

    # Wait for sysfs to be available
    for i in $(seq 1 10); do
      [ -d "${sysfsBase}" ] && break
      sleep 0.5
    done

    if [ ! -d "${sysfsBase}" ]; then
      echo "ec_su_axb35 sysfs not available" >&2
      exit 1
    fi

    ${lib.optionalString (cfg.powerMode != null) ''
      # Set APU power mode
      ${writeSysfs "${sysfsBase}/apu/power_mode" cfg.powerMode}
    ''}

    ${fanConfig "fan1" cfg.fans.fan1}
    ${fanConfig "fan2" cfg.fans.fan2}
    ${fanConfig "fan3" cfg.fans.fan3}

    echo "ec_su_axb35 configuration applied"
  '';

  hasAnyConfig = cfg.powerMode != null
    || cfg.fans.fan1.mode != null
    || cfg.fans.fan2.mode != null
    || cfg.fans.fan3.mode != null;
in {
  options.services.ec-su-axb35 = {
    enable = lib.mkEnableOption "EC-SU_AXB35 embedded controller kernel module";

    monitor.enable = lib.mkEnableOption "EC-SU_AXB35 monitor script";

    powerMode = lib.mkOption {
      type = lib.types.nullOr (lib.types.enum ["quiet" "balanced" "performance"]);
      default = null;
      description = "APU power mode";
    };

    fans = {
      fan1 = lib.mkOption {
        type = lib.types.submodule fanOptions;
        default = {};
        description = "CPU fan 1 configuration";
      };

      fan2 = lib.mkOption {
        type = lib.types.submodule fanOptions;
        default = {};
        description = "CPU fan 2 configuration";
      };

      fan3 = lib.mkOption {
        type = lib.types.submodule fanOptions;
        default = {};
        description = "System fan configuration";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    boot.extraModulePackages = [
      (pkgs.ec-su-axb35 {
        kernel = config.boot.kernelPackages.kernel;
      })
    ];

    boot.kernelModules = ["ec_su_axb35"];

    environment.systemPackages = lib.mkIf cfg.monitor.enable [
      pkgs.ec-su-axb35-monitor
    ];

    systemd.services.ec-su-axb35-config = lib.mkIf hasAnyConfig {
      description = "Configure EC-SU_AXB35 embedded controller settings";
      after = ["systemd-modules-load.service"];
      wantedBy = ["multi-user.target"];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = configScript;
      };
    };
  };
}
