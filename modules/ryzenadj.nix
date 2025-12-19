{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.ryzenadj;

  # Build command line arguments from config
  mkArg = flag: value:
    lib.optionalString (value != null) " --${flag}=${toString value}";

  mkBoolArg = flag: value:
    lib.optionalString value " --${flag}";

  ryzenadj = "${pkgs.ryzenadj}/bin/ryzenadj";

  # Curve optimizer encoding: positive values direct, negative as 1048576 + value
  coallValue = if cfg.curveOptimizer.offset >= 0 then cfg.curveOptimizer.offset
    else 1048576 + cfg.curveOptimizer.offset;

  # Base args (without curve optimizer - applied separately with grace period)
  baseArgs = lib.concatStrings [
    (mkArg "stapm-limit" cfg.stapmLimit)
    (mkArg "fast-limit" cfg.fastLimit)
    (mkArg "slow-limit" cfg.slowLimit)
    (mkArg "apu-slow-limit" cfg.apuSlowLimit)
    (mkArg "tctl-temp" cfg.tctlTemp)
    (mkArg "apu-skin-temp" cfg.apuSkinTemp)
    (mkArg "dgpu-skin-temp" cfg.dgpuSkinTemp)
    (mkBoolArg "max-performance" cfg.maxPerformance)
    (mkBoolArg "power-saving" cfg.powerSaving)
  ];

  hasBaseConfig = cfg.stapmLimit != null
    || cfg.fastLimit != null
    || cfg.slowLimit != null
    || cfg.apuSlowLimit != null
    || cfg.tctlTemp != null
    || cfg.apuSkinTemp != null
    || cfg.dgpuSkinTemp != null
    || cfg.maxPerformance
    || cfg.powerSaving;

  # Verification checks - map config option to output name and expected value
  checks = lib.filter (x: x.value != null) [
    { name = "STAPM LIMIT"; value = cfg.stapmLimit; divisor = 1000; }
    { name = "PPT LIMIT FAST"; value = cfg.fastLimit; divisor = 1000; }
    { name = "PPT LIMIT SLOW"; value = cfg.slowLimit; divisor = 1000; }
    { name = "PPT LIMIT APU"; value = cfg.apuSlowLimit; divisor = 1000; }
    { name = "THM LIMIT CORE"; value = cfg.tctlTemp; divisor = 1; }
    { name = "STT LIMIT APU"; value = cfg.apuSkinTemp; divisor = 1; }
    { name = "STT LIMIT dGPU"; value = cfg.dgpuSkinTemp; divisor = 1; }
  ];

  verifyScript = lib.concatMapStringsSep "\n" (check: ''
    expected="${toString (check.value / check.divisor)}"
    actual=$(echo "$output" | grep -F "| ${check.name}" | awk -F'|' '{print $3}' | tr -d ' ' | cut -d. -f1)
    if [ "$actual" != "$expected" ]; then
      echo "Verification failed for ${check.name}: expected $expected, got $actual" >&2
      failed=1
    else
      echo "Verified ${check.name}: $actual"
    fi
  '') checks;

  configScript = pkgs.writeShellScript "ryzenadj-config" ''
    set -euo pipefail

    ${lib.optionalString hasBaseConfig ''
      echo "Applying ryzenadj settings..."
      ${ryzenadj}${baseArgs}

      echo "Verifying settings..."
      output=$(${ryzenadj} -i)
      failed=0

      ${verifyScript}

      if [ "$failed" -eq 1 ]; then
        echo "Some settings could not be verified" >&2
        exit 1
      fi

      echo "All ryzenadj settings applied and verified"
    ''}

    ${lib.optionalString cfg.curveOptimizer.enable ''
      ${lib.optionalString (cfg.curveOptimizer.graceSeconds > 0) ''
        echo "Waiting ${toString cfg.curveOptimizer.graceSeconds}s before applying curve optimizer (CO=${toString cfg.curveOptimizer.offset})..."
        echo "To abort: sudo systemctl stop ryzenadj"
        sleep ${toString cfg.curveOptimizer.graceSeconds}
      ''}
      echo "Applying curve optimizer: ${toString cfg.curveOptimizer.offset} (encoded: ${toString coallValue})"
      ${ryzenadj} --set-coall=${toString coallValue}
      echo "Curve optimizer applied"
    ''}
  '';

  hasAnyConfig = hasBaseConfig || cfg.curveOptimizer.enable;
in {
  options.services.ryzenadj = {
    enable = lib.mkEnableOption "ryzenadj power management";

    stapmLimit = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      example = 150000;
      description = "Sustained Power Limit (STAPM) in mW";
    };

    fastLimit = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      example = 200000;
      description = "Actual Power Limit (PPT FAST) in mW";
    };

    slowLimit = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      example = 160000;
      description = "Average Power Limit (PPT SLOW) in mW";
    };

    apuSlowLimit = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      example = 70000;
      description = "APU PPT Slow Power limit in mW";
    };

    tctlTemp = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      example = 98;
      description = "Tctl Temperature Limit in °C";
    };

    apuSkinTemp = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      example = 98;
      description = "APU Skin Temperature Limit in °C";
    };

    dgpuSkinTemp = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      example = 98;
      description = "dGPU Skin Temperature Limit in °C";
    };

    maxPerformance = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Enable max performance mode";
    };

    powerSaving = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Enable power saving mode";
    };

    curveOptimizer = {
      enable = lib.mkEnableOption "curve optimizer undervolting";

      offset = lib.mkOption {
        type = lib.types.int;
        default = 0;
        example = -30;
        description = "All-core Curve Optimizer offset. Negative = undervolt (more efficient), positive = overvolt. Use ryzenadj-co-test to find optimal value.";
      };

      graceSeconds = lib.mkOption {
        type = lib.types.int;
        default = 60;
        example = 120;
        description = "Seconds to wait before applying curve optimizer. Allows recovery via SSH if value causes instability. Set to 0 to disable.";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = !(cfg.maxPerformance && cfg.powerSaving);
        message = "Cannot enable both maxPerformance and powerSaving";
      }
    ];

    hardware.cpu.amd.ryzen-smu.enable = true;

    environment.systemPackages = [ pkgs.ryzenadj pkgs.ryzenadj-co-test ];

    systemd.services.ryzenadj = lib.mkIf hasAnyConfig {
      description = "Configure AMD Ryzen power limits with ryzenadj";
      after = ["systemd-modules-load.service"];
      wantedBy = ["multi-user.target"];
      path = [ pkgs.gawk pkgs.gnugrep pkgs.coreutils ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = configScript;
      };
    };
  };
}
