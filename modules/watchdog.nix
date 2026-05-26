{ config
, lib
, pkgs
, ...
}:
let
  cfg = config.hardware.strixHalo.watchdog;
in
{
  options.hardware.strixHalo.watchdog = {
    enable = lib.mkEnableOption "Strix Halo hardware watchdog and panic recovery policy";

    hardwareModule = lib.mkOption {
      type = lib.types.str;
      default = "sp5100_tco";
      description = "Kernel watchdog module to load for the platform hardware watchdog.";
    };

    heartbeatSec = lib.mkOption {
      type = lib.types.ints.positive;
      default = 30;
      description = "Heartbeat passed to the hardware watchdog driver at module load.";
    };

    nowayout = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Keep the hardware watchdog armed once started.";
    };

    runtimeWatchdogSec = lib.mkOption {
      type = lib.types.str;
      default = "15s";
      description = "systemd RuntimeWatchdogSec value.";
    };

    rebootWatchdogSec = lib.mkOption {
      type = lib.types.str;
      default = "30s";
      description = "systemd RebootWatchdogSec value.";
    };

    kexecWatchdogSec = lib.mkOption {
      type = lib.types.str;
      default = "30s";
      description = "systemd KExecWatchdogSec value.";
    };

    panicTimeoutSec = lib.mkOption {
      type = lib.types.ints.unsigned;
      default = 5;
      description = "Seconds to wait before rebooting after a kernel panic.";
    };

    panicOnOops = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Convert kernel oopses into panics.";
    };

    panicOnSoftLockup = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Convert soft lockups into panics.";
    };

    panicOnHardLockup = lib.mkOption {
      type = lib.types.bool;
      default = pkgs.stdenv.hostPlatform.isx86_64;
      description = "Enable the x86 NMI watchdog and panic on hard lockups.";
    };

    panicOnHungTask = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Convert hung task detector events into panics.";
    };
  };

  config = lib.mkIf cfg.enable {
    boot.kernelModules = [ cfg.hardwareModule ];
    boot.extraModprobeConfig = ''
      options ${cfg.hardwareModule} heartbeat=${toString cfg.heartbeatSec} nowayout=${if cfg.nowayout then "1" else "0"} action=0
    '';

    systemd.settings.Manager = {
      RuntimeWatchdogSec = cfg.runtimeWatchdogSec;
      RebootWatchdogSec = cfg.rebootWatchdogSec;
      KExecWatchdogSec = cfg.kexecWatchdogSec;
    };

    boot.kernelParams =
      [ "panic=${toString cfg.panicTimeoutSec}" ]
      ++ lib.optional cfg.panicOnOops "panic_on_oops=1"
      ++ lib.optional cfg.panicOnSoftLockup "softlockup_panic=1"
      ++ lib.optional cfg.panicOnHungTask "hung_task_panic=1"
      ++ lib.optional cfg.panicOnHardLockup "nmi_watchdog=panic,1";

    boot.kernel.sysctl =
      {
        "kernel.panic" = cfg.panicTimeoutSec;
        "kernel.watchdog" = 1;
      }
      // lib.optionalAttrs cfg.panicOnOops {
        "kernel.panic_on_oops" = 1;
      }
      // lib.optionalAttrs cfg.panicOnSoftLockup {
        "kernel.softlockup_panic" = 1;
      }
      // lib.optionalAttrs cfg.panicOnHungTask {
        "kernel.hung_task_panic" = 1;
      }
      // lib.optionalAttrs (pkgs.stdenv.hostPlatform.isx86_64 && cfg.panicOnHardLockup) {
        "kernel.nmi_watchdog" = 1;
        "kernel.hardlockup_panic" = 1;
      };
  };
}
