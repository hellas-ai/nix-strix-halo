# Runs amdgpu-smu-exporter as a node_exporter textfile source on a Strix
# Halo node. The SMU reader exports the APU's richer v3.0 metrics table as
# amd_smu_*. A companion sysfs reader exports stable, device-labelled
# amd_gpu_* metrics for every AMD GPU, including discrete Radeon Pro V620s.
# Both use node_exporter's textfile endpoint without pulling ROCm into the
# system closure or opening another port.
# Import on the cluster nodes.
{
  config,
  lib,
  pkgs,
  ...
}:
with lib;
let
  cfg = config.services.strix-halo.smu-exporter;

  sysfsExporter = pkgs.writeShellApplication {
    name = "amdgpu-sysfs-exporter";
    runtimeInputs = [ pkgs.coreutils ];
    text = ''
      output_path=${escapeShellArg "${cfg.textfileDirectory}/amdgpu-sysfs.prom"}
      temporary_path="$output_path.tmp.$$"

      cleanup() {
        rm -f "$temporary_path"
      }
      trap cleanup EXIT
      trap 'exit 0' INT TERM

      write_snapshot() {
        {
          echo '# HELP amd_gpu_info Static identity of an AMD GPU.'
          echo '# TYPE amd_gpu_info gauge'
          echo '# HELP amd_gpu_busy_percent Current GPU engine utilization.'
          echo '# TYPE amd_gpu_busy_percent gauge'
          echo '# HELP amd_gpu_memory_bytes GPU memory capacity and usage.'
          echo '# TYPE amd_gpu_memory_bytes gauge'
          echo '# HELP amd_gpu_power_watts GPU board power reported by hwmon.'
          echo '# TYPE amd_gpu_power_watts gauge'
          echo '# HELP amd_gpu_temperature_celsius GPU temperatures reported by hwmon.'
          echo '# TYPE amd_gpu_temperature_celsius gauge'
          echo '# HELP amd_gpu_clock_hertz GPU clocks reported by hwmon.'
          echo '# TYPE amd_gpu_clock_hertz gauge'
          echo '# HELP amd_gpu_metrics_timestamp_seconds Unix time of this sysfs sample.'
          echo '# TYPE amd_gpu_metrics_timestamp_seconds gauge'

          for card in /sys/class/drm/card[0-9]*; do
            device_path="$card/device"
            [[ -r "$device_path/vendor" && -r "$device_path/device" ]] || continue

            IFS= read -r vendor_id < "$device_path/vendor"
            [[ "$vendor_id" == "0x1002" ]] || continue
            IFS= read -r device_id < "$device_path/device"

            card_name=$(basename "$card")
            pci_address=$(basename "$(readlink -f "$device_path")")
            case "$device_id" in
              0x73a1)
                model="Radeon Pro V620"
                kind="discrete"
                ;;
              0x1586)
                model="Strix Halo APU"
                kind="integrated"
                ;;
              *)
                model="$device_id"
                kind="unknown"
                ;;
            esac
            labels="card=\"$card_name\",pci=\"$pci_address\",device_id=\"$device_id\",model=\"$model\",kind=\"$kind\""

            printf 'amd_gpu_info{%s} 1\n' "$labels"

            if [[ -r "$device_path/gpu_busy_percent" ]]; then
              IFS= read -r value < "$device_path/gpu_busy_percent"
              printf 'amd_gpu_busy_percent{%s} %s\n' "$labels" "$value"
            fi

            for pool in vram gtt; do
              for state in total used; do
                value_path="$device_path/mem_info_''${pool}_''${state}"
                [[ -r "$value_path" ]] || continue
                IFS= read -r value < "$value_path"
                printf 'amd_gpu_memory_bytes{%s,pool="%s",state="%s"} %s\n' \
                  "$labels" "$pool" "$state" "$value"
              done
            done

            for hwmon in "$device_path"/hwmon/hwmon*; do
              [[ -d "$hwmon" ]] || continue

              for input in "$hwmon"/power*_average; do
                [[ -r "$input" ]] || continue
                base="''${input%_average}"
                sensor="''${base##*/}"
                if [[ -r "''${base}_label" ]]; then
                  IFS= read -r sensor < "''${base}_label"
                fi
                IFS= read -r value < "$input"
                printf 'amd_gpu_power_watts{%s,sensor="%s"} %se-6\n' \
                  "$labels" "$sensor" "$value"
              done

              for input in "$hwmon"/temp*_input; do
                [[ -r "$input" ]] || continue
                base="''${input%_input}"
                sensor="''${base##*/}"
                if [[ -r "''${base}_label" ]]; then
                  IFS= read -r sensor < "''${base}_label"
                fi
                IFS= read -r value < "$input"
                printf 'amd_gpu_temperature_celsius{%s,sensor="%s"} %se-3\n' \
                  "$labels" "$sensor" "$value"
              done

              for input in "$hwmon"/freq*_input; do
                [[ -r "$input" ]] || continue
                base="''${input%_input}"
                domain="''${base##*/}"
                if [[ -r "''${base}_label" ]]; then
                  IFS= read -r domain < "''${base}_label"
                fi
                IFS= read -r value < "$input"
                printf 'amd_gpu_clock_hertz{%s,domain="%s"} %s\n' \
                  "$labels" "$domain" "$value"
              done
            done
          done

          printf 'amd_gpu_metrics_timestamp_seconds %s\n' "$(date +%s)"
        } > "$temporary_path"
        mv -f "$temporary_path" "$output_path"
      }

      while true; do
        write_snapshot
        sleep ${toString cfg.intervalMs}e-3
      done
    '';
  };
in
{
  options.services.strix-halo.smu-exporter = {
    enable = mkEnableOption "AMDGPU SMU Prometheus textfile exporter";

    package = mkOption {
      type = types.package;
      default = pkgs.amdgpu-smu-exporter;
      defaultText = literalExpression "pkgs.amdgpu-smu-exporter";
      description = "The amdgpu-smu-exporter package.";
    };

    textfileDirectory = mkOption {
      type = types.path;
      default = "/var/lib/amdgpu-smu-exporter";
      description = ''
        Directory the exporter writes its .prom file into. Other textfile
        sources (e.g. the NPU exporter) share it; node_exporter's textfile
        collector is pointed here when configureNodeExporter is on.
      '';
    };

    intervalMs = mkOption {
      type = types.ints.positive;
      default = 1000;
      description = "Sampling interval in milliseconds.";
    };

    configureNodeExporter = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Point node_exporter's textfile collector at textfileDirectory.
        Disable if the collector is already configured elsewhere.
      '';
    };
  };

  config = mkIf cfg.enable {
    services.prometheus.exporters.node = mkIf cfg.configureNodeExporter {
      enabledCollectors = [ "textfile" ];
      extraFlags = [ "--collector.textfile.directory=${cfg.textfileDirectory}" ];
    };

    systemd.tmpfiles.rules = [
      "d ${cfg.textfileDirectory} 0750 node-exporter node-exporter - -"
    ];

    systemd.services.amdgpu-smu-exporter = {
      description = "Export Strix Halo SMU metrics for Prometheus";
      wantedBy = [ "multi-user.target" ];
      before = [ "prometheus-node-exporter.service" ];
      serviceConfig = {
        Type = "simple";
        User = "node-exporter";
        Group = "node-exporter";
        ExecStart = ''
          ${getExe cfg.package} \
            --output ${cfg.textfileDirectory}/amdgpu-smu.prom \
            --interval-ms ${toString cfg.intervalMs}
        '';
        Restart = "on-failure";
        RestartSec = "1s";
        ReadWritePaths = [ cfg.textfileDirectory ];
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
      };
    };

    systemd.services.amdgpu-sysfs-exporter = {
      description = "Export per-GPU AMD sysfs metrics for Prometheus";
      wantedBy = [ "multi-user.target" ];
      before = [ "prometheus-node-exporter.service" ];
      serviceConfig = {
        Type = "simple";
        User = "node-exporter";
        Group = "node-exporter";
        ExecStart = getExe sysfsExporter;
        Restart = "on-failure";
        RestartSec = "1s";
        ReadWritePaths = [ cfg.textfileDirectory ];
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
      };
    };
  };
}
