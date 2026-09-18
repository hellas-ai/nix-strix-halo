# Runs amdgpu-smu-exporter as a node_exporter textfile source on a Strix
# Halo node. node_exporter's hwmon collector sees only edge temperature,
# PPT and sclk on this APU; this exports AMDGPU's richer, versioned SMU
# metrics table (amd_smu_*) through the existing node_exporter endpoint
# without pulling ROCm into the system closure or opening another port.
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
  };
}
