# Runs amd-npu-exporter as a node_exporter textfile source on a Strix Halo
# node. Additive: it writes into the directory node_exporter's textfile
# collector already scans (the same one the SMU exporter uses) and does not
# touch the node_exporter or SMU wiring. Import on the cluster nodes.
{
  config,
  lib,
  pkgs,
  ...
}:
with lib;
let
  cfg = config.services.strix-halo.npu-exporter;
in
{
  options.services.strix-halo.npu-exporter = {
    enable = mkEnableOption "AMD XDNA NPU Prometheus textfile exporter";

    package = mkOption {
      type = types.package;
      default = pkgs.amd-npu-exporter;
      defaultText = literalExpression "pkgs.amd-npu-exporter";
      description = "The amd-npu-exporter package (rebuilt against the host kernel headers).";
    };

    device = mkOption {
      type = types.str;
      default = "/dev/accel/accel0";
      description = "amdxdna accel render node to read telemetry from.";
    };

    textfileDirectory = mkOption {
      type = types.path;
      default = "/var/lib/amdgpu-smu-exporter";
      description = ''
        Directory node_exporter's textfile collector scans. Defaults to the
        directory the SMU exporter already uses so NPU metrics appear on the
        same endpoint with no extra collector configuration.
      '';
    };

    intervalMs = mkOption {
      type = types.ints.positive;
      default = 1000;
      description = "Sampling interval in milliseconds.";
    };
  };

  config = mkIf cfg.enable (
    let
      kernelHeaders = pkgs.makeLinuxHeaders {
        inherit (config.boot.kernelPackages.kernel) src version;
      };
      exporter = cfg.package.override { linuxHeaders = kernelHeaders; };
    in
    {
      systemd.tmpfiles.rules = [
        "d ${cfg.textfileDirectory} 0750 node-exporter node-exporter - -"
      ];

      systemd.services.amd-npu-exporter = {
        description = "Export Strix Halo NPU (amdxdna) metrics for Prometheus";
        wantedBy = [ "multi-user.target" ];
        before = [ "prometheus-node-exporter.service" ];
        serviceConfig = {
          Type = "simple";
          User = "node-exporter";
          Group = "node-exporter";
          # The accel render node is root:video 0660; join video to open it.
          SupplementaryGroups = [ "video" ];
          ExecStart = "${getExe exporter} --device ${cfg.device} --output ${cfg.textfileDirectory}/amd-npu.prom --interval-ms ${toString cfg.intervalMs}";
          Restart = "on-failure";
          RestartSec = "1s";
          ReadWritePaths = [ cfg.textfileDirectory ];
          NoNewPrivileges = true;
          ProtectSystem = "strict";
          ProtectHome = true;
          PrivateTmp = true;
          # Not PrivateDevices: the exporter needs the host's /dev/accel node.
        };
      };
    }
  );
}
