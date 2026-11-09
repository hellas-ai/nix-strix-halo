# Provisions the strix-halo cluster dashboards into a Grafana instance
# running on the same host (services.grafana). Import on the monitoring
# host, not on the cluster nodes.
{
  config,
  lib,
  pkgs,
  ...
}:
with lib;
let
  cfg = config.services.strix-halo.grafana-dashboards;

  dashboards = import ../dashboards { inherit lib pkgs; };

  clusterView = dashboards.clusterViewWith {
    inherit (cfg)
      instancePattern
      fabricHcaPattern
      lanDevicePattern
      expectedNodes
      vllmJob
      ;
  };

  dashboardDir = pkgs.linkFarm "strix-halo-grafana-dashboards" [
    {
      name = "strix-halo-cluster.json";
      path = clusterView;
    }
  ];
in
{
  options.services.strix-halo.grafana-dashboards = {
    enable = mkEnableOption "strix-halo Grafana dashboards";

    folder = mkOption {
      type = types.str;
      default = "Strix Halo";
      description = "Grafana folder the dashboards are provisioned into";
    };

    instancePattern = mkOption {
      type = types.str;
      default = "strix-.*";
      description = "Regex matching the `instance` label of the cluster nodes across all scrape jobs";
    };

    fabricHcaPattern = mkOption {
      type = types.str;
      default = "mlx5.*";
      description = "Regex matching the fabric RDMA HCA (InfiniBand device) names; the fabric is RoCE, so throughput comes from node_infiniband_port_data_* not netdev counters";
    };

    lanDevicePattern = mkOption {
      type = types.str;
      default = "eno.*";
      description = "Regex matching the LAN network device names";
    };

    expectedNodes = mkOption {
      type = types.int;
      default = 4;
      description = "Number of nodes in the cluster; shown in the node status table's description";
    };

    vllmJob = mkOption {
      type = types.str;
      default = "vllm";
      description = "Scrape job name of the vLLM OpenAI API servers' /metrics endpoints";
    };
  };

  config = mkIf cfg.enable {
    services.grafana.provision.dashboards.settings = {
      apiVersion = 1;
      providers = [
        {
          name = "strix-halo";
          inherit (cfg) folder;
          type = "file";
          disableDeletion = false;
          allowUiUpdates = false;
          updateIntervalSeconds = 30;
          options.path = dashboardDir;
        }
      ];
    };
  };
}
