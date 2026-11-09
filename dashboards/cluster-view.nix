# Cluster-wide view over the strix-halo nodes: one dashboard, all nodes.
# Metric selectors are parameterized so the same dashboard works for any
# deployment of the cluster (node naming, fabric NIC naming, node count).
{
  glib,
  uid ? "strix-halo-cluster",
  title ? "Strix Halo Cluster",
  instancePattern ? "strix-.*",
  # The fabric is RDMA/RoCE, whose traffic never touches netdev byte counters;
  # match the HCA (InfiniBand device) name instead and read its port counters.
  fabricHcaPattern ? "mlx5.*",
  lanDevicePattern ? "eno.*",
  expectedNodes ? 4,
  vllmJob ? "vllm",
}:
let
  inherit (glib)
    dashboard
    timeseries
    stat
    table
    target
    instantTarget
    ;

  p = instancePattern;

  # vLLM engines come and go with the workload, so every panel here must be
  # legible with zero engines up: gaps instead of bridged lines, "none"
  # instead of "N/A", and 0 engines is a neutral state rather than an alert.
  v = "job=\"${vllmJob}\",instance=~\"${p}\"";
  vllmSeries = timeseriesArgs: timeseries (timeseriesArgs // { spanNulls = false; });
  quantile =
    q: metric:
    "histogram_quantile(${q}, sum by (le, instance, model_name) (rate(vllm:${metric}_bucket{${v}}[$__rate_interval])))";

  green = {
    color = "green";
    value = null;
  };
  warnAbove = value: {
    color = "orange";
    inherit value;
  };
  critAbove = value: {
    color = "red";
    inherit value;
  };

  # Helpers for the node status table's field overrides.
  byName = name: properties: {
    matcher = {
      id = "byName";
      options = name;
    };
    inherit properties;
  };
  prop = id: value: { inherit id value; };
  valueMap = options: [
    {
      type = "value";
      inherit options;
    }
  ];
  upDownMap = valueMap {
    "0" = {
      text = "Down";
      color = "red";
    };
    "1" = {
      text = "Up";
      color = "green";
    };
  };
  linkThresholds = prop "thresholds" {
    mode = "absolute";
    steps = [
      {
        color = "red";
        value = null;
      }
      {
        color = "green";
        value = 1;
      }
    ];
  };

  # Per-node status queries feed the header table: instant, table format,
  # aggregated down to a bare instance label so Grafana joins them into one
  # row per node. label_replace drops the :port from the instance label.
  nodeCol = q: instantTarget ''label_replace(${q}, "instance", "$1", "instance", "(.+):[0-9]+")'';
in
dashboard {
  inherit uid title;
  description = "Cluster-wide view across the strix-halo nodes (generated from nix-strix-halo, do not edit in the UI).";
  tags = [
    "strix-halo"
    "cluster"
  ];
  rows = [
    {
      title = "Cluster Overview";
      height = 6;
      placements = [
        {
          x = 0;
          w = 15;
          panel = table {
            title = "Nodes";
            description = "One row per node (${toString expectedNodes} expected). Status is the node-exporter up{} series: when a node's scrape fails the row shows Down and the other columns go blank. Link columns count ports in state ACTIVE. USB4 link speed is not exported yet, so that column is a placeholder.";
            targets = [
              (nodeCol "max by (instance) (up{job=\"node\",instance=~\"${p}\"})")
              (nodeCol "count by (instance) (node_infiniband_state_id{device=~\"usb4_rdma.*\",instance=~\"${p}\"} == 4)")
              (nodeCol "0 * max by (instance) (up{job=\"node\",instance=~\"${p}\"})")
              (nodeCol "count by (instance) (node_infiniband_state_id{device=~\"${fabricHcaPattern}\",instance=~\"${p}\"} == 4)")
              (nodeCol "count by (instance) (node_network_up{device=~\"${lanDevicePattern}\",instance=~\"${p}\"} == 1)")
              (nodeCol "max by (instance) (amd_npu_up{instance=~\"${p}\"})")
              (nodeCol "max by (instance) (node_hwmon_temp_celsius{instance=~\"${p}\"})")
              (nodeCol "max by (instance) (node_load1{instance=~\"${p}\"})")
            ];
            columns = [
              {
                from = "instance";
                to = "Node";
              }
              {
                from = "Value";
                to = "Status";
              }
              {
                from = "Value 2";
                to = "USB4";
              }
              {
                from = "Value 3";
                to = "USB4 speed";
              }
              {
                from = "Value 4";
                to = "InfiniBand";
              }
              {
                from = "Value 5";
                to = "Ethernet";
              }
              {
                from = "Value 6";
                to = "NPU";
              }
              {
                from = "Value 7";
                to = "Temp";
              }
              {
                from = "Value 8";
                to = "Load";
              }
            ];
            overrides = [
              (byName "Status" [ (prop "mappings" upDownMap) ])
              (byName "USB4" [
                (prop "unit" "short")
                (prop "decimals" 0)
                linkThresholds
              ])
              (byName "USB4 speed" [
                (prop "mappings" (valueMap {
                  "0" = {
                    text = "n/a";
                    color = "text";
                  };
                }))
              ])
              (byName "InfiniBand" [
                (prop "unit" "short")
                (prop "decimals" 0)
                linkThresholds
              ])
              (byName "Ethernet" [
                (prop "unit" "short")
                (prop "decimals" 0)
                linkThresholds
              ])
              (byName "NPU" [ (prop "mappings" upDownMap) ])
              (byName "Temp" [
                (prop "unit" "celsius")
                (prop "decimals" 0)
                (prop "thresholds" {
                  mode = "absolute";
                  steps = [
                    green
                    (warnAbove 80)
                    (critAbove 95)
                  ];
                })
              ])
              (byName "Load" [ (prop "decimals" 1) ])
            ];
          };
        }
        {
          x = 15;
          w = 9;
          y = 0;
          h = 2;
          panel = stat {
            title = "Cluster APU Power";
            targets = [
              (target "sum(amd_smu_power_watts{domain=\"apu\",instance=~\"${p}\"})" "APUs")
            ];
            unit = "watt";
            decimals = 0;
          };
        }
        {
          x = 15;
          w = 9;
          y = 2;
          h = 2;
          panel = stat {
            title = "Avg CPU Utilization";
            targets = [
              (target "100 * (1 - avg(rate(node_cpu_seconds_total{mode=\"idle\",instance=~\"${p}\"}[$__rate_interval])))" "cpu")
            ];
            unit = "percent";
            decimals = 0;
            thresholds = [
              green
              (warnAbove 70)
              (critAbove 90)
            ];
          };
        }
        {
          x = 15;
          w = 9;
          y = 4;
          h = 2;
          panel = stat {
            title = "Avg APU Activity";
            targets = [
              (target "avg(amd_smu_activity_percent{domain=\"gfx\",instance=~\"${p}\"})" "gfx")
            ];
            unit = "percent";
            decimals = 0;
          };
        }
      ];
    }
    {
      title = "Cluster Health";
      height = 4;
      panels = [
        (stat {
          title = "Nodes Up";
          targets = [ (target "count(up{job=\"node\",instance=~\"${p}\"} == 1)" "nodes") ];
          decimals = 0;
        })
        (stat {
          title = "NPUs Up";
          targets = [ (target "sum(amd_npu_up{instance=~\"${p}\"})" "npus") ];
          decimals = 0;
          noValue = "none";
        })
        (stat {
          title = "V620s Detected";
          targets = [
            (target "count(amd_gpu_info{device_id=\"0x73a1\",instance=~\"${p}\"})" "gpus")
          ];
          decimals = 0;
          noValue = "none";
        })
        (stat {
          title = "Memory Used";
          targets = [
            (target "100 * (1 - sum(node_memory_MemAvailable_bytes{instance=~\"${p}\"}) / sum(node_memory_MemTotal_bytes{instance=~\"${p}\"}))" "memory")
          ];
          unit = "percent";
          decimals = 0;
        })
        (stat {
          title = "Hottest Sensor";
          targets = [ (target "max(node_hwmon_temp_celsius{instance=~\"${p}\"})" "temperature") ];
          unit = "celsius";
          decimals = 0;
          thresholds = [
            green
            (warnAbove 80)
            (critAbove 95)
          ];
        })
      ];
    }
    {
      title = "Compute";
      panels = [
        (timeseries {
          title = "CPU Utilization";
          targets = [
            (target "100 * (1 - avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\",instance=~\"${p}\"}[$__rate_interval])))" "{{instance}}")
          ];
          unit = "percent";
          min = 0;
          max = 100;
        })
        (timeseries {
          title = "Load Average (1m)";
          targets = [ (target "node_load1{instance=~\"${p}\"}" "{{instance}}") ];
          min = 0;
        })
        (timeseries {
          title = "CPU Frequency";
          targets = [
            (target "avg by (instance) (node_cpu_scaling_frequency_hertz{instance=~\"${p}\"})" "{{instance}}")
          ];
          unit = "hertz";
          min = 0;
        })
        (timeseries {
          title = "Memory Usage";
          description = "Whole-system memory usage. The APU's unified GPU memory is part of system RAM; discrete V620 VRAM is shown separately.";
          targets = [
            (target "100 * (1 - node_memory_MemAvailable_bytes{instance=~\"${p}\"} / node_memory_MemTotal_bytes{instance=~\"${p}\"})" "{{instance}} system")
          ];
          unit = "percent";
          min = 0;
          max = 100;
        })
      ];
    }
    {
      title = "APU SMU";
      panels = [
        (timeseries {
          title = "Power";
          description = "Strix Halo APU power domains from the v3.0 SMU metrics table. V620 board power is shown in its own row.";
          targets = [
            (target "amd_smu_power_watts{instance=~\"${p}\",domain=~\"socket|apu|gfx|cpu_cores|ipu\"}" "{{instance}} {{domain}}")
          ];
          unit = "watt";
          min = 0;
        })
        (timeseries {
          title = "Thermals";
          description = "CPU Tctl plus the Strix Halo SMU thermal domains. V620 edge, junction and memory temperatures are shown separately.";
          targets = [
            (target "max by (instance, label) (node_hwmon_temp_celsius{instance=~\"${p}\"} * on (chip, sensor, instance) group_left (label) node_hwmon_sensor_label{instance=~\"${p}\",label=\"Tctl\"})" "{{instance}} {{label}}")
            (target "amd_smu_temperature_celsius{instance=~\"${p}\"}" "{{instance}} {{domain}}")
          ];
          unit = "celsius";
        })
        (timeseries {
          title = "Clocks";
          description = "Time-filtered Strix Halo APU clocks from the SMU table.";
          targets = [
            (target "amd_smu_clock_hertz{instance=~\"${p}\",domain=~\"gfx|soc|fclk|memory\"}" "{{instance}} {{domain}}")
          ];
          unit = "hertz";
        })
        (timeseries {
          title = "DRAM Bandwidth";
          targets = [
            (target "amd_smu_dram_bandwidth_bytes_per_second{instance=~\"${p}\"}" "{{instance}} {{direction}}")
          ];
          unit = "Bps";
          min = 0;
        })
        (timeseries {
          title = "Throttles";
          description = "Firmware PM-timer ticks per second spent in each throttle condition. Zero means the counter is not advancing.";
          targets = [
            (target "rate(amd_smu_throttle_residency_ticks_total{instance=~\"${p}\"}[$__rate_interval])" "{{instance}} {{reason}}")
          ];
          unit = "short";
          min = 0;
        })
      ];
    }
    {
      title = "Radeon Pro V620";
      panels = [
        (timeseries {
          title = "V620 Activity";
          targets = [
            (target "amd_gpu_busy_percent{device_id=\"0x73a1\",instance=~\"${p}\"}" "{{instance}} {{model}}")
          ];
          unit = "percent";
          min = 0;
          max = 100;
        })
        (timeseries {
          title = "V620 VRAM Usage";
          targets = [
            (target "100 * amd_gpu_memory_bytes{device_id=\"0x73a1\",pool=\"vram\",state=\"used\",instance=~\"${p}\"} / ignoring(state) amd_gpu_memory_bytes{device_id=\"0x73a1\",pool=\"vram\",state=\"total\",instance=~\"${p}\"}" "{{instance}} {{model}}")
          ];
          unit = "percent";
          min = 0;
          max = 100;
        })
        (timeseries {
          title = "V620 Power";
          targets = [
            (target "amd_gpu_power_watts{device_id=\"0x73a1\",instance=~\"${p}\"}" "{{instance}} {{sensor}}")
          ];
          unit = "watt";
          min = 0;
        })
        (timeseries {
          title = "V620 Thermals";
          targets = [
            (target "amd_gpu_temperature_celsius{device_id=\"0x73a1\",instance=~\"${p}\"}" "{{instance}} {{sensor}}")
          ];
          unit = "celsius";
        })
        (timeseries {
          title = "V620 Clocks";
          targets = [
            (target "amd_gpu_clock_hertz{device_id=\"0x73a1\",instance=~\"${p}\"}" "{{instance}} {{domain}}")
          ];
          unit = "hertz";
          min = 0;
        })
      ];
    }
    {
      title = "NPU";
      panels = [
        (timeseries {
          title = "NPU Clocks";
          targets = [
            (target "amd_npu_clock_hertz{instance=~\"${p}\"}" "{{instance}} {{domain}}")
          ];
          unit = "hertz";
          min = 0;
        })
        (timeseries {
          title = "NPU Occupancy";
          targets = [
            (target "amd_npu_hw_contexts{instance=~\"${p}\"}" "{{instance}} contexts")
            (target "amd_npu_columns_in_use{instance=~\"${p}\"}" "{{instance}} columns")
          ];
          unit = "short";
          min = 0;
        })
      ];
    }
    {
      title = "Fabric IO";
      panels = [
        (timeseries {
          title = "NVMe";
          targets = [
            (target "rate(node_disk_read_bytes_total{device=~\"nvme.*\",instance=~\"${p}\"}[$__rate_interval])" "{{instance}} {{device}} read")
            (target "rate(node_disk_written_bytes_total{device=~\"nvme.*\",instance=~\"${p}\"}[$__rate_interval])" "{{instance}} {{device}} write")
          ];
          unit = "Bps";
        })
        (timeseries {
          title = "InfiniBand";
          description = "RDMA throughput on the ConnectX fabric, read from the HCA port data counters. Netdev byte counters miss RDMA entirely, so this uses node_infiniband_port_data_*.";
          targets = [
            (target "8 * rate(node_infiniband_port_data_received_bytes_total{device=~\"${fabricHcaPattern}\",instance=~\"${p}\"}[$__rate_interval])" "{{instance}} {{device}} rx")
            (target "8 * rate(node_infiniband_port_data_transmitted_bytes_total{device=~\"${fabricHcaPattern}\",instance=~\"${p}\"}[$__rate_interval])" "{{instance}} {{device}} tx")
          ];
          unit = "bps";
        })
        (timeseries {
          title = "USB4";
          description = "Thunderbolt/USB4 RDMA mesh link health (count of ports in state ACTIVE). The thunderbolt-ibverbs module exposes no data counters, so byte-level throughput is not available here yet.";
          targets = [
            (target "count by (instance) (node_infiniband_state_id{device=~\"usb4_rdma.*\",instance=~\"${p}\"} == 4)" "{{instance}} active links")
          ];
          unit = "short";
          min = 0;
        })
        (timeseries {
          title = "Ethernet";
          targets = [
            (target "8 * rate(node_network_receive_bytes_total{device=~\"${lanDevicePattern}\",instance=~\"${p}\"}[$__rate_interval])" "{{instance}} {{device}} rx")
            (target "8 * rate(node_network_transmit_bytes_total{device=~\"${lanDevicePattern}\",instance=~\"${p}\"}[$__rate_interval])" "{{instance}} {{device}} tx")
          ];
          unit = "bps";
        })
      ];
    }
    {
      title = "vLLM";
      lines = [
        {
          height = 4;
          panels = [
            (stat {
              title = "Engines Up";
              targets = [ (target "sum(up{${v}})" "engines") ];
              decimals = 0;
              thresholds = [
                {
                  color = "text";
                  value = null;
                }
                {
                  color = "green";
                  value = 1;
                }
              ];
            })
            (stat {
              title = "Serving";
              targets = [
                (target "group by (model_name, instance) (vllm:num_requests_running{${v}})" "{{model_name}} on {{instance}}")
              ];
              textMode = "name";
              noValue = "none";
              colorMode = "none";
              graphMode = "none";
            })
            (stat {
              title = "Generation Rate";
              targets = [
                (target "sum(rate(vllm:generation_tokens_total{${v}}[$__rate_interval]))" "tok/s")
              ];
              unit = "short";
              decimals = 1;
              noValue = "idle";
            })
            (stat {
              title = "Requests In Flight";
              targets = [
                (target "sum(vllm:num_requests_running{${v}}) + sum(vllm:num_requests_waiting{${v}})" "requests")
              ];
              decimals = 0;
              noValue = "idle";
            })
          ];
        }
        {
          panels = [
            (vllmSeries {
              title = "Token Throughput";
              targets = [
                (target "sum by (instance, model_name) (rate(vllm:prompt_tokens_total{${v}}[$__rate_interval]))" "{{instance}} {{model_name}} prompt")
                (target "sum by (instance, model_name) (rate(vllm:generation_tokens_total{${v}}[$__rate_interval]))" "{{instance}} {{model_name}} generation")
              ];
              unit = "short";
              description = "Prompt and generation tokens per second. Gaps mean no vLLM engine was running.";
            })
            (vllmSeries {
              title = "Requests";
              targets = [
                (target "sum by (instance) (vllm:num_requests_running{${v}})" "{{instance}} running")
                (target "sum by (instance) (vllm:num_requests_waiting{${v}})" "{{instance}} waiting")
              ];
            })
            (vllmSeries {
              title = "KV Cache Usage";
              targets = [
                (target "100 * max by (instance, model_name) (vllm:kv_cache_usage_perc{${v}})" "{{instance}} {{model_name}}")
              ];
              unit = "percent";
              max = 100;
            })
          ];
        }
        {
          panels = [
            (vllmSeries {
              title = "Time To First Token";
              targets = [
                (target (quantile "0.50" "time_to_first_token_seconds") "{{instance}} {{model_name}} p50")
                (target (quantile "0.95" "time_to_first_token_seconds") "{{instance}} {{model_name}} p95")
              ];
              unit = "s";
            })
            (vllmSeries {
              title = "Inter-Token Latency";
              targets = [
                (target (quantile "0.50" "inter_token_latency_seconds") "{{instance}} {{model_name}} p50")
                (target (quantile "0.95" "inter_token_latency_seconds") "{{instance}} {{model_name}} p95")
              ];
              unit = "s";
            })
            (vllmSeries {
              title = "Request Queue Time";
              targets = [
                (target (quantile "0.95" "request_queue_time_seconds") "{{instance}} {{model_name}} p95")
              ];
              unit = "s";
              description = "p95 time requests spend queued before the engine starts them. Sustained growth means the cluster is saturated.";
            })
          ];
        }
      ];
    }
  ];
}
