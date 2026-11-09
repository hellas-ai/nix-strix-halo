# Minimal Grafana dashboard builders. Dashboards are plain attrsets rendered
# with builtins.toJSON; layout (gridPos, panel ids, target refIds) is computed
# from rows so panel definitions never carry positioning by hand.
{ lib }:
let
  # Panels reference the dashboard-level datasource template variable so the
  # rendered JSON works against any Prometheus-compatible datasource.
  dsRef = {
    type = "prometheus";
    uid = "\${datasource}";
  };

  datasourceVariable = {
    name = "datasource";
    label = "Data source";
    type = "datasource";
    query = "prometheus";
    current = { };
    hide = 0;
    refresh = 1;
    regex = "";
  };

  # Turn "strix-1:9100" into "strix-1" in legends.
  stripPortTransform = {
    id = "renameByRegex";
    options = {
      regex = "([^:]+):[0-9]+";
      renamePattern = "$1";
    };
  };

  refIds = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

  withRefIds = lib.imap0 (
    i: t:
    t
    // {
      refId = builtins.substring i 1 refIds;
    }
  );

  # Split 24 columns between n panels, spreading any remainder left-to-right.
  mkWidths =
    n:
    let
      base = 24 / n;
      rem = 24 - base * n;
    in
    lib.genList (i: base + (if i < rem then 1 else 0)) n;
in
rec {
  inherit dsRef datasourceVariable;

  target = expr: legendFormat: {
    inherit expr legendFormat;
    datasource = dsRef;
  };

  # Instant query in table format, for the `table` builder. legendFormat is
  # unused there; the column header comes from the organize transform.
  instantTarget =
    expr:
    (target expr "")
    // {
      instant = true;
      format = "table";
    };

  timeseries =
    {
      title,
      targets,
      unit ? "short",
      min ? 0,
      max ? null,
      decimals ? null,
      fillOpacity ? 12,
      stacking ? false,
      # Set false for intermittent sources (e.g. vLLM) so downtime renders
      # as a gap instead of a line bridging it.
      spanNulls ? true,
      legendCalcs ? [
        "mean"
        "max"
        "lastNotNull"
      ],
      stripInstancePort ? true,
      # Grafana fieldConfig.overrides, e.g. to put a series on the right axis.
      overrides ? [ ],
      description ? null,
    }:
    {
      type = "timeseries";
      inherit title;
      datasource = dsRef;
      fieldConfig = {
        defaults = {
          inherit unit;
          color.mode = "palette-classic";
          custom = {
            drawStyle = "line";
            lineInterpolation = "linear";
            lineWidth = 1;
            inherit fillOpacity;
            showPoints = "never";
            inherit spanNulls;
            stacking = {
              mode = if stacking then "normal" else "none";
              group = "A";
            };
          };
        }
        // lib.optionalAttrs (min != null) { inherit min; }
        // lib.optionalAttrs (max != null) { inherit max; }
        // lib.optionalAttrs (decimals != null) { inherit decimals; };
        inherit overrides;
      };
      options = {
        legend = {
          showLegend = true;
          displayMode = "table";
          placement = "bottom";
          calcs = legendCalcs;
        };
        tooltip = {
          mode = "multi";
          sort = "desc";
        };
      };
      targets = withRefIds targets;
      transformations = lib.optional stripInstancePort stripPortTransform;
    }
    // lib.optionalAttrs (description != null) { inherit description; };

  stat =
    {
      title,
      targets,
      unit ? "short",
      decimals ? null,
      thresholds ? [
        {
          color = "green";
          value = null;
        }
      ],
      colorMode ? "value",
      graphMode ? "area",
      # Shown when the query returns nothing (e.g. the service is down).
      noValue ? null,
      # "name" shows the series label instead of its value.
      textMode ? "auto",
      stripInstancePort ? true,
      description ? null,
    }:
    {
      type = "stat";
      inherit title;
      datasource = dsRef;
      fieldConfig = {
        defaults = {
          inherit unit;
          thresholds = {
            mode = "absolute";
            steps = thresholds;
          };
        }
        // lib.optionalAttrs (decimals != null) { inherit decimals; }
        // lib.optionalAttrs (noValue != null) { inherit noValue; };
        overrides = [ ];
      };
      options = {
        reduceOptions = {
          calcs = [ "lastNotNull" ];
          fields = "";
          values = false;
        };
        inherit colorMode graphMode textMode;
        justifyMode = "auto";
        orientation = "auto";
      };
      targets = withRefIds targets;
      transformations = lib.optional stripInstancePort stripPortTransform;
    }
    // lib.optionalAttrs (description != null) { inherit description; };

  # One-row-per-node status table. Targets must be instantTarget queries
  # aggregated down to a bare `instance` label; the builder drops the Time
  # columns, outer-joins the frames on instance, and renames the value
  # columns. Grafana display-names the joined value fields "Value",
  # "Value 2", ... in target order, so `columns` is an ordered list of
  # { from, to } mapping those names to the wanted headers.
  table =
    {
      title,
      targets,
      columns,
      overrides ? [ ],
      sortBy ? (builtins.head columns).to,
      description ? null,
    }:
    {
      type = "table";
      inherit title;
      datasource = dsRef;
      fieldConfig = {
        defaults = {
          custom = {
            align = "auto";
            cellOptions.type = "color-text";
            inspect = false;
            filterable = false;
          };
          thresholds = {
            mode = "absolute";
            steps = [
              {
                color = "green";
                value = null;
              }
            ];
          };
        };
        inherit overrides;
      };
      options = {
        showHeader = true;
        cellHeight = "sm";
        footer.show = false;
        sortBy = [
          {
            displayName = sortBy;
            desc = false;
          }
        ];
      };
      targets = withRefIds targets;
      transformations = [
        {
          id = "filterFieldsByName";
          options.exclude.names = [ "Time" ];
        }
        {
          id = "joinByField";
          options = {
            byField = "instance";
            mode = "outerTabular";
          };
        }
        {
          id = "organize";
          options = {
            excludeByName = { };
            renameByName = builtins.listToAttrs (
              map (c: {
                name = c.from;
                value = c.to;
              }) columns
            );
            indexByName = builtins.listToAttrs (
              lib.imap0 (i: c: {
                name = c.from;
                value = i;
              }) columns
            );
          };
        }
      ];
    }
    // lib.optionalAttrs (description != null) { inherit description; };

  # rows: [ { title, height ? 8, ... } ] -> flat panel list with row headers,
  # sequential ids, and computed gridPos. Each row is laid out as one or more
  # `lines` (default: a single line made from the row itself):
  # - a line with `panels` flow-layouts them across the 24 columns at the
  #   line's height;
  # - a line with `placements` ([ { panel, x, w, y ? 0, h ? line height } ])
  #   positions panels explicitly, for headers that mix a wide panel with
  #   stacked narrow ones.
  layoutRows =
    rows:
    (lib.foldl'
      (
        acc: row:
        let
          height = row.height or 8;
          lines = row.lines or [ (builtins.removeAttrs row [ "title" ]) ];
          rowPanel = {
            type = "row";
            inherit (row) title;
            collapsed = false;
            inherit (acc) id;
            panels = [ ];
            gridPos = {
              h = 1;
              w = 24;
              x = 0;
              inherit (acc) y;
            };
          };
          placeLine =
            lacc: line:
            let
              h = line.height or height;
              placed =
                if line ? placements then
                  lib.imap0 (
                    i: pl:
                    pl.panel
                    // {
                      id = lacc.id + i;
                      gridPos = {
                        h = pl.h or h;
                        inherit (pl) w x;
                        y = lacc.y + (pl.y or 0);
                      };
                    }
                  ) line.placements
                else
                  let
                    n = lib.length line.panels;
                    widths = mkWidths n;
                    offsets = lib.genList (i: lib.foldl' lib.add 0 (lib.take i widths)) n;
                  in
                  lib.imap0 (
                    i: panel:
                    panel
                    // {
                      id = lacc.id + i;
                      gridPos = {
                        h = h;
                        w = lib.elemAt widths i;
                        x = lib.elemAt offsets i;
                        y = lacc.y;
                      };
                    }
                  ) line.panels;
              n = if line ? placements then lib.length line.placements else lib.length line.panels;
            in
            {
              y = lacc.y + h;
              id = lacc.id + n;
              panels = lacc.panels ++ placed;
            };
          res = lib.foldl' placeLine {
            y = acc.y + 1;
            id = acc.id + 1;
            panels = [ ];
          } lines;
        in
        {
          y = res.y;
          id = res.id;
          panels = acc.panels ++ [ rowPanel ] ++ res.panels;
        }
      )
      {
        y = 0;
        id = 1;
        panels = [ ];
      }
      rows
    ).panels;

  dashboard =
    {
      uid,
      title,
      rows,
      description ? "",
      tags ? [ ],
      refresh ? "30s",
      timeFrom ? "now-1h",
      extraVariables ? [ ],
    }:
    {
      inherit
        uid
        title
        description
        tags
        refresh
        ;
      editable = true;
      graphTooltip = 1; # shared crosshair across panels
      schemaVersion = 39;
      time = {
        from = timeFrom;
        to = "now";
      };
      timezone = "browser";
      templating.list = [ datasourceVariable ] ++ extraVariables;
      annotations.list = [ ];
      links = [ ];
      panels = layoutRows rows;
    };
}
