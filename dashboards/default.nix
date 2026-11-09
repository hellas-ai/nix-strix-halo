# Grafana dashboards shipped with the cluster. Dashboards are authored as
# Nix (see lib.nix for the builders) and rendered to Grafana JSON here.
{ lib, pkgs }:
let
  glib = import ./lib.nix { inherit lib; };

  mkClusterView = args: import ./cluster-view.nix ({ inherit glib; } // args);

  renderJSON = dash: pkgs.writeText "${dash.uid}.json" (builtins.toJSON dash);
in
rec {
  inherit glib mkClusterView renderJSON;

  # Rendered with defaults; use clusterViewWith to override selectors.
  clusterView = renderJSON (mkClusterView { });
  clusterViewWith = args: renderJSON (mkClusterView args);

  # Fast-iteration helper: render + push straight to a live Grafana without
  # a deploy. See dashboards/push.sh for usage.
  pushScript = pkgs.writeShellApplication {
    name = "push-grafana-dashboard";
    runtimeInputs = with pkgs; [
      curl
      jq
      openssh
    ];
    text = ''
      DEFAULT_DASHBOARD=${clusterView}
    ''
    + builtins.readFile ./push.sh;
  };
}
