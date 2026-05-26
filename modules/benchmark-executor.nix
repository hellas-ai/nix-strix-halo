# Shared module for machines that submit benchmark builds to remote runners.
{ config
, lib
, pkgs
, ...
}:

let
  inherit (lib)
    filter
    mapAttrsToList
    mkEnableOption
    mkIf
    mkOption
    nameValuePair
    types
    ;

  common = import ./benchmark-common.nix { inherit lib pkgs; };
  cfg = config.benchmark.executor;
  enabledBuilderEntries = filter (entry: entry.value.enable) (
    mapAttrsToList (name: value: { inherit name value; }) cfg.builders
  );
in
{
  options.benchmark.executor = {
    enable = mkEnableOption "benchmark executor remote builder configuration";

    buildersUseSubstitutes = mkOption {
      type = types.bool;
      default = true;
      description = "Whether remote benchmark builders may use configured substituters.";
    };

    builders = mkOption {
      type = types.attrsOf common.builderModule;
      default = { };
      description = "Remote benchmark runner machines available as Nix builders.";
    };
  };

  config = mkIf (cfg.enable && enabledBuilderEntries != [ ]) {
    nix = {
      distributedBuilds = true;
      buildMachines = map (entry: common.mkBuildMachine entry.value) enabledBuilderEntries;
      settings.builders-use-substitutes = cfg.buildersUseSubstitutes;
    };

    programs.ssh.knownHosts = builtins.listToAttrs (
      map (entry: nameValuePair entry.value.hostName { publicKey = entry.value.publicKey; }) (
        filter (entry: entry.value.publicKey != null) enabledBuilderEntries
      )
    );
  };
}
