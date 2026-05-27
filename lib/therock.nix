{ lib, inputs }:

# TheRock overlay composition: read the source-of-truth JSON pins from
# pkgs/therock/sources/, then expose builders that turn a target record
# (from lib/rocm-targets.nix) into the three overlays the flake actually
# applies (rocm, python wheels, vllm).

let
  defaultTherockSources = {
    rocm = builtins.fromJSON (builtins.readFile ../pkgs/therock/sources/rocm.json);
    pythonWheels = builtins.fromJSON (builtins.readFile ../pkgs/therock/sources/python-wheels.json);
    rocmSourcePins = builtins.fromJSON (builtins.readFile ../pkgs/therock/sources/rocm-source.json);
    rocmSourceTrees = import ../pkgs/therock/sources/source-tree.nix { inherit inputs; };
    rocmThirdParty = builtins.fromJSON (builtins.readFile ../pkgs/therock/sources/rocm-third-party.json);
  };

  mkTherockRocmOverlay =
    {
      rocmTargets,
      target ? builtins.head rocmTargets,
      sources ? defaultTherockSources,
    }:
    import ../overlays/therock-rocm.nix {
      inherit lib rocmTargets target;
      therockRocmSources = sources.rocm;
      therockPythonWheelSources = sources.pythonWheels;
      therockRocmSourcePins = sources.rocmSourcePins;
      therockRocmSourceTrees = sources.rocmSourceTrees;
      therockRocmThirdPartySources = sources.rocmThirdParty;
    };

  mkTherockPythonOverlay =
    { target }:
    import ../overlays/therock-python.nix {
      inherit lib target;
    };

  mkTherockOverlays =
    args:
    let
      target = args.target or (builtins.head args.rocmTargets);
      rocmOverlay = mkTherockRocmOverlay args;
      pythonOverlay = mkTherockPythonOverlay { inherit target; };
      vllmOverlay = import ../overlays/therock-vllm.nix {
        inherit lib target;
        vllmSrc = inputs.vllm-src;
        vllmVersion = "0.21.0";
      };
    in
    {
      rocm = rocmOverlay;
      python = pythonOverlay;
      vllm = vllmOverlay;
    };
in
{
  inherit
    defaultTherockSources
    mkTherockRocmOverlay
    mkTherockPythonOverlay
    mkTherockOverlays
    ;
}
