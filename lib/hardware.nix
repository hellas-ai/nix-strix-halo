{ lib }:

let
  mkRocmTarget =
    {
      name,
      packageSuffix,
      hostnames ? [ ],
      marketingName,
      pciId ? null,
      runtimeArch,
      buildTargets ? [ runtimeArch ],
      systemFeature ? runtimeArch,
      hsaOverride ? null,
      therockTarget ? null,
      notes ? [ ],
    }:
    {
      inherit
        name
        packageSuffix
        hostnames
        marketingName
        pciId
        runtimeArch
        buildTargets
        systemFeature
        hsaOverride
        therockTarget
        notes
        ;
    };

  mkNpuTarget =
    {
      name,
      packageSuffix,
      hostnames ? [ ],
      marketingName,
      pciId ? null,
      xrtAgent,
      systemFeature ? xrtAgent,
      fastflowlm,
      notes ? [ ],
    }:
    {
      inherit
        name
        packageSuffix
        hostnames
        marketingName
        pciId
        xrtAgent
        systemFeature
        fastflowlm
        notes
        ;
    };

  rocm = {
    gfx1010 = mkRocmTarget {
      name = "trex-navi10-dgpu";
      packageSuffix = "gfx1010";
      hostnames = [ "trex" ];
      marketingName = "AMD Radeon RX 5700-class Navi 10 dGPU";
      pciId = "1002:731f";
      runtimeArch = "gfx1010";
      therockTarget = "gfx101X-dgpu";
      notes = [
        "trex was described as RX 6700 XT, but PCI and rocminfo currently report Navi 10 / gfx1010."
      ];
    };

    gfx1036 = mkRocmTarget {
      name = "fuckup-granite-ridge-igpu";
      packageSuffix = "gfx1036";
      hostnames = [ "fuckup" ];
      marketingName = "AMD Granite Ridge integrated Radeon Graphics";
      pciId = "1002:13c0";
      runtimeArch = "gfx1036";
      buildTargets = [ "gfx1030" ];
      hsaOverride = "10.3.0";
      therockTarget = "gfx103X-all";
      notes = [
        "nixpkgs ROCm defaults do not currently list gfx1036 as a first-class target; build as gfx1030 and keep gfx1036 as the scheduler feature."
      ];
    };

    gfx1103 = mkRocmTarget {
      name = "router-hawk-point-igpu";
      packageSuffix = "gfx1103";
      hostnames = [ "router" ];
      marketingName = "AMD Radeon 780M / Hawk Point integrated GPU";
      pciId = "1002:1900";
      runtimeArch = "gfx1103";
      buildTargets = [ "gfx1102" ];
      hsaOverride = "11.0.2";
      therockTarget = "gfx110X-all";
    };

    gfx1151 = mkRocmTarget {
      name = "strix-halo";
      packageSuffix = "gfx1151";
      marketingName = "AMD Radeon 8060S / Strix Halo integrated GPU";
      runtimeArch = "gfx1151";
      hsaOverride = "11.5.1";
      therockTarget = "gfx1151";
    };
  };

  npu = {
    strixHaloXdna2 = mkNpuTarget {
      name = "strix-halo-xdna2";
      packageSuffix = "xdna2";
      marketingName = "Ryzen AI XDNA2 NPU";
      xrtAgent = "aie2p";
      systemFeature = "xdna2";
      fastflowlm = {
        supported = true;
        minDriverVersion = "32.0.203.304";
      };
    };

    routerAie2 = mkNpuTarget {
      name = "router-aie2";
      packageSuffix = "aie2";
      hostnames = [ "router" ];
      marketingName = "RyzenAI-npu1";
      pciId = "1022:1502";
      xrtAgent = "aie2";
      fastflowlm = {
        supported = false;
        reason = "FastFlowLM currently ships XDNA2 kernels; router exposes the older aie2/XDNA1 NPU.";
      };
      notes = [
        "router reports XRT device RyzenAI-npu1 with firmware 1.5.5.391."
      ];
    };
  };

  values = attrs: map (name: attrs.${name}) (builtins.attrNames attrs);
in
rec {
  inherit rocm npu;

  defaultRocmTarget = rocm.gfx1151;
  defaultNpuTarget = npu.strixHaloXdna2;

  rocmTargets = values rocm;
  npuTargets = values npu;

  rocmPackageNames = map (target: target.packageSuffix) rocmTargets;
  gpuSystemFeatures = lib.unique (map (target: target.systemFeature) rocmTargets);
  npuSystemFeatures = lib.unique (map (target: target.systemFeature) npuTargets);

  byHost = {
    trex = {
      gpu = rocm.gfx1010;
    };
    fuckup = {
      gpu = rocm.gfx1036;
    };
    router = {
      gpu = rocm.gfx1103;
      npu = npu.routerAie2;
    };
  };
}
