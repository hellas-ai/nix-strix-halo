{ inputs, inputVersion }:

final: prev: {
  xrt = prev.callPackage ../pkgs/xrt {
    src = inputs.xrt-src;
    version = inputVersion "2.21" inputs.xrt-src;
    xdnaSrc = inputs.xdna-driver-src;
    xdnaVersion = inputVersion "1.7" inputs.xdna-driver-src;
  };

  xrt-amdxdna = final.xrt.xdna;
}
