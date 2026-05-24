# Python-side extras for RDMA-enabled vLLM: rixl (ROCm NIXL port),
# lmcache (KV-cache layer on top of rixl), and cupy-rocm-7-0. Kept
# here because they're ROCm/Strix-specific; thunderbolt-ibverbs
# owns only the kernel + rdma-core layer.
_final: prev: {
  pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
    (pyfinal: _pyprev: {
      rixl = pyfinal.callPackage ../pkgs/rixl { };
      cupy-rocm-7-0 = pyfinal.callPackage ../pkgs/cupy-rocm { };
      lmcache = pyfinal.callPackage ../pkgs/lmcache {
        inherit (pyfinal) rixl;
      };
    })
  ];
}
