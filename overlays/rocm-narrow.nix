# Narrows rocmPackages.clr to a single hardware target's buildTargets,
# so downstream rocm derivations build only for that arch. Applying this
# overlay invalidates downstream caches and removes support for other
# GPUs from the system OpenCL ICD — only enable on hosts that exclusively
# run that arch.
target: _final: prev: {
  rocmPackages = prev.rocmPackages.overrideScope (
    _: rocmPrev: {
      clr = rocmPrev.clr.override {
        localGpuTargets = target.buildTargets;
      };
    }
  );
}
