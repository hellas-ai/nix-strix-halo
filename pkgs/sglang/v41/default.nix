{
  lib,
  callPackage,
  fetchurl,
  pythonPackages,
  sglang-rocm,
}:

let
  revision = "fdf14605e5791ad3bcb93b27a495d447f2746838";
  version = "0.5.20.post1.dev41308";
  src = fetchurl {
    name = "sglang-${revision}.tar.gz";
    url = "https://github.com/sgl-project/sglang/archive/${revision}.tar.gz";
    hash = "sha256-oXyEki1hlrdnqFW7ySzYZNQJiIPRGFrU41VyyzO/mAQ=";
  };
  nativeExtensions = callPackage ./native-extensions.nix {
    inherit
      src
      revision
      version
      pythonPackages
      ;
  };
  base = sglang-rocm.override {
    runtimePatches = [
      ./patches/0001-portable-mxfp4.patch
      ./patches/0002-portable-mhc.patch
      ./patches/0003-fp4-indexer.patch
      ./patches/0004-fp8-ue8m0.patch
      ./patches/0005-engram-store.patch
      ./patches/0006-engram-loader.patch
      ./patches/0007-engram-model.patch
      ./patches/0008-vision-reference-precision.patch
      ./patches/0009-bounded-weight-loading.patch
    ];
  };
  baseKernel = lib.findFirst (
    p: lib.hasPrefix "sglang-kernel-" (p.pname or "")
  ) (throw "DeepSeek V4.1 requires the native SGLang kernel package") base.dependencies;
  kernel = baseKernel.overridePythonAttrs (_: {
    pname = "sglang-kernel-v41-gfx1151";
    version = "0.4.7-${builtins.substring 0 8 revision}";
    inherit src;
    sourceRoot = "sglang-${revision}/python/sglang/kernels/aot";
  });
in
base.overridePythonAttrs (old: {
  pname = "sglang-v41-rocm-gfx1151";
  inherit version src;
  sourceRoot = "sglang-${revision}/python";
  format = "pyproject";
  postPatch = (old.postPatch or "") + ''
    # The extensions are built separately with the upstream Rust build hooks.
    substituteInPlace pyproject.toml \
      --replace-fail '"setuptools-rust>=1.11",' ""
  '';
  build-system = with pythonPackages; [
    setuptools
    setuptools-scm
    torch
    wheel
  ];
  dependencies = map (p: if p == baseKernel then kernel else p) old.dependencies;
  env = (old.env or { }) // {
    SETUPTOOLS_SCM_PRETEND_VERSION = version;
    SGLANG_BUILD_RUST_EXTS = "none";
  };
  postInstall = old.postInstall + ''
    extension_suffix="$(${pythonPackages.python.interpreter} -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
    for module_stem in ${lib.escapeShellArgs nativeExtensions.moduleStems}; do
      ln -s "${nativeExtensions}/${pythonPackages.python.sitePackages}/$module_stem$extension_suffix" \
        "$out/${pythonPackages.python.sitePackages}/$module_stem$extension_suffix"
    done
  '';
  passthru = (old.passthru or { }) // {
    inherit nativeExtensions;
  };
  pythonImportsCheck = [
    "sglang"
    "sglang.srt.configs.deepseek_v41"
    "sglang.srt.server_args"
    "sglang.srt.mem_cache.rust_tree_core.mem_cache"
    "sglang.srt.rust_extensions._multimodal"
  ];
  meta = old.meta // {
    description = "DeepSeek V4.1 SGLang candidate with native gfx1151 kernels";
  };
})
