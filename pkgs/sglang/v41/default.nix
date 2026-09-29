{
  lib,
  fetchurl,
  cargo,
  openssl,
  pkg-config,
  rustPlatform,
  rustc,
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
  # The image processor and radix tree use independent Rust workspaces. This
  # lock vendors their union; cargoSetupHook must still see the original main
  # workspace lock, with its local packages and dependency resolution intact.
  cargoDeps =
    (rustPlatform.importCargoLock { lockFile = ./Cargo-vendor.lock; }).overrideAttrs
      (previous: {
        buildCommand = previous.buildCommand + ''
          rm "$out/Cargo.lock"
          tar -xOf ${src} sglang-${revision}/rust/Cargo.lock > "$out/Cargo.lock"
        '';
      });
  cargoRoot = "../rust";
  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
    cargo
    rustc
    rustPlatform.cargoSetupHook
    pkg-config
  ];
  buildInputs = (old.buildInputs or [ ]) ++ [ openssl.dev ];
  build-system = with pythonPackages; [
    setuptools
    setuptools-rust
    setuptools-scm
    torch
    wheel
  ];
  dependencies = map (p: if p == baseKernel then kernel else p) old.dependencies;
  env = (old.env or { }) // {
    SETUPTOOLS_SCM_PRETEND_VERSION = version;
    SGLANG_BUILD_RUST_EXTS = "multimodal,mem_cache";
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
