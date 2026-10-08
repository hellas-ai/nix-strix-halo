{
  lib,
  stdenv,
  autoPatchelfHook,
  cargo,
  openssl,
  pkg-config,
  rustPlatform,
  rustc,
  pythonPackages,
  src,
  revision,
  version,
}:

let
  moduleStems = [
    "sglang/srt/mem_cache/rust_tree_core/mem_cache"
    "sglang/srt/rust_extensions/_multimodal"
  ];
  torchSite = pythonPackages.torch.passthru.sitePackages;
in
pythonPackages.buildPythonPackage {
  pname = "sglang-v41-rust-extensions";
  inherit version src;
  sourceRoot = "sglang-${revision}/python";
  format = "other";
  # Vendor both Rust workspaces while keeping the main workspace lock intact.
  cargoDeps =
    (rustPlatform.importCargoLock { lockFile = ./Cargo-vendor.lock; }).overrideAttrs
      (previous: {
        buildCommand = previous.buildCommand + ''
          rm "$out/Cargo.lock"
          tar -xOf ${src} sglang-${revision}/rust/Cargo.lock > "$out/Cargo.lock"
        '';
      });
  cargoRoot = "../rust";
  nativeBuildInputs = [
    autoPatchelfHook
    cargo
    pkg-config
    rustc
    rustPlatform.cargoSetupHook
  ];
  buildInputs = [
    openssl.dev
    stdenv.cc.cc.lib
  ];
  build-system = with pythonPackages; [
    setuptools
    setuptools-rust
    setuptools-scm
    torch
    wheel
  ];
  dependencies = with pythonPackages; [
    numpy
    torch
  ];
  env = {
    SETUPTOOLS_SCM_PRETEND_VERSION = version;
    SGLANG_BUILD_RUST_EXTS = "multimodal,mem_cache";
    HIP_VISIBLE_DEVICES = "";
    ROCR_VISIBLE_DEVICES = "";
  };
  dontConfigure = true;
  buildPhase = ''
    runHook preBuild
    ${pythonPackages.python.interpreter} setup.py build_rust --inplace
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    extension_suffix="$(${pythonPackages.python.interpreter} -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
    for module_stem in ${lib.escapeShellArgs moduleStems}; do
      install -Dm755 "$module_stem$extension_suffix" \
        "$out/${pythonPackages.python.sitePackages}/$module_stem$extension_suffix"
    done
    runHook postInstall
  '';
  preFixup = ''
    addAutoPatchelfSearchPath ${lib.escapeShellArg "${torchSite}/torch/lib"}
  '';
  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    ${pythonPackages.python.interpreter} ${./tests/native-extensions.py} \
      "$out" ${lib.escapeShellArg (toString torchSite)}
    runHook postInstallCheck
  '';
  passthru = {
    inherit moduleStems revision;
    inherit (pythonPackages) torch python;
  };
  meta = {
    description = "Pinned SGLang Rust extensions built against the runtime Python and Torch";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
