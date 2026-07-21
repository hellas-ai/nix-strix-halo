{
  lib,
  stdenv,
  fetchurl,
  therockPython,
  therockPythonPackages,
  zlib,
  zstd,
  ncurses,
  numactl,
  libffi,
  openssl,
  libxml2,
  expat,
  libxcrypt,
  libdrm,
  pciutils,
  ocl-icd,
  wheelSources,
  packageNames ? null,
}:

let
  runtimeLibraries = [
    stdenv.cc.cc.lib
    zlib
    zstd
    ncurses
    numactl
    libffi
    openssl
    libxml2
    expat
    libxcrypt
    libdrm
    pciutils
    ocl-icd
  ];
  runtimeLibraryPath = lib.makeLibraryPath runtimeLibraries;
  wheelRuntimeLibraryPath =
    site:
    lib.concatStringsSep ":" (
      [ runtimeLibraryPath ]
      ++ map (path: "${site}/${path}") [
        "_rocm_sdk_core/lib"
        "_rocm_sdk_core/lib64"
        "_rocm_sdk_core/lib/host-math/lib"
        "_rocm_sdk_core/lib/llvm/lib"
        "_rocm_sdk_core/lib/rocm_sysdeps/lib"
        "_rocm_sdk_libraries/lib"
        "torch/lib"
        "torchaudio/lib"
        "triton/backends/amd/lib"
      ]
    );

  # The pin file is already a deliberately curated closure. Installing every
  # pinned wheel keeps split device packages from the unified multi-arch index
  # together with their Python frontends.
  defaultPackageNames = builtins.attrNames wheelSources.packages;
  selectedPackageNames = if packageNames == null then defaultPackageNames else packageNames;

  wheels =
    lib.mapAttrsToList
      (project: source: {
        inherit project;
        filename = source.filename or "${project}.whl";
        src = fetchurl {
          inherit (source) url hash;
          name = source.filename or "${project}.whl";
        };
      })
      (lib.filterAttrs (project: _source: lib.elem project selectedPackageNames) wheelSources.packages);
in
therockPythonPackages.buildPythonPackage (finalAttrs: {
  pname = "therock-python-wheels-${wheelSources.target}";
  version = wheelSources.rocmVersion;
  format = "other";

  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;
  doCheck = false;

  # The unified TheRock wheels are a self-contained, $ORIGIN-linked SDK.
  # patchelf 0.15 cannot grow the RPATH on several of its unusual split-load
  # ELF files without breaking their segment alignment.
  dontPatchELF = true;
  dontAutoPatchelf = true;
  dontStrip = true;

  nativeBuildInputs = [
    therockPythonPackages.pip
    therockPythonPackages.setuptools
    therockPythonPackages.wheel
  ];

  buildInputs = runtimeLibraries;

  env.LD_LIBRARY_PATH = runtimeLibraryPath;

  propagatedBuildInputs = with therockPythonPackages; [
    filelock
    fsspec
    jinja2
    markupsafe
    mpmath
    networkx
    numpy
    packaging
    pillow
    pyyaml
    setuptools_80
    sympy
    typing-extensions
  ];

  installPhase = ''
    runHook preInstall

    site="$out/${therockPython.sitePackages}"
    wheelhouse="$TMPDIR/therock-wheels"
    mkdir -p "$site" "$wheelhouse"

    ${lib.concatMapStringsSep "\n" (wheel: ''
      cp ${wheel.src} "$wheelhouse/${wheel.filename}"
    '') wheels}

    export HOME="$TMPDIR"
    python -m pip install \
      --no-index \
      --no-deps \
      --no-build-isolation \
      --disable-pip-version-check \
      --target "$site" \
      "$wheelhouse"/*

    if [ -f "$site/torchvision/_meta_registrations.py" ]; then
      sed -i \
        's#^@torch.library.register_fake("torchvision::nms")$#@register_meta("nms")#' \
        "$site/torchvision/_meta_registrations.py"
    fi

    if [ -d "$site/triton-${wheelSources.packages.triton.packageVersion}.dist-info" ]; then
      for metadata in "$site"/torch-*.dist-info/METADATA; do
        [ -e "$metadata" ] || continue
        sed -i \
          's#^Requires-Dist: triton==.*#Requires-Dist: triton==${wheelSources.packages.triton.packageVersion}#' \
          "$metadata"
      done
    fi

    find "$site" -path "*/_rocm_sdk_core/bin/rocgdb-py3.*" \
      ! -name "rocgdb-py${lib.versions.majorMinor therockPython.version}" -delete

    # Python packages commonly import torch in their build-time checks. Export
    # the split wheel SDK's library closure to those consumers just like a
    # conventional Nix ROCm package exports its compiler/runtime setup.
    mkdir -p "$out/nix-support"
    cat > "$out/nix-support/setup-hook" <<EOF
    export LD_LIBRARY_PATH="${wheelRuntimeLibraryPath "$out/${therockPython.sitePackages}"}\''${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
    EOF

    runHook postInstall
  '';

  pythonImportsCheck = [
    "torch"
    "triton"
  ];

  passthru = {
    inherit wheelSources;
    pythonModule = therockPython;
    sitePackages = "${finalAttrs.finalPackage}/${therockPython.sitePackages}";
    rocmtoolkit_joined = "${finalAttrs.finalPackage}/${therockPython.sitePackages}/_rocm_sdk_core";
    gpuTargets = [ wheelSources.target ];
    gpuTargetString = wheelSources.target;
    cudaSupport = false;
    rocmSupport = true;
    rocmRuntimeEnv.LD_LIBRARY_PATH = wheelRuntimeLibraryPath "${finalAttrs.finalPackage}/${therockPython.sitePackages}";
    inherit stdenv;
  };

  meta = {
    description = "TheRock pinned ROCm/PyTorch/Triton wheels for ${wheelSources.target}";
    homepage = "https://github.com/ROCm/TheRock";
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = [ "x86_64-linux" ];
  };
})
