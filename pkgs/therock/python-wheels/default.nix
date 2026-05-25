{
  lib,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  python312,
  python312Packages,
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
  defaultPackageNames = [
    "rocm"
    "torch"
    "torchaudio"
    "triton"
    "rocm-sdk-core"
    "rocm-sdk-devel"
  ]
  ++ builtins.filter (name: lib.hasPrefix "rocm-sdk-libraries-" name) (
    builtins.attrNames wheelSources.packages
  );
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
python312Packages.buildPythonPackage (finalAttrs: {
  pname = "therock-python-wheels-${wheelSources.target}";
  version = wheelSources.rocmVersion;
  format = "other";

  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;
  doCheck = false;

  nativeBuildInputs = [
    autoPatchelfHook
    python312Packages.pip
    python312Packages.setuptools
    python312Packages.wheel
  ];

  buildInputs = [
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

  propagatedBuildInputs = with python312Packages; [
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
    setuptools
    sympy
    typing-extensions
  ];

  installPhase = ''
    runHook preInstall

    site="$out/${python312.sitePackages}"
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

    if [ -d "$site/triton-${wheelSources.packages.triton.packageVersion}.dist-info" ]; then
      for metadata in "$site"/torch-*.dist-info/METADATA; do
        [ -e "$metadata" ] || continue
        sed -i \
          's#^Requires-Dist: triton==.*#Requires-Dist: triton==${wheelSources.packages.triton.packageVersion}#' \
          "$metadata"
      done
    fi

    find "$site" -path "*/_rocm_sdk_core/bin/rocgdb-py3.*" \
      ! -name "rocgdb-py3.12" -delete

    runHook postInstall
  '';

  pythonImportsCheck = [
    "torch"
    "triton"
  ];

  passthru = {
    inherit wheelSources;
    pythonModule = python312;
    sitePackages = "${finalAttrs.finalPackage}/${python312.sitePackages}";
    rocmtoolkit_joined = "${finalAttrs.finalPackage}/${python312.sitePackages}/_rocm_sdk_core";
    gpuTargets = [ wheelSources.target ];
    gpuTargetString = wheelSources.target;
    cudaSupport = false;
    rocmSupport = true;
    inherit stdenv;
  };

  meta = {
    description = "TheRock pinned ROCm/PyTorch/Triton wheels for ${wheelSources.target}";
    homepage = "https://github.com/ROCm/TheRock";
    platforms = [ "x86_64-linux" ];
  };
})
