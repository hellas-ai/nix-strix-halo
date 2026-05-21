{
  lib,
  stdenv,
  fetchurl,
  makeWrapper,
  autoPatchelfHook,
  python312,
  python312Packages,
  bashInteractive,
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
}:

let
  basePython = python312.withPackages (
    ps: with ps; [
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
    ]
  );

  wheels = lib.mapAttrsToList (project: source: {
    filename = source.filename or "${project}.whl";
    src = fetchurl {
      inherit (source) url hash;
      name = source.filename or "${project}.whl";
    };
  }) wheelSources.packages;
in
stdenv.mkDerivation {
  pname = "therock-python-${wheelSources.target}";
  version = wheelSources.rocmVersion;

  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;

  nativeBuildInputs = [
    autoPatchelfHook
    makeWrapper
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

  installPhase = ''
    runHook preInstall

    site="$out/${python312.sitePackages}"
    wheelhouse="$TMPDIR/therock-wheels"
    mkdir -p "$site" "$out/bin" "$wheelhouse"

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

    find "$site" -path "*/_rocm_sdk_core/bin/rocgdb-py3.*" \
      ! -name "rocgdb-py3.12" -delete

    chmod -R u+w "$out"

    lib_path="$(find "$out" -type d \( -name lib -o -name lib64 \) -print | paste -sd:)"
    bin_path="$(find "$out" -type d -name bin -print | paste -sd:)"

    makeWrapper "${basePython}/bin/python" "$out/bin/therock-python" \
      --set ROCM_HOME "$site" \
      --set ROCM_PATH "$site" \
      --set HIP_PLATFORM amd \
      --set PYTHONNOUSERSITE 1 \
      --prefix PYTHONPATH : "$site" \
      --prefix PATH : "$bin_path" \
      --prefix LD_LIBRARY_PATH : "$lib_path"

    makeWrapper "$out/bin/therock-python" "$out/bin/python"

    cat > "$out/bin/therock-python-env" <<EOF
    #!${stdenv.shell}
    export ROCM_HOME="$site"
    export ROCM_PATH="$site"
    export HIP_PLATFORM=amd
    export PYTHONNOUSERSITE=1
    export PYTHONPATH="$site\''${PYTHONPATH:+:\$PYTHONPATH}"
    export PATH="$bin_path\''${PATH:+:\$PATH}"
    export LD_LIBRARY_PATH="$lib_path\''${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
    if [ "\$#" -eq 0 ]; then
      exec ${bashInteractive}/bin/bash
    fi
    exec "\$@"
    EOF
    chmod 755 "$out/bin/therock-python-env"

    runHook postInstall
  '';

  passthru = {
    inherit basePython wheelSources;
    sitePackages = "${placeholder "out"}/${python312.sitePackages}";
  };

  meta = {
    description = "TheRock pinned Python ROCm/PyTorch wheel runtime for ${wheelSources.target}";
    homepage = "https://github.com/ROCm/TheRock";
    platforms = [ "x86_64-linux" ];
  };
}
