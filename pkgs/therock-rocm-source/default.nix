{
  lib,
  stdenvNoCC,
  git,
  cacert,
  python3,
  url,
  ref,
  rev,
  hash,
  fetchArgs ? [ ],
  deepNestedSubmodules ? [ ],
  fetchDepth ? 1,
  name ? "therock-rocm-source",
}:

let
  pythonEnv = python3.withPackages (
    ps: with ps; [
      pyyaml
      tomli
    ]
  );

  fetchSources = ''
    python ./build_tools/fetch_sources.py \
      --jobs "$fetch_jobs" \
      --depth ${toString fetchDepth} \
      ${lib.escapeShellArgs fetchArgs}
  '';

  fetchDeepNestedSubmodules = lib.concatMapStringsSep "\n" (
    item:
    let
      parent = lib.escapeShellArg item.parent;
      paths = lib.escapeShellArgs item.paths;
    in
    ''
      git -C ${parent} submodule update --init --depth ${toString fetchDepth} --jobs "$fetch_jobs" -- ${paths}
    ''
  ) deepNestedSubmodules;
in
stdenvNoCC.mkDerivation {
  pname = name;
  version = builtins.substring 0 12 rev;

  nativeBuildInputs = [
    git
    cacert
    pythonEnv
  ];

  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;
  dontFixup = true;

  outputHashAlgo = "sha256";
  outputHashMode = "recursive";
  outputHash = hash;

  installPhase = ''
    runHook preInstall

    export HOME="$TMPDIR/home"
    export SSL_CERT_FILE="${cacert}/etc/ssl/certs/ca-bundle.crt"
    mkdir -p "$HOME"

    git clone --no-checkout ${lib.escapeShellArg url} source
    cd source
    git fetch --tags origin ${lib.escapeShellArg ref}
    git checkout --detach ${lib.escapeShellArg rev}

    fetch_jobs="''${NIX_BUILD_CORES:-1}"
    ${fetchSources}
    ${fetchDeepNestedSubmodules}

    find . -name .git -exec rm -rf {} +

    mkdir -p "$out"
    cp -a . "$out/"

    runHook postInstall
  '';

  meta = {
    description = "Staged TheRock ROCm source snapshot";
    homepage = "https://github.com/ROCm/TheRock";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
