{
  lib,
  stdenv,
  makeWrapper,
  therockPython,
  bashInteractive,
  wheels,
  wheelSources,
}:

let
  basePython = therockPython.withPackages (
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
  runtimeLibraryPath = wheels.passthru.rocmRuntimeEnv.LD_LIBRARY_PATH;
  site = "${wheels}/${therockPython.sitePackages}";
in
stdenv.mkDerivation {
  pname = "therock-python-${wheelSources.target}";
  version = wheelSources.rocmVersion;

  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;
  dontFixup = true;

  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/bin"

    lib_path="${runtimeLibraryPath}:$(find ${lib.escapeShellArg site} -type d \( -name lib -o -name lib64 \) -print | paste -sd:)"
    bin_path="$(find ${lib.escapeShellArg site} -type d -name bin -print | paste -sd:)"

    makeWrapper "${basePython}/bin/python" "$out/bin/therock-python" \
      --set ROCM_HOME ${lib.escapeShellArg site} \
      --set ROCM_PATH ${lib.escapeShellArg site} \
      --set HIP_PLATFORM amd \
      --set PYTHONNOUSERSITE 1 \
      --prefix PYTHONPATH : ${lib.escapeShellArg site} \
      --prefix PATH : "$bin_path" \
      --prefix LD_LIBRARY_PATH : "$lib_path"

    makeWrapper "$out/bin/therock-python" "$out/bin/python"

    cat > "$out/bin/therock-python-env" <<EOF
    #!${stdenv.shell}
    export ROCM_HOME=${lib.escapeShellArg site}
    export ROCM_PATH=${lib.escapeShellArg site}
    export HIP_PLATFORM=amd
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=${lib.escapeShellArg site}\''${PYTHONPATH:+:\$PYTHONPATH}
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
    inherit basePython wheelSources wheels;
    pythonModule = therockPython;
    sitePackages = site;
    rocmRuntimeEnv.LD_LIBRARY_PATH = runtimeLibraryPath;
  };

  meta = {
    description = "TheRock pinned Python ROCm/PyTorch wheel runtime for ${wheelSources.target}";
    homepage = "https://github.com/ROCm/TheRock";
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = [ "x86_64-linux" ];
  };
}
