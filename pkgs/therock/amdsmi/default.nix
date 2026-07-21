{
  lib,
  buildPythonPackage,
  python,
  wheels,
}:

buildPythonPackage {
  pname = "therock-amdsmi";
  version = wheels.version or "unstable";
  format = "other";

  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;
  doCheck = false;

  # amdsmi uses ctypes to open libamd_smi.so by basename, so unlike the
  # extension modules it cannot rely on the wheels' relative ELF RPATHs.
  env.LD_LIBRARY_PATH = wheels.passthru.rocmRuntimeEnv.LD_LIBRARY_PATH;

  installPhase = ''
    runHook preInstall

    site="$out/${python.sitePackages}"
    mkdir -p "$site"
    ln -s ${wheels}/${python.sitePackages}/_rocm_sdk_core/share/amd_smi/amdsmi "$site/amdsmi"

    runHook postInstall
  '';

  pythonImportsCheck = [
    "amdsmi"
  ];

  meta = {
    description = "Python AMD SMI bindings from the matching TheRock Python ROCm bundle";
    homepage = "https://github.com/ROCm/TheRock";
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = [ "x86_64-linux" ];
  };
}
