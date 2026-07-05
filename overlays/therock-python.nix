# Opt-in TheRock Python overlay. Substitute only the interpreter selected
# by pkgs/therock/python-config.nix; other Python interpreters stay on
# nixpkgs defaults.
{
  lib,
  target,
  pythonConfig ? import ../pkgs/therock/python-config.nix { inherit lib; },
}:
let
  s = target.packageSuffix;
  disablePythonChecks =
    pkg:
    pkg.overridePythonAttrs (_old: {
      doCheck = false;
      pythonImportsCheck = [ ];
    });
in
final: prev: {
  pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
    (
      _pyfinal: pyprev:
      if lib.versions.majorMinor pyprev.python.version == pythonConfig.pythonVersion then
        let
          wheels = final."therock-python-wheels-${s}";
          rocmSdk = final."therock-rocm-${s}";
          rocmSource = final."therock-rocm-source-${s}";
          therockRocmPackages =
            let
              sdk = rocmSdk;
            in
            {
              clr = sdk;
              hipblas = sdk;
              hipblas-common = sdk;
              hipblaslt = sdk;
              hipcub = sdk;
              hipfft = sdk;
              hipsolver = sdk;
              hipsparse = sdk;
              rocblas = sdk;
              rocprim = sdk;
              rocsolver = sdk;
              rocsparse = sdk;
              rocthrust = sdk;
              hipcc = sdk;
              composable_kernel = {
                src = "${rocmSource}/rocm-libraries/projects/composablekernel";
              };
            };
        in
        {
          amdsmi = final."therock-amdsmi-${s}";
          torch = wheels;
          torchvision = wheels;
          triton = wheels;
          triton-no-cuda = wheels;
          torchaudio = wheels;
          rocm = wheels;
          "rocm-sdk-core" = wheels;
          "rocm-sdk-devel" = wheels;
          "rocm-sdk-libraries-${s}" = wheels;

          "compressed-tensors" = pyprev."compressed-tensors".overridePythonAttrs (old: {
            doCheck = false;
            dependencies = (old.dependencies or [ ]) ++ [
              _pyfinal.psutil
            ];
            nativeCheckInputs = (old.nativeCheckInputs or [ ]) ++ [
              final.openssl
            ];
            propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [
              _pyfinal.psutil
            ];
          });
          depyf = disablePythonChecks pyprev.depyf;
          llguidance = disablePythonChecks pyprev.llguidance;
          pydevd = disablePythonChecks pyprev.pydevd;
          amd-aiter = _pyfinal.callPackage ../pkgs/amd-aiter {
            amd-aiter = pyprev.amd-aiter.override {
              rocmPackages = therockRocmPackages;
              inherit (_pyfinal) torch;
            };
          };
          timm = pyprev.timm.overridePythonAttrs (old: {
            nativeCheckInputs = (old.nativeCheckInputs or [ ]) ++ [
              final.openssl
            ];
            disabledTestPaths = (old.disabledTestPaths or [ ]) ++ [
              "tests/test_optim.py"
            ];
          });
          "mistral-common" = (disablePythonChecks pyprev."mistral-common").overridePythonAttrs (old: {
            pythonRelaxDeps = (old.pythonRelaxDeps or [ ]) ++ [
              "numpy"
            ];
          });
          "torch-memory-saver" =
            (pyprev."torch-memory-saver".override {
              inherit (_pyfinal) torch;
            }).overridePythonAttrs
              (old: {
                postPatch = (old.postPatch or "") + ''
                  substituteInPlace setup.py \
                    --replace-fail \
                      'self.compiler.set_executable("compiler_so", "hipcc")' \
                      'self.compiler.set_executable("compiler_so", "${rocmSdk}/bin/therock-hip-clang++")' \
                    --replace-fail \
                      'self.compiler.set_executable("compiler_cxx", "hipcc")' \
                      'self.compiler.set_executable("compiler_cxx", "${rocmSdk}/bin/therock-hip-clang++")' \
                    --replace-fail \
                      'self.compiler.set_executable("linker_so", "hipcc --shared")' \
                      'self.compiler.set_executable("linker_so", "${rocmSdk}/bin/therock-hip-clang++ --shared")'
                '';
                env = {
                  HIP_PLATFORM = "amd";
                  NIX_CFLAGS_COMPILE = "-D__HIP_PLATFORM_AMD__";
                  ROCM_HOME = "${rocmSdk}";
                  ROCM_PATH = "${rocmSdk}";
                };
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                  rocmSdk
                ];
                buildInputs = (old.buildInputs or [ ]) ++ [
                  rocmSdk
                ];
                meta = (old.meta or { }) // {
                  broken = false;
                };
              });
        }
      else
        { }
    )
  ];
}
