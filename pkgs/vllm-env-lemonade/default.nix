{
  lib,
  runCommand,
  makeWrapper,
  stdenv,
  python312,
  vllm-rocm-lemonade,
  packageSuffix,
  hsaOverrideGfxVersion ? "11.5.1",
  gpuArch ? packageSuffix,
}:

let
  pythonExtrasEnv = python312.withPackages (ps: [
    ps.ray
    ps.rixl
  ]);
  lemonadeSite = "${vllm-rocm-lemonade}/lib/python3.12/site-packages";
  pythonExtrasSite = "${pythonExtrasEnv}/${python312.sitePackages}";
  rocmCore = "${vllm-rocm-lemonade}/lib/python3.12/site-packages/_rocm_sdk_core";
in
runCommand "vllm-env-lemonade-${packageSuffix}"
  {
    nativeBuildInputs = [ makeWrapper ];
  }
  ''
    mkdir -p "$out/bin"

    makeWrapper "${vllm-rocm-lemonade}/bin/vllm" "$out/bin/vllm" \
      --set ROCR "${rocmCore}/lib" \
      --set RCCL_ROCR_PATH "${rocmCore}/lib" \
      --set TRITON_LIBHIP_PATH "${rocmCore}/lib/libamdhip64.so" \
      --set PYTHONPATH "${lemonadeSite}:${pythonExtrasSite}" \
      --prefix PATH : "${pythonExtrasEnv}/bin" \
      --prefix PATH : "${vllm-rocm-lemonade}/bin"

    makeWrapper "${vllm-rocm-lemonade}/bin/python3" "$out/bin/ray" \
      --add-flags "-m ray.scripts.scripts" \
      --set HSA_OVERRIDE_GFX_VERSION ${hsaOverrideGfxVersion} \
      --set HSA_NO_SCRATCH_RECLAIM 1 \
      --set HSA_ENABLE_INTERRUPT 0 \
      --set HIP_PLATFORM amd \
      --set GPU_ARCHS ${gpuArch} \
      --set PYTORCH_ROCM_ARCH ${gpuArch} \
      --set FLASH_ATTENTION_TRITON_AMD_ENABLE TRUE \
      --set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL 1 \
      --set ROCM_HOME "${rocmCore}" \
      --set ROCM_PATH "${rocmCore}" \
      --set HIP_PATH "${rocmCore}" \
      --set ROCR "${rocmCore}/lib" \
      --set RCCL_ROCR_PATH "${rocmCore}/lib" \
      --set TRITON_LIBHIP_PATH "${rocmCore}/lib/libamdhip64.so" \
      --set DEVICE_LIB_PATH "${rocmCore}/lib/llvm/amdgcn/bitcode" \
      --set HIP_DEVICE_LIB_PATH "${rocmCore}/lib/llvm/amdgcn/bitcode" \
      --set CC "${stdenv.cc}/bin/cc" \
      --set CXX "${stdenv.cc}/bin/c++" \
      --set PYTHONPATH "${lemonadeSite}:${pythonExtrasSite}" \
      --prefix PATH : "${vllm-rocm-lemonade}/bin" \
      --prefix PATH : "${lib.makeBinPath [ stdenv.cc ]}" \
      --set PYTHONNOUSERSITE true \
      --run 'export LD_PRELOAD="${vllm-rocm-lemonade}/lib/libvllm-rocm-c10-hip-compat.so''${LD_PRELOAD:+ $LD_PRELOAD}"'

    makeWrapper "${vllm-rocm-lemonade}/bin/python3" "$out/bin/python3" \
      --set PYTHONPATH "${lemonadeSite}:${pythonExtrasSite}" \
      --set PYTHONNOUSERSITE true
    makeWrapper "${vllm-rocm-lemonade}/bin/python3" "$out/bin/python" \
      --set PYTHONPATH "${lemonadeSite}:${pythonExtrasSite}" \
      --set PYTHONNOUSERSITE true
  ''
