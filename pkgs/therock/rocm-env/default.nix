{
  lib,
  writeShellApplication,
  bashInteractive,
  stdenv,
  gfortran,
  zlib,
  ncurses,
  ocl-icd,
  numactl,
  therockRocmSdk,
  rdmaCore,
  hsaOverrideGfxVersion ? null,
  packageSuffix,
}:

writeShellApplication {
  name = "therock-rocm-${packageSuffix}-env";
  text = ''
    rocm=${therockRocmSdk}

    export ROCM_HOME="$rocm"
    export ROCM_PATH="$rocm"
    export HIP_PATH="$rocm"
    export HIP_PLATFORM=amd
    ${lib.optionalString (hsaOverrideGfxVersion != null) ''
      export HSA_OVERRIDE_GFX_VERSION="${hsaOverrideGfxVersion}"
    ''}

    export PATH="$rocm/bin:$rocm/llvm/bin''${PATH:+:$PATH}"

    lib_paths=(
      "$rocm/lib"
      "$rocm/lib64"
      "$rocm/lib/llvm/lib"
      "$rocm/llvm/lib"
      "${stdenv.cc.cc.lib}/lib"
      "${gfortran.cc.lib}/lib"
      "${zlib}/lib"
      "${ncurses}/lib"
      "${ocl-icd}/lib"
      "${numactl}/lib"
      "${rdmaCore}/lib"
    )

    ld_path=
    for path in "''${lib_paths[@]}"; do
      if [ -d "$path" ]; then
        if [ -n "$ld_path" ]; then
          ld_path="$ld_path:$path"
        else
          ld_path="$path"
        fi
      fi
    done
    export LD_LIBRARY_PATH="$ld_path''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    for path in \
      "$rocm/lib/llvm/amdgcn/bitcode" \
      "$rocm/amdgcn/bitcode" \
      "$rocm/lib/llvm/lib/clang"/*/amdgcn/bitcode \
      "$rocm/llvm/lib/clang"/*/amdgcn/bitcode; do
      if [ -d "$path" ]; then
        export DEVICE_LIB_PATH="$path"
        export HIP_DEVICE_LIB_PATH="$path"
        break
      fi
    done

    if [ "$#" -eq 0 ]; then
      exec ${bashInteractive}/bin/bash
    fi

    exec "$@"
  '';

  meta = with lib; {
    description = "Run a command inside the TheRock ${packageSuffix} ROCm SDK environment";
    platforms = platforms.linux;
  };
}
