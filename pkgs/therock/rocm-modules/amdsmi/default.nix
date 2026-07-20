{
  lib,
  stdenv,
  fetchFromGitHub,
  rocmUpdateScript,
  cmake,
  pkg-config,
  libdrm,
  libmnl,
  libnl,
  esmiIbLibrarySource,
  python,
  wrapPython,
  autoPatchelfHook,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "amdsmi";
  version = "7.2.2";
  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "rocm-systems";
    rev = "rocm-${finalAttrs.version}";
    sparseCheckout = [
      "projects/amdsmi"
      "shared"
    ];
    hash = "sha256-TFi+3txemvV6K827e8S3hZOd9jcj4Qzop6V9CdKrpLg=";
  };
  sourceRoot = "${finalAttrs.src.name}/projects/amdsmi";

  postPatch = ''
    substituteInPlace goamdsmi_shim/CMakeLists.txt \
      --replace-fail "amd_smi)" ${"'"}''${AMD_SMI_TARGET})'
    substituteInPlace CMakeLists.txt \
      --replace-fail "if(NOT latest_esmi_tag STREQUAL current_esmi_tag)" "if(OFF)"

    # Vendor the exact ESMI revision from TheRock's generated third-party
    # manifest so AMD SMI never runs its configure-time git clone.
    cp -rf --no-preserve=mode ${esmiIbLibrarySource} ./esmi_ib_library
    mkdir -p ./esmi_ib_library/include/asm
    cp ./include/amd_smi/impl/amd_hsmp.h ./esmi_ib_library/include/asm/amd_hsmp.h
  '';

  # ./drm-struct-redefinition-fix.patch targets include/amd_smi/impl/
  # amdgpu_drm.h, which 7.13 moved to include/libdrm/amdgpu_drm.h. The
  # drm_color_ctm_3x4 struct it was deleting is no longer present anyway.
  patches = [ ];

  nativeBuildInputs = [
    cmake
    pkg-config
    wrapPython
    autoPatchelfHook
  ];

  buildInputs = [
    libdrm
    libmnl
    libnl
  ];

  cmakeFlags = [
    # Manually define CMAKE_INSTALL_<DIR>
    # See: https://github.com/NixOS/nixpkgs/pull/197838
    "-DCMAKE_INSTALL_BINDIR=bin"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DCMAKE_INSTALL_INCLUDEDIR=include"
  ];

  postInstall = ''
    mkdir -p $out/${python.sitePackages}
    ln -s $out/share/amd_smi/amdsmi $out/${python.sitePackages}/amdsmi

    makeWrapperArgs=(--prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath [ libdrm ]})
    wrapPythonProgramsIn $out
    rm $out/bin/amd-smi
    ln -sf $out/libexec/amdsmi_cli/amdsmi_cli.py $out/bin/amd-smi
  '';

  passthru.updateScript = rocmUpdateScript { inherit finalAttrs; };

  meta = {
    description = "System management interface for AMD GPUs supported by ROCm";
    homepage = "https://github.com/ROCm/rocm-systems/tree/develop/projects/amdsmi";
    license = with lib.licenses; [ mit ];
    maintainers = with lib.maintainers; [ lovesegfault ];
    teams = [ lib.teams.rocm ];
    platforms = [ "x86_64-linux" ];
    mainProgram = "amd-smi";
  };
})
