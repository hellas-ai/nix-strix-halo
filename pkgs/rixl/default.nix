{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  meson-python,
  meson,
  ninja,
  pkg-config,
  pybind11,
  pyyaml,
  pytest,
  setuptools,

  # runtime
  numpy,
  torch,

  # native libs
  ucx,
  cxxopts,
  liburing,
  libaio,
  rocmPackages,
  # rixl's `subprojects/*.wrap` files would normally download these at
  # build time. Provide them as system deps and pass `--wrap-mode=nofallback`
  # so meson uses pkg-config and the build stays hermetic.
  abseil-cpp,
  asio,
  prometheus-cpp,
  taskflow,

  # ROCm-aware UCX is required. Caller supplies an override via the flake
  # overlay (`ucx.override { enableRocm = true; }`); we won't try to do it
  # here so we don't accidentally rebuild stock UCX.
}:

buildPythonPackage rec {
  pname = "rixl";
  # No tagged release for the ROCm port yet — track latest tip on develop.
  # Upstream commit is from 2026-04-27. Bump to a tag once one exists.
  version = "1.1.0-unstable-2026-04-27";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "RIXL";
    rev = "39be1de82479132aff4ba20cfdab39a9d0e325df";
    hash = "sha256-QAWTavjNV9vR5xfhHlyNgqa1+Wg9UWXI4vVhy+oDtiY=";
    fetchSubmodules = true;
  };

  # nixpkgs's stdenv strips/patches RPATHs in a postFixup hook, so we
  # don't need the upstream `patchelf` PyPI build-time dep (which isn't
  # in nixpkgs anyway). Drop it from pyproject.toml so meson-python
  # doesn't fail at "getting build dependencies for wheel". Same for
  # `types-PyYAML` — used only by the dev/lint pipeline, not the build.
  # The ROCm branch pins build-time torch to 2.11.*, but this flake supplies
  # the matched ROCm torch through Nix. Keep the dependency while dropping the
  # version guard so rixl follows the pinned runtime stack.
  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace-fail '"patchelf",' "" \
      --replace-fail '"types-PyYAML",' "" \
      --replace-fail '"torch==2.11.*"' '"torch"'

    # `src/utils/common/meson.build` shells out to `git rev-parse` to
    # bake a commit-id into the binary. fetchFromGitHub strips the .git
    # directory so this fails. Replace with a static shim — we already
    # know the rev (it's pinned in this file).
    substituteInPlace src/utils/common/meson.build \
      --replace-fail \
        "git_hash = run_command('git', 'rev-parse', '--short=8', 'HEAD', check: false, capture: true)" \
        "git_hash = run_command('true', check: false, capture: true)" \
      --replace-fail \
        "git_commit = git_hash.returncode() == 0 ? git_hash.stdout().strip() : 'unknown'" \
        "git_commit = '${builtins.substring 0 8 src.rev}'"

    # Upstream rixl is pinned against asio 1.30.x but nixpkgs has 1.36.0.
    # `io_context::post()` (member function) was removed; the modern API
    # is the free function `asio::post(io_context, handler)`. One call
    # site, one-line fix.
    substituteInPlace src/plugins/ucx/ucx_backend.cpp \
      --replace-fail \
        'io_->post([&, i]()' \
        'asio::post(*io_, [&, i]()'
  '';

  build-system = [
    meson-python
    meson
    ninja
    pkg-config
    pybind11
    pyyaml
    pytest
    setuptools
    torch
  ];

  buildInputs = [
    ucx
    cxxopts
    liburing
    libaio
    rocmPackages.clr
    rocmPackages.rocm-runtime
    rocmPackages.rocm-device-libs
    abseil-cpp
    asio
    prometheus-cpp
    taskflow
  ];

  dependencies = [
    numpy
    torch
  ];

  # vLLM only uses the UCX plugin for ROCm; trim everything else so we
  # don't drag in CUDA-only / NV-only deps (GDS, GPUNETIO, MOONCAKE,
  # AZURE_BLOB, etc.). POSIX + OBJ are kept as cheap local backends so
  # rixl can fall back when running tests on a single host.
  mesonFlags = [
    "-Dbuildtype=release"
    "-Dwerror=false"
    "-Ducx_path=${ucx}"
    "-Drocm_path=${rocmPackages.clr}"
    "-Duse_rocm=true"
    "-Ddisable_gds_backend=true"
    "-Ddisable_mooncake_backend=true"
    "-Denable_plugins=UCX,POSIX,OBJ"
    "-Dbuild_tests=false"
    "-Dbuild_examples=false"
    "-Dbuild_nixl_ep=false" # requires sm_90 (Hopper); pure CDNA/RDNA build
    "-Dinstall_headers=false"
    "-Drust=false"
    # No network: refuse to download any subproject.wrap deps. We supply
    # abseil-cpp, asio, prometheus-cpp, taskflow as buildInputs and meson
    # finds them via pkg-config.
    "--wrap-mode=nofallback"
  ];

  # ROCm header in some places includes `<roctracer/roctracer_ext.h>`
  # only when RPATH happens to find it; we don't actually want roctracer
  # because we built UCX without it. Suppress the warnings.
  env.CXXFLAGS = "-Wno-error";

  pythonImportsCheck = [ "rixl" ];

  # Tests need a live RDMA fabric + accelerator + multi-process etcd; off.
  doCheck = false;

  meta = {
    description = "ROCm Inference Transfer Library — KV-cache transfer for vLLM disaggregated prefill (ROCm port of NVIDIA NIXL)";
    homepage = "https://github.com/ROCm/RIXL";
    license = with lib.licenses; [
      asl20
      mit
    ];
    platforms = lib.platforms.linux;
    sourceProvenance = with lib.sourceTypes; [ fromSource ];
  };
}
