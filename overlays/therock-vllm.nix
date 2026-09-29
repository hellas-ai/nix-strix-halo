{
  lib,
  target,
  vllmSrc,
  vllmVersion,
  therockPythonConfig ? import ../pkgs/therock/python-config.nix { inherit lib; },
  enabled ? true,
}:
let
  s = target.packageSuffix;
in
final: prev:
let
  hasTherockVllmInputs = enabled && prev.stdenv.hostPlatform.isLinux;
  sdkBase = final."therock-rocm-${s}";
  vllmGpuTargets = target.buildTargets;
  sdk = sdkBase // {
    localGpuTargets = vllmGpuTargets;
    gpuTargets = vllmGpuTargets;
  };
  # PyTorch's LoadHIP.cmake replaces CMAKE_HIP_COMPILER with
  # $HIP_CLANG_PATH/clang++. Point that conventional name back at TheRock's
  # Nix-aware wrapper so HIP compilation retains libc/libstdc++ search paths.
  vllmHipClangPath = final.linkFarm "therock-vllm-hip-clang-${s}" [
    {
      name = "clang++";
      path = "${sdk}/bin/therock-hip-clang++";
    }
  ];
  py = final.${therockPythonConfig.packagesAttr};
  vllmSrcWithTag = vllmSrc // {
    tag = vllmSrc.tag or "v${vllmVersion}";
  };
  opentelemetrySemanticConventionsAi = py.opentelemetry-semantic-conventions-ai;
  mistralCommon = py.mistral-common.overridePythonAttrs (old: rec {
    version = "1.11.6";
    src = py.fetchPypi {
      pname = "mistral_common";
      inherit version;
      hash = "sha256-Ne1Cjg6IaAjwwEitrFaEPOd9bO4TwbVKBFrY0R53x1Y=";
    };
    dependencies = lib.unique ((old.dependencies or [ ]) ++ [ py.pycountry ]);
    propagatedBuildInputs = lib.unique ((old.propagatedBuildInputs or [ ]) ++ [ py.pycountry ]);
    pythonRelaxDeps = (old.pythonRelaxDeps or [ ]) ++ [ "numpy" ];
    # PyPI sdists do not include all fixtures needed by the upstream tests.
    doCheck = false;
  });
  xgrammar = final.callPackage ../pkgs/xgrammar-0_2.nix { pythonPackages = py; };
  tritonKernels = prev.fetchFromGitHub {
    owner = "ROCm";
    repo = "triton";
    rev = "0f380657dbf3ee86eb57558ff71df24f03b5d4e7";
    hash = "sha256-UQ+N7JJNtk9ZlleeoIhwxwtpmX9+cc2WkyrliS9j5Aw=";
  };
  withSetuptools80 =
    pkg:
    (pkg.overridePythonAttrs (old: {
      build-system = dropNamedDeps [ "setuptools" ] (old.build-system or [ ]) ++ [ py.setuptools_80 ];
      dependencies = dropNamedDeps [ "setuptools" ] (old.dependencies or [ ]) ++ [ py.setuptools_80 ];
      propagatedBuildInputs = dropNamedDeps [ "setuptools" ] (old.propagatedBuildInputs or [ ]) ++ [
        py.setuptools_80
      ];
    }));
  grpcioToolsForSetup = withSetuptools80 py.grpcio-tools;
  setuptoolsScmForSetup = withSetuptools80 py.setuptools-scm;
  setuptoolsRustForSetup = (withSetuptools80 py.setuptools-rust).overrideAttrs (_old: {
    setupHook = null;
  });
  rocmRuntimeLibraryPath = (py.torch.passthru.rocmRuntimeEnv or { }).LD_LIBRARY_PATH or "";

  therockRocmPackages = {
    clr = sdk;
    hipcc = sdk;
    rocminfo = sdk;
    rocm-device-libs = sdk;
    llvm = sdk;
    rocthrust = sdk;
    rocprim = sdk;
    hipcub = sdk;
    hipblas = sdk;
    hipblas-common = sdk;
    hipblaslt = sdk;
    hipfft = sdk;
    hipsparse = sdk;
    hiprand = sdk;
    hipsolver = sdk;
    miopen-hip = sdk;
    miopen = sdk;
    rccl = sdk;
    rocshmem = sdk;
    rocm-smi = sdk;
    hipsparselt = sdk;
    rocblas = sdk;
    rocm-comgr = sdk;
    rocfft = sdk;
    rocrand = sdk;
    rocm-runtime = sdk;
    rocsolver = sdk;
    rocsparse = sdk;
    composable_kernel = sdk;
  };

  dropVllmDependencyNames = [
    "amd-quark"
    "apache-tvm-ffi"
    "bitsandbytes"
    "conch-triton-kernels"
    "datasets"
    "fastsafetensors"
    "mistral-common"
    "mistral_common"
    "mistralai"
    "mooncake-transfer-engine-rocm"
    "opencv-python-headless"
    "outlines"
    "peft"
    "pyarrow"
    "pytest-asyncio"
    "runai-model-streamer"
    "runai_model_streamer"
    "tensorizer"
    "tilelang"
    "timm"
    "torchcodec"
    "xformers"
  ];

  dropNamedDeps =
    names: deps:
    prev.lib.filter (
      dep:
      let
        name = dep.pname or dep.name or "";
      in
      !(prev.lib.elem name names)
    ) deps;

  unsupportedFeatureReasons = {
    aiter = "AITer needs a separately packaged ROCm aiter build and upstream only enables it on MI300-class targets";
    fastsafetensors = "fastsafetensors is not packaged in this nixpkgs input";
    grpc = "smg-grpc-servicer is not packaged in this nixpkgs input";
    helion = "helion is marked broken in this nixpkgs input";
    instanttensor = "instanttensor is not packaged in this nixpkgs input";
    mooncake = "Mooncake requires a separately packaged ROCm transfer engine for KV-cache offload and disaggregated serving";
    rixl = "RIXL needs separate ROCm RIXL/UCX/RDMA packaging and is only used by KV-transfer/disaggregated serving paths";
    runai = "runai-model-streamer is not packaged in this nixpkgs input";
    tensorizer = "tensorizer is not packaged in this nixpkgs input";
    zen = "zentorch-weekly is not packaged and is a CPU optimization";
  };

  mkVllmTherock =
    {
      aiterSupport ? false,
      audioSupport ? true,
      benchSupport ? true,
      fastsafetensorsSupport ? false,
      flashinferSupport ? false,
      grpcSupport ? false,
      helionSupport ? false,
      instanttensorSupport ? false,
      mooncakeSupport ? false,
      otelSupport ? true,
      rixlSupport ? false,
      runaiSupport ? false,
      tensorizerSupport ? false,
      tritonSupport ? true,
      tritonKernelsSupport ? true,
      videoSupport ? false,
      zenSupport ? false,
    }:
    let
      featureFlags = {
        aiter = aiterSupport;
        audio = audioSupport;
        bench = benchSupport;
        fastsafetensors = fastsafetensorsSupport;
        flashinfer = flashinferSupport;
        grpc = grpcSupport;
        helion = helionSupport;
        instanttensor = instanttensorSupport;
        mooncake = mooncakeSupport;
        otel = otelSupport;
        rixl = rixlSupport;
        runai = runaiSupport;
        tensorizer = tensorizerSupport;
        triton = tritonSupport;
        triton-kernels = tritonKernelsSupport;
        video = videoSupport;
        zen = zenSupport;
      };
      unsupportedEnabledFeatures = lib.attrNames (
        lib.filterAttrs (
          name: enabled: enabled && builtins.hasAttr name unsupportedFeatureReasons
        ) featureFlags
      );
      unsupportedEnabledMessages = map (
        name: "${name}: ${unsupportedFeatureReasons.${name}}"
      ) unsupportedEnabledFeatures;
      optionalDependencies = {
        bench = [
          py.pandas
          py.matplotlib
          py.seaborn
          py.datasets
          py.scipy
          py.plotly
        ];
        audio = [
          py.av
          py.scipy
          py.soundfile
        ]
        ++ (mistralCommon.optional-dependencies.audio or [ ]);
        flashinfer = [ ];
        otel = [
          py.opentelemetry-api
          py.opentelemetry-exporter-otlp
          py.opentelemetry-sdk
          opentelemetrySemanticConventionsAi
        ];
        video = [ ];
      };
      featureDependencies =
        lib.optionals benchSupport optionalDependencies.bench
        ++ lib.optionals audioSupport optionalDependencies.audio
        ++ lib.optionals flashinferSupport optionalDependencies.flashinfer
        ++ lib.optionals otelSupport optionalDependencies.otel
        ++ lib.optionals videoSupport optionalDependencies.video;
      baseExtraDependencies = [
        py.amdsmi
        py.cloudpickle
        py.diskcache
        py.lark
        mistralCommon
        py.outlines-core
        py.pillow
        py.prometheus-client
        py.protobuf
        py.pyyaml
        py.regex
        py.requests
        setuptoolsRustForSetup
        py.six
        py.tqdm
        py.watchfiles
        xgrammar
      ];
      extraDependencies = lib.unique (baseExtraDependencies ++ featureDependencies);
    in
    assert lib.assertMsg tritonSupport "TheRock vLLM currently requires tritonSupport=true";
    assert lib.assertMsg tritonKernelsSupport
      "TheRock vLLM currently requires tritonKernelsSupport=true";
    assert lib.assertMsg otelSupport
      "TheRock vLLM currently requires otelSupport=true because upstream vLLM lists OpenTelemetry in common requirements";
    assert lib.assertMsg (
      unsupportedEnabledFeatures == [ ]
    ) "unsupported TheRock vLLM feature(s): ${lib.concatStringsSep "; " unsupportedEnabledMessages}";
    (py.vllm.override {
      rocmSupport = true;
      cudaSupport = false;
      gpuTargets = vllmGpuTargets;
      rocmPackages = therockRocmPackages;
      inherit (py) amdsmi;
    }).overridePythonAttrs
      (old: {
        version = vllmVersion;
        src = vllmSrcWithTag;
        # This overlay disables the optional Rust frontend below. Newer
        # nixpkgs adds Cargo vendoring for its own vLLM revision; inheriting
        # that fixed hash with our source both fetches unused dependencies
        # and fails the build.
        cargoDeps = null;

        patches = builtins.filter (
          patch: !(lib.hasSuffix "0006-drop-rocm-extra-reqs.patch" (toString patch))
        ) (old.patches or [ ]);

        postPatch = ''
          rm vllm/third_party/pynvml.py
          substituteInPlace tests/utils.py \
            --replace-fail \
              "from vllm.third_party.pynvml import" \
              "from pynvml import"
          substituteInPlace vllm/utils/import_utils.py \
            --replace-fail \
              "import vllm.third_party.pynvml as pynvml" \
              "import pynvml"

          substituteInPlace pyproject.toml \
            --replace-fail '"torch == 2.13.0"' '"torch"'

          substituteInPlace CMakeLists.txt \
            --replace-fail \
              'set(PYTHON_SUPPORTED_VERSIONS' \
              'set(PYTHON_SUPPORTED_VERSIONS "${therockPythonConfig.pythonVersion}"'

          # vLLM 0.25's Rust frontend is optional. Keep this local build on
          # the existing Python/C++/HIP surface until the Cargo deps are
          # vendored for Nix.
          substituteInPlace setup.py \
            --replace-fail \
              "rust_extensions=rust_extensions," \
              "rust_extensions=[],"

          # Model inspection starts a fresh Python interpreter. Upstream
          # replaces its entire environment with PYTHONPATH, which strips the
          # TheRock wheel runtime's LD_LIBRARY_PATH and makes torch fail to
          # load libstdc++ before an architecture can even be inspected.
          substituteInPlace vllm/model_executor/models/registry.py \
            --replace-fail \
              "env={'PYTHONPATH': ':'.join(sys.path)}," \
              "env={**os.environ, 'PYTHONPATH': ':'.join(sys.path)},"

        '';

        env = (old.env or { }) // {
          VLLM_VERSION_OVERRIDE = vllmVersion;
          HIP_CLANG_PATH = "${vllmHipClangPath}";
          HIP_PATH = "${sdk}";
          HIP_PLATFORM = "amd";
          TRITON_KERNELS_SRC_DIR = "${tritonKernels}/python/triton_kernels/triton_kernels";
          CMAKE_ARGS = prev.lib.concatStringsSep " " [
            ((old.env or { }).CMAKE_ARGS or "")
            "-DCMAKE_HIP_COMPILER=${sdk}/bin/therock-hip-clang++"
            "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${sdk}"
            "-DHIP_ROOT_DIR=${sdk}"
            # Pin the HIP arch so CMakeDetermineHIPCompiler.cmake doesn't
            # have to probe the wrapper for it. The target is known from
            # the package's overlay context — there's nothing to detect.
            "-DCMAKE_HIP_ARCHITECTURES=${prev.lib.concatStringsSep ";" vllmGpuTargets}"
          ];
          LD_LIBRARY_PATH = rocmRuntimeLibraryPath;
        };

        makeWrapperArgs = (old.makeWrapperArgs or [ ]) ++ [
          "--prefix LD_LIBRARY_PATH : ${rocmRuntimeLibraryPath}"
        ];

        # vllm-rocm compiles a few hundred HIP kernels through the TheRock
        # toolchain — heavy enough to need a big-parallel builder. Pair with
        # the same tag on therock-rocm-from-source so Hydra schedules both
        # on the same kind of host.
        requiredSystemFeatures = (old.requiredSystemFeatures or [ ]) ++ [ "big-parallel" ];

        nativeBuildInputs =
          lib.subtractLists [
            final.rustPlatform.cargoSetupHook
            final.cargo
            final.rustc
          ] (old.nativeBuildInputs or [ ])
          ++ [ final.pkg-config ];
        build-system =
          dropNamedDeps [
            "grpcio-tools"
            "setuptools"
            "setuptools-rust"
            "setuptools-scm"
          ] (old.build-system or [ ])
          ++ [
            grpcioToolsForSetup
            py.setuptools_80
            setuptoolsRustForSetup
            setuptoolsScmForSetup
          ];
        # rocm_smi-config.cmake (pulled in by torch's LoadHIP.cmake
        # during vllm's configure) does pkg_check_modules(libdrm REQUIRED).
        # The source-built SDK uses nixpkgs' libdrm via buildInputs and
        # doesn't re-export libdrm.pc, so surface it here. Idempotent
        # when the binary SDK already ships it under lib/rocm_sysdeps/.
        buildInputs = (old.buildInputs or [ ]) ++ [ final.libdrm.dev ];
        pythonRemoveDeps = (old.pythonRemoveDeps or [ ]) ++ dropVllmDependencyNames;
        dependencies = lib.unique (
          dropNamedDeps (dropVllmDependencyNames ++ [ "xgrammar" ]) (old.dependencies or [ ])
          ++ extraDependencies
        );
        propagatedBuildInputs = lib.unique (
          dropNamedDeps (dropVllmDependencyNames ++ [ "xgrammar" ]) (old.propagatedBuildInputs or [ ])
          ++ extraDependencies
        );
        optional-dependencies = optionalDependencies;
        pythonImportsCheck = (old.pythonImportsCheck or [ ]) ++ [
          "vllm.entrypoints.cli.main"
          "vllm.parser.harmony"
          "xgrammar.openai_tool_call_schema"
        ];
        postPythonImportsCheck = (old.postPythonImportsCheck or "") + ''
          ${py.python.interpreter} -c 'from vllm.utils.import_utils import import_triton_kernels; import_triton_kernels(); from triton_kernels.matmul_ogs import PrecisionConfig'
        '';
        passthru = (old.passthru or { }) // {
          vllmFeatureOptions = featureFlags;
          vllmUnsupportedFeatures = unsupportedFeatureReasons;
        };
        meta = (old.meta or { }) // {
          knownVulnerabilities = [ ];
          maintainers = with lib.maintainers; [ georgewhewell ];
        };
      });

  vllmTherock = lib.makeOverridable mkVllmTherock { };
in
lib.optionalAttrs hasTherockVllmInputs {
  "vllm-rocm-therock-${s}" = vllmTherock;
}
