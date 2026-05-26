{
  lib,
  target,
  vllmSrc,
  vllmVersion,
}:
let
  s = target.packageSuffix;
in
final: prev:
let
  sdkBase = final."therock-rocm-${s}";
  vllmGpuTargets = target.buildTargets;
  sdk = sdkBase // {
    localGpuTargets = vllmGpuTargets;
    gpuTargets = vllmGpuTargets;
  };
  py = final.python312Packages;
  vllmSrcWithTag = vllmSrc // {
    tag = vllmSrc.tag or "v${vllmVersion}";
  };
  opentelemetrySemanticConventionsAi =
    py.callPackage ../pkgs/opentelemetry-semantic-conventions-ai
      { };
  mistralCommon = py.mistral-common.overridePythonAttrs (old: rec {
    version = "1.11.2";
    src = py.fetchPypi {
      pname = "mistral_common";
      inherit version;
      hash = "sha256-efaPwtEZDyhjf0DgU/kZyMJpfgCyqmed3uViqVGD9K0=";
    };
    dependencies = lib.unique ((old.dependencies or [ ]) ++ [ py.pycountry ]);
    propagatedBuildInputs = lib.unique ((old.propagatedBuildInputs or [ ]) ++ [ py.pycountry ]);
    pythonRelaxDeps = (old.pythonRelaxDeps or [ ]) ++ [ "numpy" ];
    # PyPI sdists do not include all fixtures needed by the upstream tests.
    doCheck = false;
  });
  tritonKernels = prev.fetchFromGitHub {
    owner = "triton-lang";
    repo = "triton";
    tag = "v3.6.0";
    hash = "sha256-JFSpQn+WsNnh7CAPlcpOcUp0nyKXNbJEANdXqmkt4Tc=";
  };

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
    "bitsandbytes"
    "conch-triton-kernels"
    "datasets"
    "mistral-common"
    "mistral_common"
    "mistralai"
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
    "torchvision"
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
        py.six
        py.tqdm
        py.watchfiles
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

        patches =
          builtins.filter (patch: !(lib.hasSuffix "0006-drop-rocm-extra-reqs.patch" (toString patch))) (
            old.patches or [ ]
          )
          ++ [
            ../pkgs/vllm/patches/0001-hipify-copy-unchanged-cu-to-hip.patch
          ];

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
            --replace-fail '"torch == 2.11.0"' '"torch"' \
            --replace-fail "setuptools>=77.0.3,<81.0.0" "setuptools"

          substituteInPlace CMakeLists.txt \
            --replace-fail \
              'set(PYTHON_SUPPORTED_VERSIONS' \
              'set(PYTHON_SUPPORTED_VERSIONS "${lib.versions.majorMinor final.python312.version}"'
        '';

        env = (old.env or { }) // {
          VLLM_VERSION_OVERRIDE = vllmVersion;
          HIP_PATH = "${sdk}";
          HIP_PLATFORM = "amd";
          TRITON_KERNELS_SRC_DIR = "${tritonKernels}/python/triton_kernels/triton_kernels";
          CMAKE_ARGS = prev.lib.concatStringsSep " " [
            ((old.env or { }).CMAKE_ARGS or "")
            "-DCMAKE_HIP_COMPILER=${sdk}/bin/therock-hip-clang++"
            "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${sdk}"
            "-DHIP_ROOT_DIR=${sdk}"
          ];
        };

        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
          final.pkg-config
        ];
        pythonRemoveDeps = (old.pythonRemoveDeps or [ ]) ++ dropVllmDependencyNames;
        dependencies = lib.unique (
          dropNamedDeps dropVllmDependencyNames (old.dependencies or [ ]) ++ extraDependencies
        );
        propagatedBuildInputs = lib.unique (
          dropNamedDeps dropVllmDependencyNames (old.propagatedBuildInputs or [ ]) ++ extraDependencies
        );
        optional-dependencies = optionalDependencies;
        passthru = (old.passthru or { }) // {
          vllmFeatureOptions = featureFlags;
          vllmUnsupportedFeatures = unsupportedFeatureReasons;
        };
      });

  vllmTherock = lib.makeOverridable mkVllmTherock { };
in
{
  "vllm-rocm-therock-${s}" = vllmTherock;
}
