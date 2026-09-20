{
  lib,
  pkgs,
  nix-darwin,
  module,
}:
let
  fakeVllm = pkgs.writeScriptBin "vllm" ''
    #!${pkgs.python3}/bin/python3
    import json, os, sys
    print(json.dumps({"argv": sys.argv[1:], "home": os.environ["HOME"]}))
  '';
  evaluate =
    options:
    (nix-darwin.lib.darwinSystem {
      modules = [
        module
        {
          nixpkgs.pkgs = pkgs;
          nix.enable = false;
          system.stateVersion = 6;
          services.vllm-metal = {
            enable = true;
            package = lib.mkForce fakeVllm;
            model = "model with spaces and 'quotes'";
            user = "vllm-test";
          }
          // options;
        }
      ];
    }).config;
  defaults = evaluate { };
  custom = evaluate {
    stateDirectory = "/var/lib/vllm-test";
    revision = "test-revision";
    enablePrefixCaching = true;
    enableAutoToolChoice = true;
    toolCallParser = "qwen3_coder";
    speculativeConfig = {
      method = "mtp";
      num_speculative_tokens = 1;
    };
    extraArgs = [
      "--api-key"
      "test-only key; $(false)"
    ];
  };
  failures = configuration: lib.filter (item: !item.assertion) configuration.assertions;
  service = defaults.launchd.daemons.vllm-metal;
  customService = custom.launchd.daemons.vllm-metal;
in
assert failures defaults == [ ];
assert failures custom == [ ];
assert
  failures (evaluate {
    maxNumSeqs = 2;
    speculativeConfig.method = "mtp";
  }) != [ ];
assert
  failures (evaluate {
    enableAutoToolChoice = true;
  }) != [ ];
assert
  failures (evaluate {
    gpuMemoryUtilization = 0.0;
  }) != [ ];
assert
  failures (evaluate {
    user = "root";
  }) != [ ];
assert service.serviceConfig.UserName == "vllm-test";
assert service.serviceConfig.WorkingDirectory == "/var/lib/vllm-metal";
assert service.serviceConfig.KeepAlive.SuccessfulExit == false;
assert service.environment.HF_HOME == "/var/lib/vllm-metal/huggingface";
assert service.environment.XDG_CACHE_HOME == "/var/lib/vllm-metal/cache";
assert customService.serviceConfig.StandardErrorPath == "/var/lib/vllm-test/stderr.log";
assert lib.hasInfix "install -d -m 0700 -o vllm-test /var/lib/vllm-test"
  custom.system.activationScripts.launchd.text;
pkgs.runCommand "ci-vllm-metal-module" { nativeBuildInputs = [ pkgs.python3 ]; } ''
  mkdir "$out"
  HOME=${lib.escapeShellArg service.environment.HOME} ${service.command} > "$out/default.json"
  HOME=${lib.escapeShellArg customService.environment.HOME} ${customService.command} > "$out/custom.json"
  python - "$out" <<'PY'
  import json, pathlib, sys
  root = pathlib.Path(sys.argv[1])
  default = json.loads((root / "default.json").read_text())
  custom = json.loads((root / "custom.json").read_text())
  args = default["argv"]
  assert args[:2] == ["serve", "model with spaces and 'quotes'"]
  assert args[args.index("--host") + 1] == "127.0.0.1"
  assert args[args.index("--max-model-len") + 1] == "8192"
  assert "--no-enable-prefix-caching" in args
  assert "--speculative-config" not in args
  assert default["home"] == "/var/lib/vllm-metal"
  args = custom["argv"]
  assert "--enable-prefix-caching" in args
  assert "--no-enable-prefix-caching" not in args
  assert args[args.index("--revision") + 1] == "test-revision"
  assert json.loads(args[args.index("--speculative-config") + 1]) == {"method": "mtp", "num_speculative_tokens": 1}
  assert args[-2:] == ["--api-key", "test-only key; $(false)"]
  assert custom["home"] == "/var/lib/vllm-test"
  PY
''
