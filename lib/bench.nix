{ lib }:

let
  stringify = value: if builtins.isBool value then if value then "1" else "0" else toString value;

  shellArgs = args: lib.concatMapStringsSep " " lib.escapeShellArg (map stringify args);

  nonNullAttrs = lib.filterAttrs (_: value: value != null);

  envExports =
    env:
    lib.concatStringsSep "\n" (
      lib.mapAttrsToList (
        name: value:
        assert lib.assertMsg (
          builtins.match "[A-Za-z_][A-Za-z0-9_]*" name != null
        ) "invalid benchmark environment variable name: ${name}";
        "export ${name}=${lib.escapeShellArg (stringify value)}"
      ) (nonNullAttrs env)
    );

  optionalValueArg =
    flag: value:
    lib.optionals (value != null) [
      flag
      (stringify value)
    ];

  normalizeArgs =
    args:
    if builtins.isList args then
      map stringify args
    else if args == "" then
      [ ]
    else
      lib.splitString " " args;

  packageName = package: package.pname or package.name;

  emptyRequirements = {
    systemFeatures = [ ];
    hostProfiles = [ ];
    sandboxPaths = [ ];
  };

  # Every benchmark is gated on a `benchmark` system feature in addition to
  # whatever per-suite features it asks for (GPU/NPU/dataset paths). Hydra
  # builders that don't advertise `benchmark` (general CI boxes like ax102,
  # trex) don't end up running model-dependent jobs that would fail on a
  # missing /models tree. Suites that already include `benchmark` (Metal,
  # cuda-smoke) stay idempotent via the `lib.unique` in mergeRequirements.
  normalizeRequirements =
    requirements:
    emptyRequirements
    // {
      systemFeatures = lib.unique (
        (requirements.systemFeatures or emptyRequirements.systemFeatures) ++ [ "benchmark" ]
      );
      hostProfiles = requirements.hostProfiles or emptyRequirements.hostProfiles;
      sandboxPaths = requirements.sandboxPaths or emptyRequirements.sandboxPaths;
    }
    // removeAttrs requirements [
      "systemFeatures"
      "hostProfiles"
      "sandboxPaths"
    ];

  mergeRequirements =
    requirements:
    let
      normalized = map normalizeRequirements requirements;
      extraAttrs = map (
        attrs:
        removeAttrs attrs [
          "systemFeatures"
          "hostProfiles"
          "sandboxPaths"
        ]
      ) normalized;
    in
    (lib.foldl' lib.recursiveUpdate { } extraAttrs)
    // {
      systemFeatures = lib.unique (lib.concatMap (attrs: attrs.systemFeatures) normalized);
      hostProfiles = lib.unique (lib.concatMap (attrs: attrs.hostProfiles) normalized);
      sandboxPaths = lib.unique (lib.concatMap (attrs: attrs.sandboxPaths) normalized);
    };

  normalizePackages =
    {
      package ? null,
      packages ? [ ],
    }:
    lib.optional (package != null) package ++ packages;

  packageNames = packages: map packageName packages;

  benchmarkMetadata =
    {
      name,
      command,
      env,
      packages,
      requirements,
      metadata,
    }:
    metadata
    // {
      inherit
        command
        env
        name
        requirements
        ;
      packages = packageNames packages;
    };

  normalizeTool =
    {
      name,
      package,
      packages ? [ ],
      executable ? name,
      backend ? name,
      env ? { },
      args ? [ ],
      requirements ? { },
      metadata ? { },
    }:
    {
      inherit
        name
        package
        backend
        executable
        packages
        metadata
        ;
      args = normalizeArgs args;
      env = nonNullAttrs env;
      requirements = mergeRequirements [ requirements ];
    };

  modelPath =
    root: components:
    let
      cleanRoot = lib.removeSuffix "/" (toString root);
      cleanComponents = map (
        component: lib.removePrefix "/" (lib.removeSuffix "/" (toString component))
      ) components;
    in
    lib.concatStringsSep "/" ([ cleanRoot ] ++ cleanComponents);

  defaultModelsRootFor =
    system: if lib.hasSuffix "-darwin" system then "/Users/Shared/models" else "/models";

  defaultModelsRoot = pkgs: defaultModelsRootFor pkgs.stdenv.hostPlatform.system;
in
rec {
  inherit
    defaultModelsRoot
    defaultModelsRootFor
    emptyRequirements
    envExports
    mergeRequirements
    modelPath
    normalizeRequirements
    normalizePackages
    normalizeTool
    packageNames
    shellArgs
    ;

  mkModel =
    {
      name,
      path,
      repo ? null,
      files ? [ ],
      description ? name,
    }:
    {
      inherit
        name
        path
        repo
        files
        description
        ;
    };

  mkBenchmark =
    {
      pkgs,
      name,
      command,
      package ? null,
      packages ? [ ],
      env ? { },
      requirements ? { },
      metadata ? { },
      meta ? { },
      description ? "Run benchmark ${name}",
    }:
    let
      normalizedEnv = nonNullAttrs env;
      normalizedCommand = map stringify command;
      normalizedPackages = normalizePackages { inherit package packages; };
      normalizedRequirements = mergeRequirements [ requirements ];
      normalizedMetadata = benchmarkMetadata {
        inherit
          name
          metadata
          ;
        command = normalizedCommand;
        env = normalizedEnv;
        requirements = normalizedRequirements;
        packages = normalizedPackages;
      };
      commandLine = shellArgs normalizedCommand;
    in
    pkgs.runCommand name
      {
        buildInputs = normalizedPackages;
        requiredSystemFeatures = normalizedRequirements.systemFeatures;
        passthru.benchmark = normalizedMetadata;
        meta = {
          inherit description;
        }
        // meta;
      }
      ''
        set -euo pipefail

        mkdir -p "$out"

        cat > "$out/metadata.json" <<'JSON'
        ${builtins.toJSON normalizedMetadata}
        JSON

        cat > "$out/command.json" <<'JSON'
        ${builtins.toJSON normalizedCommand}
        JSON

        cat > "$out/env.json" <<'JSON'
        ${builtins.toJSON normalizedEnv}
        JSON

        add_hydra_product() {
          local type="$1"
          local subtype="$2"
          local path="$3"

          if [ -s "$path" ]; then
            printf '%s %s %s\n' "$type" "$subtype" "$path" >> "$out/nix-support/hydra-build-products"
          fi
        }

        ${envExports normalizedEnv}
        set +e
        ${commandLine} > "$out/stdout.txt" 2> "$out/stderr.txt"
        status=$?
        set -e

        if [ "$status" -ne 0 ]; then
          echo "benchmark command failed with exit status $status" >&2
          echo "command: ${commandLine}" >&2

          if [ -s "$out/stdout.txt" ]; then
            echo "--- benchmark stdout ---" >&2
            sed -n '1,200p' "$out/stdout.txt" >&2
          fi

          if [ -s "$out/stderr.txt" ]; then
            echo "--- benchmark stderr ---" >&2
            sed -n '1,200p' "$out/stderr.txt" >&2
          fi

          exit "$status"
        fi

        mkdir -p "$out/nix-support"
        : > "$out/nix-support/hydra-build-products"
        add_hydra_product file benchmark-stdout "$out/stdout.txt"
        add_hydra_product file benchmark-stderr "$out/stderr.txt"
        add_hydra_product file benchmark-metadata "$out/metadata.json"
        add_hydra_product file benchmark-command "$out/command.json"
        add_hydra_product file benchmark-env "$out/env.json"
      '';

  mkLlamaCppArgs =
    {
      modelPath ? null,
      mmap ? null,
      fa ? null,
      ngl ? null,
      threads ? null,
      batch ? null,
      ubatch ? null,
      rpc ? null,
      extraArgs ? [ ],
    }:
    lib.optionals (modelPath != null) [
      "-m"
      modelPath
    ]
    ++ optionalValueArg "--mmap" mmap
    ++ optionalValueArg "-fa" fa
    ++ optionalValueArg "-ngl" ngl
    ++ optionalValueArg "-t" threads
    ++ optionalValueArg "-b" batch
    ++ optionalValueArg "-ub" ubatch
    ++ optionalValueArg "--rpc" rpc
    ++ normalizeArgs extraArgs;

  mkLlamaCppBenchmark =
    {
      pkgs,
      name,
      package,
      packages ? [ ],
      executable ? "${package}/bin/llama-bench",
      model,
      params ? { },
      extraArgs ? [ ],
      env ? { },
      requirements ? { },
      metadata ? { },
      meta ? { },
    }:
    let
      modelPath = if builtins.isAttrs model then model.path else model;
      modelMetadata =
        if builtins.isAttrs model then removeAttrs model [ "benchmarks" ] else { path = model; };
      llamaArgs = mkLlamaCppArgs (
        params
        // {
          inherit extraArgs modelPath;
        }
      );
    in
    mkBenchmark {
      inherit
        pkgs
        name
        package
        packages
        env
        requirements
        meta
        ;
      command = [ executable ] ++ llamaArgs;
      description = "Run llama.cpp benchmark ${name}";
      metadata = lib.recursiveUpdate {
        kind = "llama-cpp";
        tool = {
          name = "llama.cpp";
          executable = "llama-bench";
          package = packageName package;
        };
        model = modelMetadata;
        inherit params;
      } metadata;
    };
}
