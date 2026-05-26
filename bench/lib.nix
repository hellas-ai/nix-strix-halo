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

  normalizeRequirements =
    requirements:
    emptyRequirements
    // {
      systemFeatures = requirements.systemFeatures or emptyRequirements.systemFeatures;
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
in
rec {
  inherit
    emptyRequirements
    envExports
    mergeRequirements
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

        ${envExports normalizedEnv}
        ${commandLine} > "$out/stdout.txt" 2> "$out/stderr.txt"
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
