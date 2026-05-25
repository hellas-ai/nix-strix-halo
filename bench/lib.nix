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

  normalizeRequirements =
    requirements:
    {
      systemFeatures = requirements.systemFeatures or [ ];
      hostProfiles = requirements.hostProfiles or [ ];
      sandboxPaths = requirements.sandboxPaths or [ ];
    }
    // removeAttrs requirements [
      "systemFeatures"
      "hostProfiles"
      "sandboxPaths"
    ];

  normalizeTool =
    {
      name,
      package,
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
        executable
        backend
        env
        args
        metadata
        ;
      requirements = normalizeRequirements requirements;
    };
in
rec {
  inherit
    envExports
    normalizeRequirements
    normalizeTool
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
      packages ? lib.optional (package != null) package,
      env ? { },
      requirements ? { },
      metadata ? { },
      meta ? { },
      description ? "Run benchmark ${name}",
    }:
    let
      normalizedEnv = nonNullAttrs env;
      normalizedCommand = map stringify command;
      normalizedRequirements = normalizeRequirements requirements;
      benchmarkMetadata = metadata // {
        inherit name;
        command = normalizedCommand;
        env = normalizedEnv;
        requirements = normalizedRequirements;
      };
      commandLine = shellArgs normalizedCommand;
    in
    pkgs.runCommand name
      {
        buildInputs = packages;
        requiredSystemFeatures = normalizedRequirements.systemFeatures;
        passthru.benchmark = benchmarkMetadata;
        meta = {
          inherit description;
        }
        // meta;
      }
      ''
        set -euo pipefail

        mkdir -p "$out"

        cat > "$out/metadata.json" <<'JSON'
        ${builtins.toJSON benchmarkMetadata}
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
