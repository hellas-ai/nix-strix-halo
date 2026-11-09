{
  lib,
  callPackage,
  newScope,
  stdenv,
  runCommand,
  therockSource ? null,
  therockVersion ? "unstable",
}:
let
  requireTheRockSource =
    if therockSource == null then throw "TheRock ROCm modules require therockSource" else therockSource;
  sourcePathName = sourcePath: lib.replaceStrings [ "/" "." ] [ "-" "-" ] sourcePath;
  therockSourceSlice =
    sourcePath:
    let
      copyRoots =
        if lib.hasPrefix "rocm-systems/" sourcePath then
          [
            sourcePath
            "rocm-systems/shared"
          ]
        else if lib.hasPrefix "rocm-libraries/projects/" sourcePath then
          [
            sourcePath
            "rocm-libraries/shared"
            "rocm-libraries/cmake"
          ]
        else if lib.hasPrefix "rocm-libraries/shared/" sourcePath then
          [ "rocm-libraries/shared" ]
        else
          [ sourcePath ];
      copyCommands = lib.concatMapStringsSep "\n" (
        relPath: "copy_path ${lib.escapeShellArg relPath}"
      ) copyRoots;
    in
    runCommand "therock-rocm-source-${sourcePathName sourcePath}" { } ''
      copy_path() {
        rel="$1"
        if [ ! -e "${requireTheRockSource}/$rel" ]; then
          echo "missing TheRock source path: $rel" >&2
          exit 1
        fi
        mkdir -p "$out/$(dirname "$rel")"
        cp -a --reflink=auto "${requireTheRockSource}/$rel" "$out/$rel"
      }

      mkdir -p "$out"
      ${copyCommands}
      chmod -R u+w "$out"
    '';
  fromTheRock =
    sourcePath: drv:
    let
      src = therockSourceSlice sourcePath;
    in
    drv.overrideAttrs (old: {
      version = therockVersion;
      inherit src;
      sourceRoot = "${src.name}/${sourcePath}";
      # nixpkgs's ROCm packages typically carry backport fetchpatches for
      # bugs that landed upstream after their pinned (7.2.x) tag. TheRock
      # 7.13 already has those merged, so applying them would reverse-fail.
      # Keep local path patches (Nix-only fixes that we own), drop anything
      # else (fetchpatch-style derivations).
      patches = lib.filter (p: lib.isPath p) (old.patches or [ ]);
      passthru = builtins.removeAttrs (old.passthru or { }) [ "updateScript" ] // {
        inherit sourcePath;
        sourceProvider = "TheRock";
      };
    });

  outer = lib.makeScope newScope (
    self:
    let
      inherit (self) llvm;
      origStdenv = stdenv;
      inherit (llvm) rocmClangStdenv;
    in
    {
      inherit therockSource therockVersion rocmClangStdenv;
      stdenv = rocmClangStdenv;
      # Updaters are development metadata, never part of a live derivation.
      rocmUpdateScript = _: null;
      hip-common = fromTheRock "rocm-systems/projects/hip" (self.callPackage ./hip-common { });
      llvm = lib.recurseIntoAttrs (
        callPackage ./llvm/default.nix {
          # rocm-device-libs is used for .src only
          # otherwise would cause infinite recursion
          inherit (self) rocm-device-libs therockSource therockVersion;
        }
      );
      roctracer.src = therockSourceSlice "rocm-systems/projects/roctracer";
      clr = fromTheRock "rocm-systems/projects/clr" (self.callPackage ./clr { });
      rocm-device-libs = self.callPackage ./rocm-device-libs { };
      rocminfo = fromTheRock "rocm-systems/projects/rocminfo" (
        self.callPackage ./rocminfo { stdenv = origStdenv; }
      );
      rocprofiler-register = fromTheRock "rocm-systems/projects/rocprofiler-register" (
        self.callPackage ./rocprofiler-register {
          inherit (llvm) clang;
        }
      );
      rocm-cmake = fromTheRock "base/rocm-cmake" (self.callPackage ./rocm-cmake { stdenv = origStdenv; });
      hipcc = self.callPackage ./hipcc { stdenv = origStdenv; };
      rocm-comgr = self.callPackage ./rocm-comgr { };
      rocm-core = fromTheRock "rocm-systems/projects/rocm-core" (
        self.callPackage ./rocm-core { stdenv = origStdenv; }
      );
      rocm-runtime = fromTheRock "rocm-systems/projects/rocr-runtime" (
        self.callPackage ./rocm-runtime {
          stdenv = origStdenv;
        }
      );
      inherit (self.llvm) rocm-toolchain clang openmp;
    }
  );
  scopeForArches =
    arches:
    outer.overrideScope (
      _final: prev: {
        clr = prev.clr.override {
          localGpuTargets = arches;
        };
      }
    );
in
outer
// builtins.listToAttrs (
  map (arch: {
    name = arch;
    value = scopeForArches [ arch ];
  }) outer.clr.gpuTargets
)
// {
  gfx9 = scopeForArches [
    "gfx906"
    "gfx908"
    "gfx90a"
    "gfx942"
  ];
  gfx10 = scopeForArches [
    "gfx1010"
    "gfx1030"
  ];
  gfx11 = scopeForArches [
    "gfx1100"
    "gfx1101"
    "gfx1102"
    "gfx1150"
    "gfx1151"
  ];
  gfx12 = scopeForArches [
    "gfx1200"
    "gfx1201"
  ];
}
