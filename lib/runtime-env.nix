{ lib }:

# Runtime-env helpers shared by flake `apps`. Two primitives:
#
#   `wrapRuntimeEnv` — symlink-join a package and `makeWrapper --set`
#   every executable in `bin/` with the supplied env attrs. Returns the
#   original package untouched when `env` is empty, so it is safe to
#   call unconditionally.
#
#   `mkApp` — uniform builder for flake `apps.<name>` entries; replaces
#   the historical mkApp / mkDirectApp / mkTargetLlamaApp / mkDs4App
#   trio whose only real differences were defaulting policy and whether
#   they open-coded HSA env wrapping via writeShellScript.

{
  wrapRuntimeEnv =
    {
      pkgs,
      package,
      env ? { },
      name ? "${package.pname or package.name or "wrapped"}-env",
    }:
    if env == { } then
      package
    else
      let
        wrapperArgs = lib.concatStringsSep " " (
          lib.mapAttrsToList (
            key: value: "--set ${lib.escapeShellArg key} ${lib.escapeShellArg (toString value)}"
          ) env
        );
      in
      pkgs.symlinkJoin {
        inherit name;
        paths = [ package ];
        nativeBuildInputs = [ pkgs.makeWrapper ];
        postBuild = ''
          for bin in "$out"/bin/*; do
            [[ -L "$bin" ]] || continue
            target=$(readlink -f "$bin")
            rm "$bin"
            makeWrapper "$target" "$bin" ${wrapperArgs}
          done
        '';
        passthru = (package.passthru or { }) // {
          unwrapped = package;
        };
      };

  mkApp =
    {
      package,
      binary,
      description,
    }:
    {
      type = "app";
      program = "${package}/bin/${binary}";
      meta.description = description;
    };
}
