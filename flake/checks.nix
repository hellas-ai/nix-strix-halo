{
  lib,
  pkgs,
  src,
}:

let
  mkSourceCheck =
    name: nativeBuildInputs: command:
    pkgs.runCommandLocal "ci-${name}"
      {
        inherit nativeBuildInputs;
        src = lib.cleanSource src;
      }
      ''
        export HOME="$TMPDIR"
        export XDG_CACHE_HOME="$TMPDIR/cache"
        cp -r --no-preserve=mode "$src" source
        cd source
        ${command}
        touch "$out"
      '';
in
{
  format = mkSourceCheck "format" [ pkgs.nixfmt-tree ] ''
    treefmt --fail-on-change
  '';

  deadnix = mkSourceCheck "deadnix" [ pkgs.deadnix ] ''
    deadnix --fail .
  '';

  statix = mkSourceCheck "statix" [ pkgs.statix ] ''
    statix check .
  '';
}
