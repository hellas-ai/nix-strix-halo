{
  nix-strix-halo ? builtins.getFlake (toString ../..),
  system ? "x86_64-linux",
  extraModules ? [ ],
  specialArgs ? { },
}:
nix-strix-halo.lib.mkFevmFaex9Configuration {
  inherit
    extraModules
    specialArgs
    system
    ;
}
