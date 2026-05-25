{
  nix-strix-halo ? builtins.getFlake (toString ../..),
  diskoModule,
  system ? "x86_64-linux",
  extraModules ? [ ],
  specialArgs ? { },
}:
nix-strix-halo.lib.mkFevmFaex9Configuration {
  inherit
    diskoModule
    extraModules
    specialArgs
    system
    ;
}
