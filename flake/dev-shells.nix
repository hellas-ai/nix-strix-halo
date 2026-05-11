{ pkgs }:

{
  default = pkgs.mkShell {
    packages = with pkgs; [
      deadnix
      nix-fast-build
      nixfmt-tree
      statix
      (python3.withPackages (
        ps: with ps; [
          numpy
          pandas
          plotly
        ]
      ))
    ];
  };
}
