source:
let
  header = builtins.readFile "${source}/mlx/version.h";
  component = name: builtins.head (builtins.match ".*#define MLX_VERSION_${name} ([0-9]+).*" header);
in
builtins.concatStringsSep "." (
  map component [
    "MAJOR"
    "MINOR"
    "PATCH"
  ]
)
