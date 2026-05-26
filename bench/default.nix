# Default benchmark suite.
#
# Keep this path as the public entrypoint for callers that already import
# `bench/default.nix`; suite-specific data lives under `bench/suites/`.
args: import ./suites/llama-cpp.nix args
