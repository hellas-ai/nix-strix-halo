# Nix Llama.cpp + ROCm

A Nix flake for building llama.cpp with pre-built ROCm binaries from TheRock project. This provides reproducible builds of llama.cpp with AMD GPU acceleration without requiring a system-wide ROCm installation.

## Features

- Pre-built ROCm 7 binaries from TheRock's nightly builds
- Support for multiple GPU targets (gfx110X, gfx1151, gfx120X)
- ROCWMMA support for faster flash attention

## Supported GPU Targets

- **gfx110X** - RDNA3 GPUs (RX 7900 XTX/XT/GRE, RX 7800 XT, RX 7700 XT)
- **gfx1151** - STX Halo GPUs (Ryzen AI MAX+ Pro 395)
- **gfx120X** - RDNA4 GPUs (RX 9070 XT/GRE/9070, RX 9060 XT)

## Prerequisites

- Nix with flakes enabled
- Python 3 (for updating ROCm sources)

## Available Nix Packages

### Standard Llama.cpp Packages
- `llama-cpp-gfx110X` - Llama.cpp built for RDNA3 GPUs
- `llama-cpp-gfx1151` - Llama.cpp built for STX Halo GPUs  
- `llama-cpp-gfx120X` - Llama.cpp built for RDNA4 GPUs

### ROCWMMA-Optimized Packages (15x faster flash attention)
- `llama-cpp-gfx110X-rocwmma` - RDNA3 with ROCWMMA and hipBLASLt
- `llama-cpp-gfx1151-rocwmma` - STX Halo with ROCWMMA and hipBLASLt
- `llama-cpp-gfx120X-rocwmma` - RDNA4 with ROCWMMA and hipBLASLt

### ROCm Packages
- `rocm-gfx110X` - Pre-built ROCm for RDNA3
- `rocm-gfx1151` - Pre-built ROCm for STX Halo
- `rocm-gfx120X` - Pre-built ROCm for RDNA4

## Updating Dependencies

### Updating ROCm Sources

The ROCm binaries are fetched from TheRock's nightly builds. To update to the latest versions:

```bash
# Update all targets
python3 update-rocm.py

# Update specific targets only
python3 update-rocm.py --targets gfx110X,gfx1151

# Specify output file
python3 update-rocm.py --output rocm-sources.json
```

### Updating llama.cpp

To update llama.cpp to the latest version:

```bash
# Update the flake lock to get latest llama.cpp
nix flake update llama-cpp

# Or update all dependencies
nix flake update
```

## Using as an Overlay

This flake provides an overlay that can be used in other Nix projects to easily access the llama.cpp ROCm packages.

### In a NixOS Configuration

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    llamacpp-rocm.url = "github:hellas-ai/nix-strix-halo";
  };

  outputs = { self, nixpkgs, llamacpp-rocm, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ({ pkgs, ... }: {
          nixpkgs.overlays = [ llamacpp-rocm.overlays.default ];
          
          # Now you can use the packages
          environment.systemPackages = with pkgs; [
            llamacpp-rocm.gfx1151
          ];
        })
      ];
    };
  };
}
```

## Benchmarks

❯ cat $(nix build .\#benchmarks.llama2-7b.llama-cpp-rocm-gfx1151-rocwmma-b512-fa1 --print-out-paths)
warning: Git tree '/Users/grw/src/nix-llamacpp-rocm' is dirty
───────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       │ File: /nix/store/rxsqd36vcnmkaq73d7n1qkw584jgzia7-benchmark-llama-cpp-rocm-gfx1151-rocwmma
───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1   │ | model                          |       size |     params | backend    | ngl | n_batch | fa |            test |                  t/s |
   2   │ | ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -: | --------------: | -------------------: |
   3   │ | llama 7B Q4_K - Medium         |   3.80 GiB |     6.74 B | ROCm,RPC   |  99 |     512 |  1 |           pp512 |        821.55 ± 1.72 |
   4   │ | llama 7B Q4_K - Medium         |   3.80 GiB |     6.74 B | ROCm,RPC   |  99 |     512 |  1 |           tg128 |         33.81 ± 0.10 |
   5   │ 
   6   │ build: unknown (0)
───────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

## NixOS Modules

This flake provides NixOS modules for running llama.cpp services.

### RPC Server Module

The RPC server module allows you to run a llama.cpp RPC server as a systemd service.

#### Basic Configuration

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    llamacpp-rocm.url = "github:hellas-ai/nix-strix-halo";
  };

  outputs = { self, nixpkgs, llamacpp-rocm, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        # Import the overlay
        ({ pkgs, ... }: {
          nixpkgs.overlays = [ llamacpp-rocm.overlays.default ];
        })
        
        # Import the RPC server module
        llamacpp-rocm.nixosModules.rpc-server
        
        # Configure the service
        ({ pkgs, ... }: {
          services.llamacpp-rpc-server = {
            enable = true;
            package = pkgs.llamacpp-rocm.gfx1151;  # Choose your GPU target
            threads = 32;
            host = "0.0.0.0";  # Listen on all interfaces
            port = 50052;
            memory = 8192;  # 8GB backend memory
            device = "0";  # GPU device ID
            enableCache = true;
            openFirewall = true;
          };
        })
      ];
    };
  };
}
```

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable` | bool | false | Enable the RPC server service |
| `package` | package | `pkgs.llamacpp-rocm.gfx1151` | llama.cpp package to use |
| `threads` | int | 64 | Number of CPU threads |
| `device` | null or string | null | GPU device ID (e.g., "0") |
| `host` | string | "127.0.0.1" | Host to bind to |
| `port` | int | 50052 | Port to bind to |
| `memory` | null or int | null | Backend memory size in MB |
| `enableCache` | bool | false | Enable local file cache |
| `cacheDirectory` | path | "/var/cache/llamacpp-rpc" | Cache directory location |
| `extraArgs` | list of strings | [] | Extra arguments to pass to rpc-server |
| `user` | string | "llamacpp-rpc" | User to run the service as |
| `group` | string | "llamacpp-rpc" | Group to run the service as |
| `openFirewall` | bool | false | Open firewall for the RPC port |

## Benchmarking

The flake includes a comprehensive benchmarking framework to compare different builds, models, and settings.

## Credits

- [amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) 
- [strix-halo-testing](https://github.com/lhl/strix-halo-testing/)
- [llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm)

## License

This project is licensed under the MIT License.