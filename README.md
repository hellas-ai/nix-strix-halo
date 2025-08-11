# Nix Llama.cpp + ROCm

A Nix flake for building llama.cpp with pre-built ROCm binaries from TheRock project. This provides reproducible builds of llama.cpp with AMD GPU acceleration without requiring a system-wide ROCm installation.

## Features

- 🚀 Pre-built ROCm 7 binaries from TheRock's nightly builds
- 🎯 Support for multiple GPU targets (gfx110X, gfx1151, gfx120X)
- ⚡ ROCWMMA support for 15x faster flash attention (optional)
- 📦 All ROCm runtime libraries included
- 🔄 Automated updates via Python script
- ❄️ Fully reproducible builds with Nix

## Supported GPU Targets

- **gfx110X** - RDNA3 GPUs (RX 7900 XTX/XT/GRE, RX 7800 XT, RX 7700 XT)
- **gfx1151** - STX Halo GPUs (Ryzen AI MAX+ Pro 395)
- **gfx120X** - RDNA4 GPUs (RX 9070 XT/GRE/9070, RX 9060 XT)

## Prerequisites

- Nix with flakes enabled
- Python 3 (for updating ROCm sources)

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd nix-llamacpp-rocm
   ```

2. **Update ROCm sources (first time setup):**
   ```bash
   python3 update-rocm.py
   ```
   This downloads and hashes the latest ROCm tarballs from TheRock's S3 bucket.

3. **Build llama.cpp for your GPU:**
   ```bash
   # For RDNA3 GPUs (RX 7900 series, etc.)
   nix build .#llama-cpp-gfx110X

   # For STX Halo GPUs
   nix build .#llama-cpp-gfx1151

   # For RDNA4 GPUs (RX 9070 series)
   nix build .#llama-cpp-gfx120X
   ```

4. **Run llama-server:**
   ```bash
   ./result/bin/llama-server -m path/to/model.gguf -ngl 99
   ```

## Available Nix Packages

### Standard Llama.cpp Packages
- `llama-cpp-gfx110X` - Llama.cpp built for RDNA3 GPUs
- `llama-cpp-gfx1151` - Llama.cpp built for STX Halo GPUs  
- `llama-cpp-gfx120X` - Llama.cpp built for RDNA4 GPUs

### ROCWMMA-Optimized Packages (15x faster flash attention)
- `llama-cpp-gfx110X-rocwmma` - RDNA3 with ROCWMMA and hipBLASLt
- `llama-cpp-gfx1151-rocwmma` - STX Halo with ROCWMMA and hipBLASLt
- `llama-cpp-gfx120X-rocwmma` - RDNA4 with ROCWMMA and hipBLASLt

### ROCm Packages (used internally)
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

This updates the `rocm-sources.json` file with:
- Latest tarball URLs
- SHA256 hashes (for Nix fixed-output derivations)
- Version information
- Update timestamps

### Updating llama.cpp

To update llama.cpp to the latest version:

```bash
# Update the flake lock to get latest llama.cpp
nix flake update llama-cpp

# Or update all dependencies
nix flake update
```

## Development Shell

Enter a development environment with all required tools:

```bash
nix develop
```

This provides:
- Python 3
- CMake
- Ninja
- curl
- jq

## How It Works

1. **Update Script** (`update-rocm.py`):
   - Queries TheRock's S3 bucket for latest ROCm tarballs
   - Downloads and computes SHA256 hashes
   - Updates `rocm-sources.json` with metadata

2. **ROCm Derivations**:
   - Fixed-output derivations using hashes from JSON
   - Unpacks pre-built binaries (no compilation needed)
   - Sets proper permissions for executables and libraries

3. **Llama.cpp Derivations**:
   - Fetches llama.cpp source from GitHub
   - Patches HIP version check for compatibility
   - Builds with ROCm's clang/clang++
   - Bundles all required ROCm runtime libraries
   - Creates wrapper scripts with proper LD_LIBRARY_PATH

## File Structure

```
nix-llamacpp-rocm/
├── flake.nix           # Main Nix flake with derivations
├── update-rocm.py      # Script to update ROCm sources
├── rocm-sources.json   # ROCm tarball metadata
├── modules/            # NixOS modules
│   └── rpc-server.nix  # RPC server service module
└── README.md          # This file
```

## Troubleshooting

### ROCm sources not populated error
Run `python3 update-rocm.py` to fetch and hash the ROCm tarballs.

### Build fails with HIP version error
The flake automatically patches the HIP version check. If issues persist, check that the patch in `flake.nix` is being applied correctly.

### Missing libraries at runtime
The flake copies essential ROCm libraries to the output. If you encounter missing library errors, check the `postInstall` phase in `flake.nix`.

## Using as an Overlay

This flake provides an overlay that can be used in other Nix projects to easily access the llama.cpp ROCm packages.

### In a NixOS Configuration

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    llamacpp-rocm.url = "github:user/nix-llamacpp-rocm";
  };

  outputs = { self, nixpkgs, llamacpp-rocm, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ({ pkgs, ... }: {
          nixpkgs.overlays = [ llamacpp-rocm.overlays.default ];
          
          # Now you can use the packages
          environment.systemPackages = with pkgs; [
            llamacpp-rocm.gfx1151  # For STX Halo GPUs
            # Or: llamacpp-rocm.gfx110X for RDNA3
            # Or: llamacpp-rocm.gfx120X for RDNA4
          ];
        })
      ];
    };
  };
}
```

### In a Development Shell

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    llamacpp-rocm.url = "github:user/nix-llamacpp-rocm";
  };

  outputs = { self, nixpkgs, llamacpp-rocm, ... }: 
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ llamacpp-rocm.overlays.default ];
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.llamacpp-rocm.gfx1151
        ];
        
        shellHook = ''
          echo "Llama.cpp with ROCm is available!"
          echo "Run: llama-server -m model.gguf -ngl 99"
        '';
      };
    };
}
```

### Available Overlay Packages

When using the overlay, packages are available under `pkgs.llamacpp-rocm.*`:

#### Standard Builds
- `pkgs.llamacpp-rocm.gfx110X` - Llama.cpp for RDNA3 GPUs
- `pkgs.llamacpp-rocm.gfx1151` - Llama.cpp for STX Halo GPUs
- `pkgs.llamacpp-rocm.gfx120X` - Llama.cpp for RDNA4 GPUs

#### ROCWMMA-Optimized Builds
- `pkgs.llamacpp-rocm.gfx110X-rocwmma` - RDNA3 with ROCWMMA
- `pkgs.llamacpp-rocm.gfx1151-rocwmma` - STX Halo with ROCWMMA
- `pkgs.llamacpp-rocm.gfx120X-rocwmma` - RDNA4 with ROCWMMA

#### ROCm Toolchains (internal use)
- `pkgs.llamacpp-rocm.rocm-gfx110X` - ROCm toolchain for RDNA3
- `pkgs.llamacpp-rocm.rocm-gfx1151` - ROCm toolchain for STX Halo
- `pkgs.llamacpp-rocm.rocm-gfx120X` - ROCm toolchain for RDNA4

## NixOS Modules

This flake provides NixOS modules for running llama.cpp services.

### RPC Server Module

The RPC server module allows you to run a llama.cpp RPC server as a systemd service.

#### Basic Configuration

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    llamacpp-rocm.url = "github:user/nix-llamacpp-rocm";
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

#### Service Management

Once configured, the RPC server runs as a systemd service:

```bash
# Check service status
systemctl status llamacpp-rpc-server

# View logs
journalctl -u llamacpp-rpc-server -f

# Restart service
systemctl restart llamacpp-rpc-server
```

#### Security Notes

The module includes security hardening:
- Runs as a dedicated system user
- Uses systemd sandboxing features
- Restricts file system access (except cache directory if enabled)
- Keeps GPU device access for ROCm functionality

## Advanced Usage

### Using specific ROCm versions

Edit `update-rocm.py` to fetch specific versions instead of latest, or manually edit `rocm-sources.json` with known good URLs and hashes.

### Custom llama.cpp versions

You can use a specific llama.cpp version by overriding the input:

```bash
# Use a specific commit or tag
nix build .#llama-cpp-gfx110X --override-input llama-cpp github:ggerganov/llama.cpp/specific-commit-or-tag

# Use a local llama.cpp checkout
nix build .#llama-cpp-gfx110X --override-input llama-cpp path:/path/to/local/llama.cpp
```

### Adding new GPU targets

1. Add the target to `update-rocm.py`
2. Update `rocm-sources.json`
3. Add corresponding packages in `flake.nix`

## Credits

- [TheRock](https://github.com/ROCm/TheRock) - Pre-built ROCm binaries
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient LLM inference
- [llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm) - Inspiration for this project

## License

This project is licensed under the MIT License.