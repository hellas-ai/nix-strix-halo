{
  lib,
  buildLinux,
  fetchFromGitHub,
  source ? fetchFromGitHub {
    owner = "multikernel";
    repo = "linux";
    # Annotated tag v7.0-mk2, released 2026-08-25.
    rev = "3bdd35b64413da0b4e089ce931bfc2e8b031cbf7";
    hash = "sha256-tMCLOHzGon+i5jws7tzaHgR5D1/2ef9bQJYegG42r74=";
  },
  ...
}@args:

buildLinux (
  args
  // {
    pname = "linux-multikernel";
    version = "7.0-mk2";
    modDirVersion = "7.0.0-mk2";

    # Override with `linux-multikernel.override { source = /path/to/linux; }`
    # for local kernel development while the public output stays reproducible.
    src = source;

    # This is a complete upstream-derived tree, not a patch against the
    # nixpkgs kernel. Keep its exact source while reusing nixpkgs' generic
    # x86_64 configuration and module packaging.
    kernelPatches = [ ];
    ignoreConfigErrors = false;

    structuredExtraConfig = with lib.kernel; {
      KEXEC = yes;
      KEXEC_FILE = yes;
      KEXEC_HANDOVER = yes;
      KEXEC_HANDOVER_ENABLE_DEFAULT = yes;
      MULTIKERNEL = yes;
      MKTTY = yes;
      SMP = yes;
      HOTPLUG_CPU = yes;
      MEMORY_HOTPLUG = yes;
      MEMORY_HOTREMOVE = yes;

      # Useful, but not required by the minimal initramfs demo: DAXFS can
      # expose a shared root image.
      DAXFS = yes;
      VSOCKETS = yes;

      # mk2's optional transport still implements the pre-7.0
      # vsock_transport.stream_allow callback signature and does not compile
      # in the release tree. Keep the core architecture buildable without a
      # downstream source patch; the device-free MKTTY demo does not use it.
      MULTIKERNEL_VSOCKETS = lib.mkForce no;

      # Kexec HandOver excludes deferred struct-page initialization. The
      # in-tree mk2 allocator selects CMA through KEXEC_HANDOVER and uses
      # alloc_contig_range() to grow its pool at runtime.
      DEFERRED_STRUCT_PAGE_INIT = lib.mkForce unset;

      RELOCATABLE = yes;
      BLK_DEV_INITRD = yes;
      DEVTMPFS = yes;
      DEVTMPFS_MOUNT = yes;
    };

    extraMeta = {
      branch = "7.0-mk2";
      description = "Linux 7.0 with Multikernel Technologies' bare-metal multi-kernel architecture";
      homepage = "https://github.com/multikernel/linux";
      license = lib.licenses.gpl2Only;
      platforms = [ "x86_64-linux" ];
      maintainers = with lib.maintainers; [ georgewhewell ];
    };

    passthru = {
      multikernel = {
        abi = 1;
        release = "v7.0-mk2";
        rev = "3bdd35b64413da0b4e089ce931bfc2e8b031cbf7";
      };
    };
  }
  // (args.argsOverride or { })
)
