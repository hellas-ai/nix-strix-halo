# Bare-metal multikernel Linux

This flake packages Multikernel Technologies' first public release as four
reproducible pieces:

- `linux-multikernel`: exact upstream `v7.0-mk2`, commit
  `3bdd35b64413da0b4e089ce931bfc2e8b031cbf7`;
- `kerf-multikernel`: Kerf `v0.2.0`, the resource and lifecycle manager;
- `kerf-init`: a static PID 1 for spawn initramfs images;
- `multikernel-demo-initrd`: a static BusyBox root with an interactive MKTTY
  console and independent-kernel heartbeat.

`v7.0-mk2` has an in-tree runtime contiguous allocator. The `lazy_cma` module
shown by older upstream instructions is neither needed nor loaded here.

## NixOS module

Import the overlay (normally through `nixosModules.default`) and the dedicated
module. Keep automatic preparation disabled until the machine's APIC topology
and recovery path have been checked.

```nix
{
  imports = [
    inputs.nix-strix-halo.nixosModules.default
    inputs.nix-strix-halo.nixosModules.multikernel
  ];

  boot.multikernel = {
    enable = true;

    # These are physical APIC IDs, not blindly copied Linux CPU numbers.
    # The example contains both SMT threads of four complete cores.
    pool = {
      cpus = "24-31";
      memory = "8GB";
      prepareAtBoot = false;
    };

    instances = {
      blue = {
        id = 1;
        cpus = "24-27";
        memory = "2GB";
      };
      red = {
        id = 2;
        cpus = "28-31";
        memory = "2GB";
      };
    };
  };
}
```

The module selects `linuxPackages_multikernel`, makes Kerf and
`multikernelctl` available, and creates one inert systemd preparation unit per
instance. `prepareAtBoot` and `startAtBoot` both default to false.

## Bring-up

Confirm the host is running the packaged kernel and that Secure Boot lockdown
is not blocking unsigned kexec images:

```console
$ uname -r
7.0.0-mk2
$ zgrep -E 'MULTIKERNEL|MKTTY|KEXEC_HANDOVER' /proc/config.gz
$ sudo multikernelctl show
```

Prepare and start each spawn explicitly:

```console
$ sudo multikernelctl prepare blue
$ sudo multikernelctl start blue
$ sudo multikernelctl prepare red
$ sudo multikernelctl start red
$ sudo multikernelctl show
$ sudo multikernelctl console blue
```

Detach from a Kerf console with `Ctrl+]` followed by `.`. A spawn prints its
kernel release, boot ID, online logical CPUs, physical APIC IDs, memory size, PID identity, and a
heartbeat. The shell and applets are entirely inside the initramfs; no disk,
NFS root, DAXFS image, or PCI device is required.

## Panic-isolation demonstration

With both `blue` and `red` active, attach to `blue` and deliberately panic it:

```console
[blue root@multikernel]# echo c > /proc/sysrq-trigger
```

SSH to the host and the `red` console should remain alive. Record their boot
IDs and heartbeats. A panic normally returns the failed instance to `loaded`,
so respawn its already-loaded pristine kernel image directly:

```console
$ sudo multikernelctl start blue
$ sudo multikernelctl console blue
```

If a wedged kernel remains `active` instead, use
`sudo multikernelctl force-stop blue` before restarting it.

This is a fault-isolation demonstration, not a hostile-guest security claim.
Spawn kernels cooperate with the host and are not protected by a hypervisor or
second-level page tables. Do not run untrusted kernels.

## Teardown and rollback

Cleanly stop and return one instance's resources:

```console
$ sudo multikernelctl stop blue
$ sudo multikernelctl unload blue
$ sudo multikernelctl delete blue
```

After all instances have been deleted, return the complete pool to the host:

```console
$ sudo multikernelctl reset-pool
```

For a netboot host, keep the previous PXE artifact GC-rooted and record its
served symlink before the first reboot. Reverting the server's per-host symlink
and rebooting is the recovery path; runtime pool operations never need to
rewrite firmware or a bootloader.

## Local kernel development

The public package defaults to a pinned GitHub source and content hash so
remote builders and binary caches see the same derivation. A local checkout can
replace only the source while preserving the package recipe:

```nix
linux-multikernel = prev.linux-multikernel.override {
  source = /mnt/Home/src/mklinux-7.0-mk2;
};
linuxPackages_multikernel = final.linuxPackagesFor final.linux-multikernel;
```

Nix flakes include only Git-tracked local-source changes. Stage new kernel files
before evaluating a flake that references the checkout.

## Sharp edges

- Upstream supports x86-64 in this release; Strix Halo is an empirical target,
  not an upstream-tested platform.
- Host and spawn kernels must come from the same source/layout. The module uses
  one package for both by default.
- The optional multikernel-vsock transport is disabled: mk2 ships a stale
  `stream_allow` callback signature that does not compile against its own 7.0
  VSOCK API. The CPU, memory, kexec, MKTTY, and DAXFS paths are unaffected.
- Pool memory allocation is best-effort against live memory and can fail after
  fragmentation. Prepare it early, but only after the configuration is proven.
- Use complete physical cores. Splitting SMT siblings across independent
  kernels defeats isolation and is not a sensible performance experiment.
- Start without PCI devices. Add exclusive device assignment only after CPU,
  memory, console, kill, respawn, and pool-return paths all pass.
