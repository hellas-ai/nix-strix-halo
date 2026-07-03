# Strix Halo live ISO example.
#
# Builds a USB-flashable NixOS image with the strix-halo overlay and
# tooling pre-installed on top of nixpkgs' installer CD modules. The
# configuration is intentionally generic across Strix Halo boards; the
# AXB35 EC driver is built in and loaded by default but no-ops on other
# hardware.
{
  config,
  lib,
  modulesPath,
  pkgs,
  inputs,
  ...
}:
{
  imports = [
    "${modulesPath}/installer/cd-dvd/installation-cd-base.nix"

    inputs.self.nixosModules.default
    inputs.self.nixosModules.tuning
    inputs.self.nixosModules.ec-su-axb35
    inputs.self.nixosModules.rpc-server
    inputs.self.nixosModules.fastflowlm
  ];

  boot = {
    # Strix Halo gfx1151 needs a recent kernel; the installer base defaults
    # to the LTS line which lags behind.
    kernelPackages = lib.mkDefault pkgs.linuxPackages_latest;

    # The Thunderbolt project kernel can outrun ZFS compatibility in nixpkgs.
    # This live image is a benchmark/bring-up image, not a ZFS installer.
    supportedFilesystems.zfs = lib.mkForce false;

    # Serial console for headless boots and the QEMU-based boot smoke test.
    # `tty1` stays the primary console for users plugged into a monitor.
    kernelParams = [
      "console=tty1"
      "console=ttyS0,115200"
    ];
  };

  # Build the AXB35 EC kernel module into the image and load it at boot.
  # The driver no-ops on non-AXB35 boards, so the live image stays bootable
  # elsewhere while exposing fans/power controls on the target hardware.
  services.ec-su-axb35.enable = lib.mkDefault true;

  # Build Thunderbolt/USB4 RDMA support into the image without claiming
  # Thunderbolt services during a generic live boot.
  hardware.thunderbolt-ibverbs = {
    enable = lib.mkDefault true;
    kernel.useProjectKernel = lib.mkDefault true;
    loadOnBoot = lib.mkDefault false;
    userspaceTools.package = pkgs.rdma-core-usb4;
  };

  networking.hostName = lib.mkForce "strix-halo-live";

  time.timeZone = lib.mkDefault "UTC";
  i18n.defaultLocale = lib.mkDefault "en_US.UTF-8";

  environment.systemPackages = with pkgs; [
    llama-cpp-rocm
    llama-cpp-vulkan
    mlx-rocm
    fastflowlm
    ec-su-axb35-monitor
    thunderbolt-ibverbs-bench-tools
    thunderbolt-ibverbs-perftest
    pciutils
    usbutils
    htop
    tmux
    vim
  ];

  isoImage.edition = lib.mkForce "strix-halo";

  # The live image closure includes large accelerator runtimes. Letting
  # mksquashfs use every Hydra core at zstd level 19 can exceed builder RAM.
  system.build.isoImage =
    let
      maxSquashfsJobs = 8;
      baseIso = pkgs.callPackage (modulesPath + "/../lib/make-iso9660-image.nix") (
        {
          inherit (config.isoImage) compressImage volumeID contents;
          isoName = "${config.image.baseName}.iso";
          bootable = config.isoImage.makeBiosBootable;
          bootImage = "/isolinux/isolinux.bin";
          syslinux = if config.isoImage.makeBiosBootable then pkgs.syslinux else null;
          squashfsContents = config.isoImage.storeContents;
          squashfsCompression = config.isoImage.squashfsCompression;
        }
        // lib.optionalAttrs (config.isoImage.makeUsbBootable && config.isoImage.makeBiosBootable) {
          usbBootable = true;
          isohybridMbrImage = "${pkgs.syslinux}/share/syslinux/isohdpfx.bin";
        }
        // lib.optionalAttrs config.isoImage.makeEfiBootable {
          efiBootable = true;
          efiBootImage = "boot/efi.img";
        }
      );
    in
    lib.mkForce (
      baseIso.overrideAttrs (old: {
        squashfsCommand =
          builtins.replaceStrings
            [ "-processors $NIX_BUILD_CORES" ]
            [ "-processors ${toString maxSquashfsJobs}" ]
            old.squashfsCommand;
      })
    );
}
