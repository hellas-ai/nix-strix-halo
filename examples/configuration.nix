# Strix Halo live ISO example.
#
# Builds a USB-flashable NixOS image with the strix-halo overlay and
# tooling pre-installed on top of nixpkgs' installer CD modules. The
# configuration is intentionally generic across Strix Halo boards; the
# AXB35 EC driver is built in and loaded by default but no-ops on other
# hardware.
{
  lib,
  modulesPath,
  pkgs,
  inputs,
  ...
}:
let
  selfPackages = inputs.self.packages.${pkgs.stdenv.hostPlatform.system};
in
{
  imports = [
    "${modulesPath}/installer/cd-dvd/installation-cd-base.nix"

    inputs.self.nixosModules.default
    inputs.self.nixosModules.tuning
    inputs.self.nixosModules.ec-su-axb35
    inputs.self.nixosModules.rpc-server
    inputs.self.nixosModules.fastflowlm-server
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
    llama-cpp-rocm-gfx1151
    llama-cpp-vulkan
    selfPackages.mlx-rocm-gfx1151
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
}
