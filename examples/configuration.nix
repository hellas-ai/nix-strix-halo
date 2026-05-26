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
{
  imports = [
    "${modulesPath}/installer/cd-dvd/installation-cd-base.nix"

    inputs.self.nixosModules.default
    inputs.self.nixosModules.tuning
    inputs.self.nixosModules.ec-su-axb35
    inputs.self.nixosModules.rpc-server
    inputs.self.nixosModules.fastflowlm-server
  ];

  # Strix Halo gfx1151 needs a recent kernel; the installer base defaults
  # to the LTS line which lags behind.
  boot.kernelPackages = lib.mkDefault pkgs.linuxPackages_latest;

  # Serial console for headless boots and the QEMU-based boot smoke test.
  # `tty1` stays the primary console for users plugged into a monitor.
  boot.kernelParams = [
    "console=tty1"
    "console=ttyS0,115200"
  ];

  # Build the AXB35 EC kernel module into the image and load it at boot.
  # The driver no-ops on non-AXB35 boards, so the live image stays bootable
  # elsewhere while exposing fans/power controls on the target hardware.
  services.ec-su-axb35.enable = lib.mkDefault true;

  networking.hostName = lib.mkForce "strix-halo-live";

  time.timeZone = lib.mkDefault "UTC";
  i18n.defaultLocale = lib.mkDefault "en_US.UTF-8";

  environment.systemPackages = with pkgs; [
    llama-cpp-rocm-gfx1151
    llama-cpp-vulkan
    fastflowlm
    ec-su-axb35-monitor
    pciutils
    usbutils
    htop
    tmux
    vim
  ];

  isoImage.edition = lib.mkForce "strix-halo";
}
