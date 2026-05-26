# Live USB / installation image variant of the fevm-faex9 example.
# Builds a USB-flashable ISO with the strix-halo overlay and tooling
# pre-installed. Unlike `examples/fevm-faex9`, this configuration does not
# include a host-specific disk layout or system bootloader; the installer
# image module provides those for the live system itself.
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

  # Build the AXB35 EC kernel module into the image and load it at boot.
  # The driver no-ops on non-AXB35 boards, so the live image stays bootable
  # elsewhere while exposing fans/power controls on the target hardware.
  services.ec-su-axb35.enable = lib.mkDefault true;

  networking.hostName = lib.mkForce "fevm-faex9-live";

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

  isoImage.edition = lib.mkForce "fevm-faex9";
}
