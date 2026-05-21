# Disko configuration for dual NVMe RAID0 setup
{ lib, ... }:
{
  boot.loader = {
    systemd-boot.enable = true;
    efi.canTouchEfiVariables = lib.mkForce true;
  };

  boot.swraid.mdadmConf = lib.mkDefault ''
    MAILADDR root
  '';

  disko.devices = {
    disk = {
      disk1 = {
        type = "disk";
        device = "/dev/nvme0n1";
        content = {
          type = "gpt";
          partitions = {
            "boot-1" = {
              size = "512M";
              type = "EF00";
              content = {
                type = "filesystem";
                format = "vfat";
                mountpoint = "/boot";
                mountOptions = [ "umask=0077" ];
              };
            };
            mdadm = {
              size = "100%";
              content = {
                type = "mdraid";
                name = "raid0";
              };
            };
          };
        };
      };
      disk2 = {
        type = "disk";
        device = "/dev/nvme1n1";
        content = {
          type = "gpt";
          partitions = {
            "boot-2" = {
              size = "512M";
              type = "EF00";
              content = {
                type = "filesystem";
                format = "vfat";
                mountpoint = "/boot-fallback";
                mountOptions = [ "umask=0077" ];
              };
            };
            mdadm = {
              size = "100%";
              content = {
                type = "mdraid";
                name = "raid0";
              };
            };
          };
        };
      };
    };
    mdadm = {
      raid0 = {
        type = "mdadm";
        level = 0;
        content = {
          type = "gpt";
          partitions = {
            primary = {
              size = "100%";
              content = {
                type = "filesystem";
                format = "ext4";
                mountpoint = "/";
                # Enable the large_dir feature so /nix/store/.links (which
                # accumulates millions of hardlinks under auto-optimise)
                # doesn't hit ext4's default htree depth limit. Otherwise
                # dmesg fills with `Directory ... index full, reach max
                # htree level :2 / Large directory feature is not enabled`.
                extraArgs = [
                  "-O"
                  "large_dir"
                ];
              };
            };
          };
        };
      };
    };
  };
}
