final: prev:

prev.lib.optionalAttrs prev.stdenv.isLinux (
  let
    rdtsc = final.python3Packages.callPackage ../pkgs/multikernel/rdtsc.nix { };
    kerf = final.python3Packages.callPackage ../pkgs/multikernel/kerf.nix {
      inherit rdtsc;
    };
    kerfInit = final.pkgsStatic.callPackage ../pkgs/multikernel/kerf-init.nix { };
  in
  {
    linux-multikernel = final.callPackage ../pkgs/multikernel/linux.nix { };
    linuxPackages_multikernel = final.linuxPackagesFor final.linux-multikernel;

    kerf-multikernel = kerf;
    kerf-init = kerfInit;
    multikernel-demo-initrd = final.callPackage ../pkgs/multikernel/demo-initrd.nix {
      kerf-init = kerfInit;
      busybox = final.pkgsStatic.busybox;
    };
  }
)
