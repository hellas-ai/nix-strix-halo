# EC-SU_AXB35 packages - embedded controller support for Sixunited AXB35-02 board
{ pkgs, ec-su-axb35-src }:

{
  # Monitor script for the embedded controller
  monitor = pkgs.stdenv.mkDerivation {
    pname = "ec-su-axb35-monitor";
    version = "unstable-2024-12-06";

    src = ec-su-axb35-src;

    nativeBuildInputs = [ pkgs.makeWrapper ];

    dontBuild = true;
    dontConfigure = true;

    installPhase = ''
      mkdir -p $out/bin
      cp scripts/su_axb35_monitor $out/bin/
      wrapProgram $out/bin/su_axb35_monitor \
        --prefix PATH : ${with pkgs; lib.makeBinPath [ bc coreutils ]}
    '';

    meta = with pkgs.lib; {
      description = "Monitor script for the embedded controller on Sixunited AXB35-02 board";
      homepage = "https://github.com/cmetz/ec-su_axb35-linux/";
      license = licenses.gpl2Only;
      platforms = platforms.linux;
      maintainers = with maintainers; [ georgewhewell ];
      mainProgram = "su_axb35_monitor";
    };
  };

  # Kernel module for the embedded controller
  kernelModule = { kernel }: pkgs.stdenv.mkDerivation {
    pname = "ec-su-axb35";
    version = "unstable-2024-12-06";

    src = ec-su-axb35-src;

    nativeBuildInputs = kernel.moduleBuildDependencies;

    makeFlags = kernel.makeFlags ++ [
      "KERNELRELEASE=${kernel.modDirVersion}"
      "KERNEL_DIR=${kernel.dev}/lib/modules/${kernel.modDirVersion}/build"
      "INSTALL_MOD_PATH=\${out}"
    ];

    preBuild = ''
      substituteInPlace Makefile \
        --replace-fail '/lib/modules/$(shell uname -r)/build' "${kernel.dev}/lib/modules/${kernel.modDirVersion}/build" \
        --replace-fail "depmod" "#depmod"
    '';

    buildPhase = ''
      runHook preBuild
      make -C ${kernel.dev}/lib/modules/${kernel.modDirVersion}/build M=$PWD modules
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      mkdir -p $out/lib/modules/${kernel.modDirVersion}/misc
      cp ec_su_axb35.ko $out/lib/modules/${kernel.modDirVersion}/misc/
      runHook postInstall
    '';

    meta = with pkgs.lib; {
      description = "Linux kernel module for embedded controller on Sixunited AXB35-02 board";
      homepage = "https://github.com/cmetz/ec-su_axb35-linux";
      license = licenses.gpl2Only;
      platforms = platforms.linux;
      maintainers = with maintainers; [ georgewhewell ];
    };
  };
}