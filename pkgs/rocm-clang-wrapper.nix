# ROCm clang wrapper derivation - creates wrapper scripts for ROCm's clang
{ pkgs, rocmSources }:

{ target }:

let
  rocm = import ./rocm7-bin.nix { inherit pkgs rocmSources; } { inherit target; };
  
  # Create a wrapper lib directory with libgcc linker script
  gccLibWrapper = pkgs.runCommand "gcc-lib-wrapper" {} ''
    mkdir -p $out/lib
    # Create a linker script that redirects to libgcc_s
    cat > $out/lib/libgcc.so << EOF
    /* GNU ld script */
    GROUP ( ${pkgs.stdenv.cc.cc.libgcc}/lib/libgcc_s.so.1 )
    EOF
  '';
in
pkgs.stdenv.mkDerivation {
  pname = "rocm-clang-wrapper-${target}";
  version = rocm.version;
  
  dontUnpack = true;
  dontBuild = true;
  
  installPhase = ''
    mkdir -p $out/bin
    
    # Create clang wrapper
    cat > $out/bin/clang << EOF
    #!${pkgs.bash}/bin/bash
    exec ${rocm}/llvm/bin/clang \\
      -isystem ${pkgs.glibc.dev}/include \\
      -B${pkgs.glibc}/lib \\
      -L${rocm}/lib \\
      -L${pkgs.stdenv.cc.cc.lib}/lib \\
      -L${gccLibWrapper}/lib \\
      -L${pkgs.glibc}/lib \\
      "\$@"
    EOF
    chmod +x $out/bin/clang
    
    # Create clang++ wrapper
    cat > $out/bin/clang++ << EOF
    #!${pkgs.bash}/bin/bash
    exec ${rocm}/llvm/bin/clang++ \\
      -isystem ${pkgs.glibc.dev}/include \\
      -isystem ${pkgs.stdenv.cc.cc}/include/c++/${pkgs.stdenv.cc.cc.version} \\
      -isystem ${pkgs.stdenv.cc.cc}/include/c++/${pkgs.stdenv.cc.cc.version}/x86_64-unknown-linux-gnu \\
      -I${rocm}/include \\
      -I${rocm}/include/hip \\
      -B${pkgs.glibc}/lib \\
      -L${rocm}/lib \\
      -L${pkgs.stdenv.cc.cc.lib}/lib \\
      -L${gccLibWrapper}/lib \\
      -L${pkgs.glibc}/lib \\
      "\$@"
    EOF
    chmod +x $out/bin/clang++
  '';
  
  passthru = {
    inherit rocm gccLibWrapper;
  };
}