{
  lib,
  runCommand,
  fetchurl,
}:

let
  mesFirmwareRev = "3d5c8135206cef364e7d353711b3e7358a90d152";
  fetchMesFirmware =
    file: hash:
    fetchurl {
      url = "https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/plain/amdgpu/${file}?id=${mesFirmwareRev}";
      inherit hash;
    };
in
runCommand "strix-halo-mes-firmware-0x80"
  {
    meta = {
      description = "Strix Halo MES 0x80 firmware blobs";
      homepage = "https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git";
      license = lib.licenses.unfreeRedistributableFirmware;
      platforms = lib.platforms.linux;
    };
  }
  ''
    install -Dm644 ${fetchMesFirmware "gc_11_5_0_mes_2.bin" "sha256-XdxUTOMcScfvDxQQyo7oi3KmrARRS9Ec+v/gBJQ5ce0="} \
      "$out/lib/firmware/amdgpu/gc_11_5_0_mes_2.bin"
    install -Dm644 ${fetchMesFirmware "gc_11_5_1_mes_2.bin" "sha256-jgeDLBjYe3ZD/CIWM9hBpfo7rg59Kq0fKyEuGQ+WOgo="} \
      "$out/lib/firmware/amdgpu/gc_11_5_1_mes_2.bin"
  ''
