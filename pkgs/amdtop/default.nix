{
  lib,
  rustPlatform,
  fetchFromGitHub,
  libdrm,
  versionCheckHook,
  nix-update-script,
}:

rustPlatform.buildRustPackage (finalAttrs: {
  pname = "amdtop";
  version = "0.2.6";

  src = fetchFromGitHub {
    owner = "lhl";
    repo = "amdtop";
    tag = "v${finalAttrs.version}";
    hash = "sha256-B7/J7jZrJosNot6hnTJ568DxqG+ZRurBPoMDrO4diBk=";
  };

  cargoHash = "sha256-JXnZ9tYXZywcemN5fQRP8efyhnV+8WVxguk2JEugpj0=";

  # libamdgpu_top reads GPU telemetry through libdrm_amdgpu_sys, which links
  # against libdrm_amdgpu at build time rather than dlopen-ing it.
  buildInputs = [ libdrm ];

  strictDeps = true;

  nativeInstallCheckInputs = [ versionCheckHook ];
  doInstallCheck = true;

  passthru.updateScript = nix-update-script { };

  meta = {
    description = "Terminal system monitor for AMD GPUs, CPUs, and XDNA NPUs";
    longDescription = ''
      A btop/nvitop-style TUI that monitors CPUs, AMD GPUs (discrete cards and
      APUs) via libamdgpu_top, and AMD XDNA NPUs exposed through the Linux
      accel class. Reports per-core CPU history, VRAM/GTT pools, memory
      bandwidth, per-process engine usage, and ships 41 bundled themes.
    '';
    homepage = "https://github.com/lhl/amdtop";
    changelog = "https://github.com/lhl/amdtop/blob/v${finalAttrs.version}/CHANGELOG.md";
    license = lib.licenses.asl20;
    mainProgram = "amdtop";
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = lib.platforms.linux;
  };
})
