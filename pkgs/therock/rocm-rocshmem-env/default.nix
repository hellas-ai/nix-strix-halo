{
  lib,
  writeShellApplication,
  coreutils,
  openmpi,
  python3,
  therockRocmSdk,
  therockRocmEnv,
  packageSuffix,
}:

writeShellApplication {
  name = "therock-rocm-${packageSuffix}-rocshmem-env";
  runtimeInputs = [ coreutils ];
  text = ''
    setup_rocshmem_sysfs_filter() {
      local root fallback_device verbs name file

      root="$(mktemp -d "''${TMPDIR:-/tmp}/rocshmem-sysfs.XXXXXX")"
      mkdir -p "$root/class/infiniband" "$root/class/infiniband_verbs"

      shopt -s nullglob

      for verbs in /sys/class/infiniband_verbs/*; do
        if [ -e "$verbs/device" ]; then
          fallback_device="$verbs/device"
          break
        fi
      done

      for verbs in /sys/class/infiniband/*; do
        ln -s "$verbs" "$root/class/infiniband/$(basename "$verbs")"
      done

      for verbs in /sys/class/infiniband_verbs/*; do
        name="$(basename "$verbs")"
        if [ -e "$verbs/device" ] || [ -z "''${fallback_device:-}" ]; then
          ln -s "$verbs" "$root/class/infiniband_verbs/$name"
          continue
        fi

        mkdir -p "$root/class/infiniband_verbs/$name"
        for file in abi_version dev ibdev; do
          if [ -e "$verbs/$file" ]; then
            ln -s "$verbs/$file" "$root/class/infiniband_verbs/$name/$file"
          fi
        done
        ln -s "$fallback_device" "$root/class/infiniband_verbs/$name/device"
      done

      printf '%s\n' "$root"
    }

    rocshmem_sysfs_filter=
    if [ -z "''${SYSFS_PATH:-}" ] && [ -d /sys/class/infiniband_verbs ]; then
      rocshmem_sysfs_filter="$(setup_rocshmem_sysfs_filter)"
      export SYSFS_PATH="$rocshmem_sysfs_filter"
    fi

    cleanup_rocshmem_sysfs_filter() {
      if [ -n "$rocshmem_sysfs_filter" ]; then
        rm -rf "$rocshmem_sysfs_filter"
      fi
    }
    trap cleanup_rocshmem_sysfs_filter EXIT

    export PATH="${openmpi}/bin:${python3}/bin:${therockRocmSdk}/share/rocshmem:$PATH"
    export LD_LIBRARY_PATH="${openmpi}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OMPI_MCA_pml="''${OMPI_MCA_pml:-^ucx}"
    export OMPI_MCA_osc="''${OMPI_MCA_osc:-^ucx}"

    ${therockRocmEnv}/bin/therock-rocm-${packageSuffix}-env "$@"
  '';

  meta = with lib; {
    description = "Run a command inside TheRock ${packageSuffix} with rocSHMEM + IB verbs sysfs filter";
    platforms = platforms.linux;
  };
}
