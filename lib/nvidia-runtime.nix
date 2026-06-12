# Host NVIDIA userspace driver paths shared by the benchmark derivations,
# the benchmark-runner module, and the cuda-host-driver-runtime CI check.
#
# libcuda.so.1 must match the host's loaded kernel module, so benchmark
# derivations cannot depend on an nvidia_x11 derivation (it would both
# diverge from the host module and leak per-host driver versions into
# benchmark hashes). NixOS already maintains /run/opengl-driver as the
# canonical host GPU userspace; the runner module binds it into the build
# sandbox, plus a symlink-free alias for LD_LIBRARY_PATH:
#
#   /run/opengl-driver                              (canonical symlink)
#   /run/opengl-driver-lib=/run/opengl-driver/lib   (resolved at bind time)
#
# The store path of config.hardware.nvidia.package is also exposed so that the
# aggregate's symlinks resolve inside the sandbox.
{
  libraryPath = "/run/opengl-driver-lib";
  fallbackLibraryPath = "/run/opengl-driver/lib";
  sandboxPaths = [
    "/run/opengl-driver"
    "/run/opengl-driver-lib"
  ];
}
