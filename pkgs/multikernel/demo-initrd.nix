{
  lib,
  makeInitrd,
  writeTextFile,
  busybox,
  kerf-init,
}:

let
  demo = writeTextFile {
    name = "multikernel-demo";
    executable = true;
    text = ''
      #!/bin/sh
      set -eu

      bb=/bin/busybox
      name="$($bb sed -n 's/.*multikernel.demo.name=\([^ ]*\).*/\1/p' /proc/cmdline)"
      [ -n "$name" ] || name=spawn

      online_cpus="$($bb cat /sys/devices/system/cpu/online)"
      apic_ids="$($bb awk '
        /^apicid[[:space:]]*:/ {
          if (count++) printf ",";
          printf "%s", $3;
        }
        END { print "" }
      ' /proc/cpuinfo)"

      echo
      echo "=== NixOS multikernel spawn: $name ==="
      echo "kernel: $($bb uname -r)"
      echo "boot-id: $($bb cat /proc/sys/kernel/random/boot_id)"
      echo "pid: $$ (PID 1 is kerf-init)"
      echo "cpus: logical $online_cpus; physical APIC $apic_ids"
      echo "memory: $($bb awk '/MemTotal/ { print $2 " kB" }' /proc/meminfo)"
      echo "cmdline: $($bb cat /proc/cmdline)"
      echo "This kernel owns its scheduler, memory map, PID space and failure domain."
      echo "An interactive shell is ready; 'echo c > /proc/sysrq-trigger' crashes only this spawn."
      echo

      (
        tick=0
        while :; do
          tick=$((tick + 1))
          echo "[$name] heartbeat $tick from kernel $($bb uname -r), boot $($bb cat /proc/sys/kernel/random/boot_id)"
          $bb sleep 5
        done
      ) &

      export PATH=/bin
      export PS1="[$name \\u@multikernel]# "
      exec /bin/sh -i
    '';
  };
in
(makeInitrd {
  name = "multikernel-demo-initrd";
  compressor = "zstd";
  compressorArgs = [ "-19" ];
  contents = [
    {
      object = kerf-init;
      symlink = "/init";
      suffix = "/bin/kerf-init";
    }
    {
      object = busybox;
      symlink = "/bin/busybox";
      suffix = "/bin/busybox";
    }
    {
      object = demo;
      symlink = "/bin/multikernel-demo";
    }
  ]
  ++
    map
      (applet: {
        object = busybox;
        symlink = "/bin/${applet}";
        suffix = "/bin/busybox";
      })
      [
        "awk"
        "cat"
        "dmesg"
        "grep"
        "head"
        "ls"
        "mount"
        "poweroff"
        "ps"
        "sed"
        "sh"
        "sleep"
        "sync"
        "uname"
      ];
}).overrideAttrs
  (_: {
    passthru.entrypoint = "/bin/multikernel-demo";
    meta = {
      description = "Self-contained initramfs for demonstrating independent multikernel spawn kernels";
      license = lib.licenses.asl20;
      platforms = [ "x86_64-linux" ];
      maintainers = with lib.maintainers; [ georgewhewell ];
    };
  })
