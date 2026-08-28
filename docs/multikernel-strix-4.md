# Strix Halo hardware validation

`strix-4` completed the bare-metal multikernel lifecycle on 2026-08-27. The
host PXE-booted Linux `7.0.0-mk2`; its boot ID remained
`b8114cee-4294-4d50-88de-c38828906fa3` throughout every runtime operation.

The host created an 8 GiB pool from physical APIC IDs 24 through 31 and booted
two independent kernels from the packaged ELF `vmlinux` and static demo
initramfs:

| Instance | APIC IDs | RAM | First boot ID |
| --- | --- | --- | --- |
| blue | 24-27 | 2 GiB | `dcfbdc80-3176-424d-8180-8afd57abaa94` |
| red | 28-31 | 2 GiB | `2dfade39-88a5-43f7-8d0b-c98fb82ffbb5` |

Both spawn kernels reported four online CPUs, Linux `7.0.0-mk2`, a private
PID 1, and about 2 GiB of memory. Blue was then deliberately crashed through
`/proc/sysrq-trigger`. Its console reported:

```text
Kernel panic - not syncing: sysrq triggered crash
Multikernel instance 1 shutting down
Instance 'blue' is no longer active (status: loaded).
```

The host remained reachable with its original boot ID. Red remained `active`
with its original boot ID and continued its heartbeat. Blue was respawned from
the loaded kernel image and came back with the new boot ID
`55f6d76c-2b2c-481c-9901-c237f8b55ff4`.

Finally both instances were stopped, unloaded and deleted. Kerf returned APIC
IDs 24 through 31 and the complete 8 GiB pool; the host again reported CPUs
`0-31` online without rebooting.
