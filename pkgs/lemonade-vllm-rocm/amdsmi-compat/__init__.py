import os


class AmdSmiException(Exception):
    pass


def _device_count():
    return int(os.getenv("AMDSMI_COMPAT_DEVICE_COUNT", "1"))


def _gfx_arch():
    return os.getenv("AMDSMI_COMPAT_GFX", "gfx1151")


def _device_name():
    return os.getenv("AMDSMI_COMPAT_DEVICE_NAME", "Radeon 8060S Graphics")


def amdsmi_init():
    return None


def amdsmi_shut_down():
    return None


def amdsmi_get_processor_handles():
    return list(range(_device_count()))


def amdsmi_get_gpu_asic_info(handle):
    if handle < 0 or handle >= _device_count():
        raise AmdSmiException(f"invalid processor handle: {handle}")

    return {
        "target_graphics_version": _gfx_arch(),
        "market_name": _device_name(),
        "device_id": "",
        "asic_serial": f"0x{handle + 1:032x}",
    }


def amdsmi_get_gpu_device_uuid(handle):
    return f"GPU-{handle}"


def amdsmi_topo_get_link_type(handle, peer_handle):
    if handle == peer_handle:
        return {"hops": 0, "type": 0}
    return {"hops": 2, "type": 0}


def amdsmi_topo_get_numa_node_number(handle):
    return 0
