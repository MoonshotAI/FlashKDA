"""Immutable runtime identity for native kernel selection and measurements."""
from dataclasses import asdict, dataclass
from functools import lru_cache


@dataclass(frozen=True)
class DeviceCaps:
    context: int
    device: int
    soc: str
    target: str
    cube_cores: int
    vector_cores: int
    total_memory: int

    @property
    def is_950(self) -> bool:
        return self.soc.startswith("Ascend950")

    def report(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=None)
def _probe(context: int, device: int) -> DeviceCaps:
    import torch_npu
    from flash_kda_npu_C import soc_name, target_option

    props = torch_npu.npu.get_device_properties(device)
    soc = soc_name()
    cube = int(getattr(props, "cube_core_num", 0))
    vector = int(getattr(props, "vector_core_num", 0))
    if soc.startswith("Ascend950") and (cube <= 0 or vector != 2 * cube):
        raise RuntimeError(f"unverified 950 core geometry: cube={cube}, vector={vector}")
    return DeviceCaps(context, device, soc, target_option(), cube, vector,
                      int(props.total_memory))


def current_caps() -> DeviceCaps:
    from flash_kda_npu_C import runtime_key
    return _probe(*runtime_key())
