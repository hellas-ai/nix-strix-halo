"""SGLang foreign-executor wrapper for remote heterogeneous expert parallelism.

Copy this module to ``sglang/srt/layers/moe/remote_ep_wrapper.py`` in the
packaged SGLang source/site-packages tree.  It follows KTEPWrapperMethod's
submit -> local GPU compute -> sync-and-merge lifecycle while replacing the
KTransformers CPU executor with an injected M1 RCCL/RDMA expert client.

``RemoteEPClient`` is the client-side adapter around M1's shared
``RCCLRDMAAllToAllTransport``.  A site-specific replacement may still be
selected with ``--remote-ep-client-class MODULE:OBJECT``.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol

import torch

from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs


GPU_OWNER = None


@dataclass(frozen=True)
class PlacementMap:
    """Global expert placement; ``None`` means a hot expert stays on this GPU."""

    owners: tuple[int | None, ...]

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PlacementMap":
        document = json.loads(Path(path).read_text())
        raw = document.get("experts", document) if isinstance(document, dict) else document
        if isinstance(raw, list):
            values = raw
        elif isinstance(raw, dict):
            try:
                maximum = max(int(key) for key in raw)
                values = [raw[str(expert_id)] for expert_id in range(maximum + 1)]
            except (KeyError, ValueError) as error:
                raise ValueError(
                    "placement object must contain contiguous string expert IDs from 0"
                ) from error
        else:
            raise ValueError("placement JSON must be a list or an expert-ID object")

        owners: list[int | None] = []
        for expert_id, value in enumerate(values):
            if value is None or value == "gpu":
                owners.append(GPU_OWNER)
            elif isinstance(value, int) and value >= 0:
                owners.append(value)
            else:
                raise ValueError(
                    f"expert {expert_id} owner must be a non-negative node ID, null, or 'gpu'"
                )
        return cls(tuple(owners))

    def validate(self, *, num_experts: int, endpoints: tuple[str, ...]) -> None:
        if len(self.owners) != num_experts:
            raise ValueError(
                f"placement has {len(self.owners)} experts, but layer has {num_experts}"
            )
        remote_nodes = [owner for owner in self.owners if owner is not GPU_OWNER]
        if remote_nodes and not endpoints:
            raise ValueError("remote placement requires --remote-ep-endpoints")
        if remote_nodes and max(remote_nodes) >= len(endpoints):
            raise ValueError(
                f"placement names node {max(remote_nodes)}, but only "
                f"{len(endpoints)} endpoints were configured"
            )

    @property
    def gpu_expert_ids(self) -> tuple[int, ...]:
        return tuple(
            expert_id
            for expert_id, owner in enumerate(self.owners)
            if owner is GPU_OWNER
        )

    @property
    def remote_owners(self) -> dict[int, int]:
        return {
            expert_id: owner
            for expert_id, owner in enumerate(self.owners)
            if owner is not GPU_OWNER
        }

    def gpu_id_lookup(self, device: torch.device) -> torch.Tensor:
        """Map global expert IDs to compact local-hot IDs, with -1 for remote."""

        lookup = torch.full((len(self.owners),), -1, dtype=torch.long, device=device)
        gpu_ids = self.gpu_expert_ids
        if gpu_ids:
            global_ids = torch.tensor(gpu_ids, dtype=torch.long, device=device)
            lookup[global_ids] = torch.arange(len(gpu_ids), device=device)
        return lookup


@dataclass(frozen=True)
class RemoteEPConfig:
    layer_idx: int
    placement: PlacementMap
    endpoints: tuple[str, ...]
    transport: str
    client_class: str | None
    timeout_ms: int


def create_remote_ep_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> RemoteEPConfig | None:
    """Build per-layer remote-EP configuration when the graft is enabled."""

    placement_path = getattr(server_args, "remote_ep_placement", None)
    if placement_path is None:
        return None
    raw_endpoints = getattr(server_args, "remote_ep_endpoints", None)
    endpoints = (
        tuple(part.strip() for part in raw_endpoints.split(",") if part.strip())
        if raw_endpoints
        else ()
    )
    timeout_ms = int(getattr(server_args, "remote_ep_timeout_ms", 30000))
    if timeout_ms <= 0:
        raise ValueError("--remote-ep-timeout-ms must be positive")
    return RemoteEPConfig(
        layer_idx=layer_idx,
        placement=PlacementMap.from_json_file(placement_path),
        endpoints=endpoints,
        transport=getattr(server_args, "remote_ep_transport", "rccl-rdma"),
        client_class=getattr(server_args, "remote_ep_client_class", None),
        timeout_ms=timeout_ms,
    )


class RemoteEPClientProtocol(Protocol):
    """Foreign-executor contract implemented by a remote transport client.

    ``submit_forward`` must enqueue scatter/remote execution without waiting.
    ``sync_forward`` waits for gather completion and returns the already
    route-weighted sum of all remote expert nodes as ``[T, H]`` with ``x``'s
    dtype and device.
    The payload expert IDs remain global; the client partitions them with the
    placement map and includes ``layer_idx`` in the M1 request envelope.
    """

    def load_weights(
        self,
        *,
        layer_idx: int,
        placement: Mapping[int, int],
        endpoints: tuple[str, ...],
    ) -> None: ...

    def submit_forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        stream: int,
    ) -> None: ...

    def sync_forward(self, x: torch.Tensor, stream: int) -> torch.Tensor: ...


class RemoteEPClient:
    """SGLang adapter for M1's shared RCCL/RDMA all-to-all transport.

    ``submit_forward`` starts M1's background dispatch continuation and returns
    immediately.  ``sync_forward`` participates in the inverse collective and
    reduces the returned unique (token, destination) pair rows, which M1 has
    already pre-reduced over duplicate same-node assignments.  The collective
    rendezvous is the enclosing torch.distributed job; ``endpoints`` retain the
    placement/node ordering and are validated here, but are not a second socket
    data path.  A transport error is unrecoverable: the RCCL transport poisons
    itself on any failure or timeout, so a forward that raises leaves the
    instance unusable and the process must be restarted.
    """

    def __init__(self, config: RemoteEPConfig) -> None:
        self.config = config
        self._transport: Any | None = None
        self._loaded_layer: int | None = None
        self._pending_shape: tuple[int, ...] | None = None
        self._no_remote = False

    @staticmethod
    def _server_ranks(endpoint_count: int) -> tuple[int, ...]:
        import torch.distributed as dist

        configured = os.environ.get("REMOTE_EP_SERVER_RANKS")
        if configured:
            try:
                ranks = tuple(int(value.strip()) for value in configured.split(","))
            except ValueError as error:
                raise ValueError(
                    "REMOTE_EP_SERVER_RANKS must be comma-separated integers"
                ) from error
            if len(ranks) != endpoint_count:
                raise ValueError(
                    f"REMOTE_EP_SERVER_RANKS has {len(ranks)} ranks, expected "
                    f"{endpoint_count} from --remote-ep-endpoints"
                )
            return ranks

        if not dist.is_initialized():
            # M1 initializes the RCCL default group when the adapter constructs
            # its transport.  torchrun has already supplied WORLD_SIZE here.
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        else:
            world_size = dist.get_world_size()
        if world_size == endpoint_count:
            return tuple(range(endpoint_count))
        if world_size == 2 * endpoint_count:
            return tuple(range(endpoint_count, 2 * endpoint_count))
        raise ValueError(
            f"cannot infer {endpoint_count} expert ranks from WORLD_SIZE={world_size}; "
            "set REMOTE_EP_SERVER_RANKS explicitly"
        )

    @staticmethod
    def _transport_class() -> type[Any]:
        # The M1 directory is installed on PYTHONPATH by the package/runbook.
        # Keeping the module name configurable avoids copying transport logic
        # into the SGLang package and permits a normal packaged module name.
        module_name = os.environ.get("REMOTE_EP_TRANSPORT_MODULE", "transport")
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise ImportError(
                f"cannot import M1 transport module {module_name!r}; add "
                "pkgs/moe-expert-server to PYTHONPATH or set "
                "REMOTE_EP_TRANSPORT_MODULE"
            ) from error
        implementation = getattr(module, "RCCLRDMAAllToAllTransport", None)
        if not isinstance(implementation, type):
            raise TypeError(
                f"{module_name} does not export RCCLRDMAAllToAllTransport"
            )
        return implementation

    def load_weights(
        self,
        *,
        layer_idx: int,
        placement: Mapping[int, int],
        endpoints: tuple[str, ...],
    ) -> None:
        if self.config.transport != "rccl-rdma":
            raise ValueError(
                f"RemoteEPClient supports transport='rccl-rdma', got "
                f"{self.config.transport!r}"
            )
        if layer_idx != self.config.layer_idx:
            raise ValueError(
                f"client configured for layer {self.config.layer_idx}, got {layer_idx}"
            )
        if self._transport is not None:
            if self._loaded_layer != layer_idx:
                raise RuntimeError("cannot reuse a remote client for another layer")
            return
        if placement and not endpoints:
            raise ValueError("remote expert placement requires endpoints")
        if not placement:
            self._loaded_layer = layer_idx
            self._no_remote = True
            return
        self._transport = self._transport_class()(
            placement=placement,
            server_ranks=self._server_ranks(len(endpoints)),
            num_experts=len(self.config.placement.owners),
            device=torch.device("cuda", torch.cuda.current_device()),
            timeout_ms=self.config.timeout_ms,
        )
        self._loaded_layer = layer_idx

    def submit_forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        stream: int,
    ) -> None:
        if self._loaded_layer is None:
            raise RuntimeError("RemoteEPClient.load_weights() must run before forward")
        if self._pending_shape is not None:
            raise RuntimeError("RemoteEPClient already has an in-flight forward")
        if not self._no_remote:
            if self._transport is None:
                raise RuntimeError("RCCL transport was not initialized")
            self._transport.submit_request(
                layer_id=self._loaded_layer,
                hidden_states=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                stream=stream,
            )
        self._pending_shape = tuple(x.shape)

    def sync_forward(self, x: torch.Tensor, stream: int) -> torch.Tensor:
        del stream  # The transport retained the submit stream for combine.
        if self._pending_shape is None:
            raise RuntimeError("sync_forward called without submit_forward")
        if tuple(x.shape) != self._pending_shape:
            raise ValueError(
                f"sync tensor shape {tuple(x.shape)} differs from submitted "
                f"shape {self._pending_shape}"
            )
        try:
            if self._no_remote:
                output = torch.zeros_like(x)
            else:
                if self._transport is None:
                    raise RuntimeError("RCCL transport was not initialized")
                output = self._transport.sync_response()
        finally:
            self._pending_shape = None
        return output.to(device=x.device, dtype=x.dtype)

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()


# Backward-compatible name for early M2 configurations.
RCCLRDMAExpertClient = RemoteEPClient


def _load_client(config: RemoteEPConfig) -> RemoteEPClientProtocol:
    if config.client_class is None:
        return RemoteEPClient(config)
    try:
        module_name, object_name = config.client_class.split(":", maxsplit=1)
    except ValueError as error:
        raise ValueError("--remote-ep-client-class must be MODULE:OBJECT") from error
    factory = getattr(importlib.import_module(module_name), object_name)
    client = factory(config)
    for method in ("load_weights", "submit_forward", "sync_forward"):
        if not callable(getattr(client, method, None)):
            raise TypeError(f"remote EP client {config.client_class} lacks {method}()")
    return client


class RemoteEPWrapperMethod(FusedMoEMethodBase):
    """Wrap any SGLang GPU MoE method with asynchronous remote experts.

    This generalizes KTransformers' ``expert_id < num_gpu_experts`` threshold.
    Every global expert has an explicit placement.  Hot GPU experts may be any
    IDs: checkpoint loads and routed IDs are remapped to a compact local index.
    Remote expert IDs are set to -1 only for the wrapped GPU method, while the
    M1 client always receives the untouched global routing tensors.
    """

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        remote_config: RemoteEPConfig,
        *,
        client: RemoteEPClientProtocol | None = None,
    ) -> None:
        self.gpu_method = gpu_method
        self.remote_config = remote_config
        self.placement = remote_config.placement
        self.client = client if client is not None else _load_client(remote_config)
        self.override_num_local_experts = True
        self.global_num_experts = 0
        self.num_gpu_experts = 0
        self._gpu_global_expert_ids: tuple[int, ...] = ()
        self._gpu_id_lookup: torch.Tensor | None = None
        self._pending = False
        self._gpu_weights_created = False

    def _remapping_weight_loader(self, original: Callable[..., Any]) -> Callable[..., Any]:
        global_to_local = {
            global_id: local_id
            for local_id, global_id in enumerate(self._gpu_global_expert_ids)
        }

        def load_hot_expert_only(*args: Any, **kwargs: Any) -> Any:
            if "expert_id" in kwargs:
                global_id = kwargs["expert_id"]
                if global_id is None:
                    return original(*args, **kwargs)
                local_id = global_to_local.get(int(global_id))
                if local_id is None:
                    return None
                remapped = dict(kwargs)
                remapped["expert_id"] = local_id
                return original(*args, **remapped)
            if len(args) < 5:
                raise TypeError(
                    "SGLang weight_loader signature changed: expected expert_id as "
                    "keyword or fifth positional argument"
                )
            global_id = args[4]
            if global_id is None:
                return original(*args, **kwargs)
            local_id = global_to_local.get(int(global_id))
            if local_id is None:
                return None
            remapped_args = (*args[:4], local_id, *args[5:])
            return original(*remapped_args, **kwargs)

        return load_hot_expert_only

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        """Create only hot GPU weights; M1 servers own all remote weights."""

        routed_experts = len(self.placement.owners)
        layer_routed_experts = getattr(layer, "_num_global_routed", routed_experts)
        shared_experts = int(getattr(layer, "num_fused_shared_experts", 0))
        if layer_routed_experts != routed_experts:
            raise ValueError(
                f"placement covers {routed_experts} routed experts, but SGLang "
                f"reports {layer_routed_experts}"
            )
        if num_experts != routed_experts + shared_experts:
            raise ValueError(
                "remote EP expects unsharded SGLang experts "
                "(--ep-size 1): create_weights received "
                f"{num_experts}, expected {routed_experts} routed + "
                f"{shared_experts} shared"
            )
        self.global_num_experts = num_experts
        self.placement.validate(
            num_experts=routed_experts, endpoints=self.remote_config.endpoints
        )
        # Fused shared experts are appended after the global routed IDs by the
        # standard FusedMoE layer.  They always remain on the dense GPU path.
        shared_ids = tuple(range(routed_experts, routed_experts + shared_experts))
        self._gpu_global_expert_ids = self.placement.gpu_expert_ids + shared_ids
        self.num_gpu_experts = len(self._gpu_global_expert_ids)
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        if self.num_gpu_experts == 0:
            return
        attrs = dict(extra_weight_attrs)
        original_loader = attrs.get("weight_loader")
        if original_loader is None:
            raise ValueError("remote EP needs SGLang's layer weight_loader callback")
        attrs["weight_loader"] = self._remapping_weight_loader(original_loader)
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **attrs,
        )
        self._gpu_weights_created = True

    def load_weights(self, layer: torch.nn.Module) -> None:
        """Finalize local hot weights and register remote placement/endpoints."""

        if self._gpu_weights_created and hasattr(
            self.gpu_method, "process_weights_after_loading"
        ):
            self.gpu_method.process_weights_after_loading(layer)
        self.client.load_weights(
            layer_idx=self.remote_config.layer_idx,
            placement=self.placement.remote_owners,
            endpoints=self.remote_config.endpoints,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.load_weights(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ) -> None:
        self.moe_runner_config = moe_runner_config
        moe_runner_config.num_local_experts = self.num_gpu_experts
        if self._gpu_weights_created:
            self.gpu_method.create_moe_runner(layer, moe_runner_config)

    def submit_forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        stream: int,
    ) -> None:
        """Non-blocking scatter/execute submission to all required M1 nodes."""

        if self._pending:
            raise RuntimeError("remote EP wrapper already has an in-flight forward")
        self.client.submit_forward(x, topk_ids, topk_weights, stream)
        self._pending = True

    def sync_forward(self, x: torch.Tensor, stream: int) -> torch.Tensor:
        """Gather and return the combined remote partial on ``x.device``."""

        if not self._pending:
            raise RuntimeError("sync_forward called without submit_forward")
        try:
            output = self.client.sync_forward(x, stream)
        finally:
            self._pending = False
        if output.shape != x.shape or output.device != x.device or output.dtype != x.dtype:
            raise ValueError(
                "remote client must return an output matching x shape, dtype, and device; "
                f"got shape={tuple(output.shape)} dtype={output.dtype} device={output.device}, "
                f"expected shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
            )
        return output

    def _mask_and_remap_gpu_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self._gpu_id_lookup is None or self._gpu_id_lookup.device != topk_ids.device:
            self._gpu_id_lookup = torch.full(
                (self.global_num_experts,),
                -1,
                dtype=torch.long,
                device=topk_ids.device,
            )
            if self._gpu_global_expert_ids:
                global_ids = torch.tensor(
                    self._gpu_global_expert_ids,
                    dtype=torch.long,
                    device=topk_ids.device,
                )
                self._gpu_id_lookup[global_ids] = torch.arange(
                    len(self._gpu_global_expert_ids), device=topk_ids.device
                )
        valid = (topk_ids >= 0) & (topk_ids < self.global_num_experts)
        safe_ids = torch.where(valid, topk_ids, torch.zeros_like(topk_ids)).long()
        remapped = self._gpu_id_lookup[safe_ids]
        return torch.where(valid, remapped, torch.full_like(remapped, -1))

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Overlap remote execution with hot GPU experts, then sum partials."""

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        if self.moe_runner_config.activation != "silu":
            raise AssertionError("remote EP scaffold currently supports only SiLU")
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        stream = torch.cuda.current_stream(x.device).cuda_stream

        self.submit_forward(x, topk_ids, topk_weights, stream)
        if self._gpu_weights_created:
            gpu_topk_ids = self._mask_and_remap_gpu_ids(topk_ids)
            gpu_topk_output = topk_output._replace(topk_ids=gpu_topk_ids)
            gpu_dispatch = dispatch_output._replace(topk_output=gpu_topk_output)
            gpu_output = self.gpu_method.apply(layer, gpu_dispatch).hidden_states
        else:
            gpu_output = torch.zeros_like(x)
        remote_output = self.sync_forward(x, stream)
        return StandardCombineInput(hidden_states=gpu_output + remote_output)

    def __getattr__(self, name: str) -> Any:
        if name in ("gpu_method", "remote_config", "placement", "client"):
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        return getattr(self.gpu_method, name)
