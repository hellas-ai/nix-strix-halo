# SPDX-License-Identifier: Apache-2.0
"""STAGED source for network-attached heterogeneous expert execution.

This file is not imported by the packaged SGLang wheel in this repository. It
is meant to be applied at:

``python/sglang/srt/layers/moe/remote_ep_wrapper.py``

The exact upstream attach point is
``python/sglang/srt/layers/moe/fused_moe_triton/layer.py:~318``, beside the
existing ``KTEPWrapperMethod`` selection in ``FusedMoE.__init__``.

With SGLang ``--enable-dp-attention``, the intended composition keeps the dense
path (attention, router, shared expert, and KV cache) data-parallel across fast
GPUs. The appropriate dense rank submits sparse routes to remote expert
servers. Those servers are foreign executors attached to a rank, not SGLang EP
ranks and not participants in the dense-path process groups.

This is an integration scaffold. In particular, checkpoint loading, arbitrary
local-expert weight remapping, process-group ownership, server arguments, and
the RoCE data plane all require upstream wiring before use.
"""

from __future__ import annotations

import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import torch

from expert_server.protocol import ForwardResponse, LoadWeightsRequest
from expert_server.transport import SocketTransportClient, TransportClient
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


LOCAL_EXPERT = -1


@dataclass(frozen=True)
class RemoteEndpoint:
    """Network endpoint for one remote expert node."""

    node_id: int
    host: str
    port: int


@dataclass(frozen=True)
class RemoteWeightShard:
    """One node's contiguous checkpoint shard for a single MoE layer.

    The range may include a retained local expert. Its weight is then stored
    redundantly on the server, but placement-aware scatter never sends that
    expert's routes to the server. This keeps M1 storage contiguous while the
    placement map remains arbitrary.

    Scale tensors remain deliberately opaque until the fp8 checkpoint and
    grouped-GEMM formats are selected.
    """

    experts_start: int
    gate_up: torch.Tensor
    down: torch.Tensor
    storage_dtype: str
    gate_up_scale: torch.Tensor | None = None
    down_scale: torch.Tensor | None = None


@dataclass(frozen=True)
class RemoteEPConfig:
    """Configuration for one layer's remote expert executor.

    ``expert_placement`` maps every global expert ID to ``None`` for a local
    expert or to a remote integer node ID. Unlike KTransformers'
    ``num_gpu_experts`` threshold, this permits arbitrary hot experts to stay on
    the dense GPU.

    TODO: construct this object from version-pinned SGLang server arguments and
    share endpoint clients across all layers in a model runner.
    """

    layer_idx: int
    expert_placement: Mapping[int, int | None]
    endpoints: Mapping[int, RemoteEndpoint]
    request_timeout: float = 60.0
    max_in_flight_per_node: int = 4


@dataclass
class _PendingRemote:
    """Bookkeeping for one node's outstanding partial output."""

    node_id: int
    token_indices: torch.Tensor
    future: Future[ForwardResponse]


def placement_to_tensor(
    placement: Mapping[int, int | None],
    num_experts: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert an expert-to-node map to a dense tensor.

    Local experts use ``LOCAL_EXPERT``. Every global expert must be present so
    a missing entry cannot silently route work to the wrong machine.
    """

    expected = set(range(num_experts))
    actual = set(placement)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"placement must cover experts [0, {num_experts}); "
            f"missing={missing}, extra={extra}"
        )
    node_ids = [
        LOCAL_EXPERT if placement[expert_id] is None else placement[expert_id]
        for expert_id in range(num_experts)
    ]
    if any(node_id < LOCAL_EXPERT for node_id in node_ids):
        raise ValueError("remote node IDs must be non-negative")
    return torch.tensor(node_ids, dtype=torch.int64, device=device)


def mask_remote_expert_ids(
    topk_ids: torch.Tensor,
    placement: torch.Tensor | Mapping[int, int | None],
) -> torch.Tensor:
    """Mask remotely placed expert IDs with ``-1`` for the local GPU kernel.

    This generalizes KTransformers' ``mask_cpu_expert_ids`` threshold to an
    arbitrary expert-to-node placement. The input is not modified in place.
    ``placement`` should normally be the cached device tensor registered by
    ``RemoteEPWrapperMethod.create_weights``.
    """

    placement_tensor = _coerce_placement(placement, topk_ids.device)
    masked = topk_ids.clone()
    valid = masked >= 0
    if valid.any():
        selected_nodes = placement_tensor.index_select(0, masked[valid].long())
        selected_ids = masked[valid]
        masked[valid] = torch.where(
            selected_nodes >= 0,
            torch.full_like(selected_ids, -1),
            selected_ids,
        )
    return masked


def remap_local_expert_ids(
    masked_global_ids: torch.Tensor,
    global_to_local: torch.Tensor,
) -> torch.Tensor:
    """Map unmasked global IDs to packed physical local-expert slots."""

    remapped = masked_global_ids.clone()
    valid = remapped >= 0
    if valid.any():
        local_ids = global_to_local.index_select(0, remapped[valid].long())
        if (local_ids < 0).any():
            raise RuntimeError("an unmasked expert has no local physical slot")
        remapped[valid] = local_ids.to(remapped.dtype)
    return remapped


class RemoteEPWrapperMethod(FusedMoEMethodBase):
    """Wrap a local fused-MoE method with remote expert execution.

    The overlap contract is:

    1. ``submit_forward`` packs and submits per-node routes without waiting for
       network or expert completion.
    2. The wrapped SGLang method computes locally placed experts.
    3. ``sync_forward`` waits for node partials and merges them into the local
       result.

    The compatibility ``submit``, ``sync``, and ``apply`` methods mirror the
    current upstream ``KTEPWrapperMethod`` integration shape.
    """

    def __init__(
        self,
        local_method: FusedMoEMethodBase,
        config: RemoteEPConfig,
        *,
        clients: Mapping[int, TransportClient] | None = None,
    ) -> None:
        self.local_method = local_method
        self.config = config
        self.tp_rank = get_parallel().tp_rank
        self.override_num_local_experts = True

        self.local_expert_ids = tuple(
            expert_id
            for expert_id, node_id in sorted(config.expert_placement.items())
            if node_id is None
        )
        self.remote_node_ids = tuple(
            sorted(
                {
                    node_id
                    for node_id in config.expert_placement.values()
                    if node_id is not None
                }
            )
        )
        missing_endpoints = set(self.remote_node_ids) - set(config.endpoints)
        if missing_endpoints:
            raise ValueError(
                f"placement refers to nodes without endpoints: "
                f"{sorted(missing_endpoints)}"
            )

        if clients is None:
            self.clients = {
                node_id: SocketTransportClient(
                    config.endpoints[node_id].host,
                    config.endpoints[node_id].port,
                    timeout=config.request_timeout,
                    max_in_flight=config.max_in_flight_per_node,
                )
                for node_id in self.remote_node_ids
            }
            self._owns_clients = True
        else:
            missing_clients = set(self.remote_node_ids) - set(clients)
            if missing_clients:
                raise ValueError(
                    f"no transport clients for nodes: {sorted(missing_clients)}"
                )
            self.clients = dict(clients)
            self._owns_clients = False

        self._pending: list[_PendingRemote] | None = None
        self._placement_cpu: torch.Tensor | None = None
        self._global_to_local_cpu: torch.Tensor | None = None
        self._staged_remote_shards: Mapping[int, RemoteWeightShard] | None = None
        self.moe_runner_config: MoeRunnerConfig | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Register placement metadata and create packed local-GPU weights.

        TODO: SGLang's checkpoint loaders and quantization methods assume their
        own global-to-physical expert mapping. The integration patch must teach
        the selected method to load ``local_expert_ids`` into packed physical
        slots and divert every remote expert tensor to ``RemoteWeightShard``.
        Merely changing ``num_experts`` is not sufficient for production.
        """

        self.global_num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition
        self._placement_cpu = placement_to_tensor(
            self.config.expert_placement, num_experts, device="cpu"
        )
        global_to_local = torch.full(
            (num_experts,), -1, dtype=torch.int64, device="cpu"
        )
        for local_id, global_id in enumerate(self.local_expert_ids):
            global_to_local[global_id] = local_id
        self._global_to_local_cpu = global_to_local

        layer.register_buffer(
            "remote_ep_placement",
            self._placement_cpu.clone(),
            persistent=False,
        )
        layer.register_buffer(
            "remote_ep_global_to_local",
            self._global_to_local_cpu.clone(),
            persistent=False,
        )

        self.local_method.num_gpu_experts = len(self.local_expert_ids)
        self.local_method.create_weights(
            layer=layer,
            num_experts=len(self.local_expert_ids),
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def stage_remote_weights(
        self, shards: Mapping[int, RemoteWeightShard]
    ) -> None:
        """Provide remote shards captured by the future checkpoint-loader hook."""

        self._staged_remote_shards = dict(shards)

    def load_weights(
        self,
        layer: torch.nn.Module,
        remote_shards: Mapping[int, RemoteWeightShard],
    ) -> None:
        """Process local weights and ship this layer's remote shards.

        The transfer is synchronous because model serving must not begin until
        all endpoints acknowledge their shard. A production loader should
        pipeline layer reads and transfers and should verify a model/shard
        digest in the acknowledgment.
        """

        if hasattr(self.local_method, "process_weights_after_loading"):
            self.local_method.process_weights_after_loading(layer)
        if set(remote_shards) != set(self.remote_node_ids):
            raise ValueError(
                "remote shards must contain exactly the configured nodes: "
                f"expected={list(self.remote_node_ids)}, "
                f"actual={sorted(remote_shards)}"
            )

        for node_id in self.remote_node_ids:
            shard = remote_shards[node_id]
            self._validate_remote_shard(node_id, shard)
            self.clients[node_id].load_weights(
                LoadWeightsRequest(
                    request_id=uuid.uuid4().hex,
                    layer_idx=self.config.layer_idx,
                    experts_start=shard.experts_start,
                    gate_up=shard.gate_up,
                    down=shard.down,
                    dtype=shard.storage_dtype,
                    gate_up_scale=shard.gate_up_scale,
                    down_scale=shard.down_scale,
                )
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """SGLang-compatible hook that consumes staged remote shards."""

        if self._staged_remote_shards is None:
            raise RuntimeError(
                "remote checkpoint shards were not staged; add the TODO "
                "SGLang loader hook before using RemoteEPWrapperMethod"
            )
        self.load_weights(layer, self._staged_remote_shards)
        self._staged_remote_shards = None

    def create_moe_runner(
        self,
        layer: torch.nn.Module,
        moe_runner_config: MoeRunnerConfig,
    ) -> None:
        """Create the wrapped runner for only the packed local experts."""

        self.moe_runner_config = moe_runner_config
        if self.override_num_local_experts:
            moe_runner_config.num_local_experts = len(self.local_expert_ids)
        self.local_method.create_moe_runner(layer, moe_runner_config)

    def submit_forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        stream: object | None,
    ) -> None:
        """Non-blockingly scatter routed token rows to remote expert nodes.

        Route packing currently uses ordinary PyTorch indexing. On CUDA/ROCm,
        ``nonzero`` can synchronize while discovering output size. A production
        implementation needs device-side bounded packing and transport-owned
        CUDA/HIP events so submission remains fully asynchronous.
        """

        if self._pending is not None:
            raise RuntimeError("a remote forward is already pending for this layer")
        self._pending = []
        if self.tp_rank != 0:
            return
        placement = self._placement_on(x.device)
        safe_ids = topk_ids.clamp_min(0).long()
        selected_nodes = placement.index_select(0, safe_ids.flatten()).view_as(
            topk_ids
        )
        selected_nodes = torch.where(
            topk_ids >= 0,
            selected_nodes,
            torch.full_like(selected_nodes, LOCAL_EXPERT),
        )

        for node_id in self.remote_node_ids:
            route_mask = selected_nodes == node_id
            token_indices = route_mask.any(dim=1).nonzero(as_tuple=False).flatten()
            if token_indices.numel() == 0:
                continue
            node_topk_ids = torch.where(
                route_mask,
                topk_ids,
                torch.full_like(topk_ids, -1),
            ).index_select(0, token_indices)
            node_topk_weights = torch.where(
                route_mask,
                topk_weights,
                torch.zeros_like(topk_weights),
            ).index_select(0, token_indices)
            node_x = x.index_select(0, token_indices)
            future = self.clients[node_id].submit_forward(
                node_x,
                node_topk_ids,
                node_topk_weights,
                layer_idx=self.config.layer_idx,
                stream=stream,
            )
            self._pending.append(
                _PendingRemote(
                    node_id=node_id,
                    token_indices=token_indices,
                    future=future,
                )
            )

    def sync_forward(
        self,
        x: torch.Tensor,
        stream: object | None = None,
    ) -> torch.Tensor:
        """Wait for remote partials and merge them into a dense-shaped tensor."""

        del stream  # The TCP fallback merges on PyTorch's current stream.
        if self._pending is None:
            raise RuntimeError("sync_forward called without submit_forward")
        pending, self._pending = self._pending, None
        output = torch.zeros_like(x)
        for submission in pending:
            response = submission.future.result()
            remote_output = response.output.to(
                device=x.device,
                dtype=x.dtype,
                non_blocking=True,
            )
            output.index_add_(0, submission.token_indices, remote_output)
        return output

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> None:
        """Adapt SGLang's dispatch object to the foreign-executor contract."""

        del layer
        if self.moe_runner_config is not None:
            assert self.moe_runner_config.activation == "silu", (
                "the expert-server scaffold supports only SiLU-gated experts"
            )
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        self.submit_forward(
            dispatch_output.hidden_states,
            topk_ids,
            topk_weights,
            torch.cuda.current_stream(dispatch_output.hidden_states.device),
        )

    def sync(self, x: torch.Tensor) -> torch.Tensor:
        """SGLang-compatible alias for ``sync_forward``."""

        return self.sync_forward(x, torch.cuda.current_stream(x.device))

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        """Overlap remote experts with the wrapped local fused-MoE method."""

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        self.submit_forward(
            x,
            topk_ids,
            topk_weights,
            torch.cuda.current_stream(x.device),
        )

        masked_global_ids = mask_remote_expert_ids(
            topk_ids, self._placement_on(topk_ids.device)
        )
        local_topk_ids = remap_local_expert_ids(
            masked_global_ids,
            self._global_to_local_on(topk_ids.device),
        )
        local_topk_output = topk_output._replace(topk_ids=local_topk_ids)
        local_dispatch_output = dispatch_output._replace(
            topk_output=local_topk_output
        )

        local_combine_input = self.local_method.apply(
            layer, local_dispatch_output
        )
        output = local_combine_input.hidden_states + self.sync_forward(x)
        return StandardCombineInput(hidden_states=output)

    def close(self) -> None:
        """Close clients owned by this wrapper.

        TODO: connect this to ModelRunner shutdown. Production code should use a
        process-wide endpoint/client registry rather than one pool per layer.
        """

        if self._owns_clients:
            for client in self.clients.values():
                client.close()

    def _placement_on(self, device: torch.device) -> torch.Tensor:
        if self._placement_cpu is None:
            raise RuntimeError("create_weights must run before forward")
        return self._placement_cpu.to(device=device, non_blocking=True)

    def _global_to_local_on(self, device: torch.device) -> torch.Tensor:
        if self._global_to_local_cpu is None:
            raise RuntimeError("create_weights must run before forward")
        return self._global_to_local_cpu.to(device=device, non_blocking=True)

    def _validate_remote_shard(
        self, node_id: int, shard: RemoteWeightShard
    ) -> None:
        node_experts = sorted(
            expert_id
            for expert_id, placed_node in self.config.expert_placement.items()
            if placed_node == node_id
        )
        shard_count = shard.gate_up.shape[0]
        if shard.down.shape[0] != shard_count:
            raise ValueError(
                f"node {node_id} gate_up/down expert counts differ: "
                f"{shard_count} != {shard.down.shape[0]}"
            )
        shard_range = range(shard.experts_start, shard.experts_start + shard_count)
        shard_ids = set(shard_range)
        out_of_model = sorted(shard_ids - set(self.config.expert_placement))
        if out_of_model:
            raise ValueError(
                f"node {node_id} shard contains out-of-model experts: {out_of_model}"
            )
        missing = sorted(set(node_experts) - shard_ids)
        if missing:
            raise ValueError(
                f"node {node_id} shard does not contain its placed experts: {missing}"
            )
        conflicting = sorted(
            expert_id
            for expert_id in shard_range
            if self.config.expert_placement[expert_id] not in (None, node_id)
        )
        if conflicting:
            raise ValueError(
                f"node {node_id} contiguous shard overlaps experts assigned "
                f"to another remote node: {conflicting}"
            )

    def __getattr__(self, name: str):
        """Delegate quantization-specific attributes to the local method."""

        if name in ("local_method", "config", "clients"):
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        return getattr(self.local_method, name)


def _coerce_placement(
    placement: torch.Tensor | Mapping[int, int | None],
    device: torch.device,
) -> torch.Tensor:
    if isinstance(placement, torch.Tensor):
        return placement.to(device=device, dtype=torch.int64, non_blocking=True)
    return placement_to_tensor(placement, len(placement), device=device)
