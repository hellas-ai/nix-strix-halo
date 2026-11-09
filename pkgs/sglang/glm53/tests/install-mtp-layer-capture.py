"""Install eager diagnostic hooks; removing calls recovers the original AST."""
import ast
from pathlib import Path
import shutil
import sys

site = Path(sys.argv[1])
path = site / "sglang/srt/models/glm5_next.py"
original = path.read_text()
text = original


def once(old, new):
    global text
    assert text.count(old) == 1, old
    text = text.replace(old, new)


def call(layer, site, **tensors):
    args = ", ".join(f"{name}={expr}" for name, expr in tensors.items())
    return f"        _mtp_capture.record({layer}, {site!r}, forward_batch, {args})\n\n"


once("from sglang.srt.environ import envs\n", "from sglang.srt.environ import envs\nfrom sglang.srt.models import glm53_mtp_capture as _mtp_capture\n")
once("        if not forward_batch.forward_mode.is_decode():\n            forget_gate = forget_gate.unsqueeze(0)",
     call("self.layer_idx", "kda_projection", qkv="mixed_qkv", beta="beta", forget="forget_gate", gate="g_proj_states") +
     "        if not forward_batch.forward_mode.is_decode():\n            forget_gate = forget_gate.unsqueeze(0)")
once("        norm_gate = g_proj_states.unflatten(-1, (-1, self.head_dim))",
     call("self.layer_idx", "kda_core", core="core_attn_out") +
     "        norm_gate = g_proj_states.unflatten(-1, (-1, self.head_dim))")
once("        hidden_states_orig = hidden_states\n\n", "        hidden_states_orig = hidden_states\n\n" +
     call("self.layer_id", "entry", hidden="hidden_states", residual="residual"))
once("        hidden_states = self.self_attn(\n", call("self.layer_id", "attn_input", hidden="hidden_states", residual="residual") + "        hidden_states = self.self_attn(\n")
once("        get_attn_tp_context().clear_attn_inputs()\n\n", "        get_attn_tp_context().clear_attn_inputs()\n\n" +
     call("self.layer_id", "attn_output", hidden="hidden_states"))
once("        should_allreduce_fusion = (\n", call("self.layer_id", "mlp_input", hidden="hidden_states", residual="residual") + "        should_allreduce_fusion = (\n")
once("        if should_allreduce_fusion:\n            hidden_states._sglang_needs_allreduce_fusion", call("self.layer_id", "mlp_output", hidden="hidden_states") + "        if should_allreduce_fusion:\n            hidden_states._sglang_needs_allreduce_fusion")
once("        return hidden_states, residual, topk_indices\n", call("self.layer_id", "exit", hidden="hidden_states", residual="residual") + "        return hidden_states, residual, topk_indices\n")


class Strip(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if any(a.asname == "_mtp_capture" for a in node.names):
            return None
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "_mtp_capture.record":
            return None
        return node


assert ast.dump(Strip().visit(ast.parse(text))) == ast.dump(ast.parse(original))
compile(text, str(path), "exec")
path.write_text(text)
shutil.copyfile(Path(sys.argv[2]), path.with_name("glm53_mtp_capture.py"))
print("Installed bounded MTP capture; model AST unchanged after stripping diagnostics")
