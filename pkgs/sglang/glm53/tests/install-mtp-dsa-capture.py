"""Bounded eager DSA localization; stripping calls recovers original AST."""
import ast
from pathlib import Path
import sys
path=Path(sys.argv[1])/'sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py'
original=path.read_text();text=original
text=text.replace('import torch\n','import torch\nfrom sglang.srt.models import glm53_mtp_capture as _mtp_capture\n',1)
def before(anchor,site,args):
    global text
    assert text.count(anchor)==1,anchor
    text=text.replace(anchor,f'        _mtp_capture.record(self.layer_id, "{site}", forward_batch, {args})\n\n'+anchor)
before('        q_nope, q_pe, k_pe = self._split_q_nope_pe(q, latent_cache)','dsa_projection','q=q, latent=latent_cache, k_nope=k_nope, topk=topk_indices')
before('        return (\n            q_pe,','dsa_prepared','q=q_nope_out, q_rope=q_pe, k=k_nope, k_rope=k_pe, topk=topk_indices')
before('        _kvb_v = None\n','dsa_core','core=attn_output')
before('        output, _ = self.o_proj(attn_bmm_output)','dsa_v_absorb','values=attn_bmm_output')
class Strip(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        return None if any(a.asname=='_mtp_capture' for a in node.names) else node
    def visit_Expr(self,node):
        return None if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='_mtp_capture.record' else node
assert ast.dump(Strip().visit(ast.parse(text)))==ast.dump(ast.parse(original))
path.write_text(text)
print('Installed DSA capture; model AST unchanged after stripping diagnostics')
