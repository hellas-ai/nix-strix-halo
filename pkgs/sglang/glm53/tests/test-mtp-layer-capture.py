"""Diagnostic capture handles quantized tensor pairs without altering inputs."""
import argparse,importlib.util,json,os,sys,types
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--helper',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
os.environ['GLM53_MTP_CAPTURE_ROOT']=str(a.out);os.environ['GLM53_MTP_CAPTURE_POSITION']='3869'
spec=importlib.util.spec_from_file_location('capture',a.helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
sys.modules['sglang.srt.distributed']=types.SimpleNamespace(get_tensor_model_parallel_rank=lambda:0)
torch.cuda.is_current_stream_capturing=lambda:False
batch=types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda:True,is_target_verify=lambda:False),positions=torch.tensor([3869]),input_ids=torch.tensor([1654]))
value=torch.arange(16,dtype=torch.float32).reshape(1,16).to(torch.float8_e4m3fn)
scales=torch.tensor([[.5,.25]]);saved=value.float().clone()
module.record(3,'dsa_v_absorb',batch,values=(value,scales),unused=None)
packet=torch.load(a.out/'rank0-layer03-dsa_v_absorb.pt',weights_only=True)
assert torch.equal(packet['tensors']['values.0'],saved)
assert torch.equal(packet['tensors']['values.1'],scales)
assert packet['tensors']['unused'] is None and torch.equal(value.float(),saved)
module.record(3,'dsa_v_absorb',batch,values=(value,scales),unused=None)
module.record(45,'draft',batch,values=value)
module.record(0,'dsa_core',batch,values=value)
batch.positions=torch.tensor([3870]);module.record(4,'wrong-position',batch,values=value)
assert len(list(a.out.glob('*.pt')))==1
print('PASS tuple/FP8 capture, unchanged inputs, one-shot/position/layer bounds')
