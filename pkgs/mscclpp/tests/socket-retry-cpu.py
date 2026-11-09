#!/usr/bin/env python3
"""Paired actual-method CPU qualification for recreated socket descriptor flags."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HELPER = HERE / 'socket-pollerr-cpu.py'
assert HELPER.read_bytes()
spec = importlib.util.spec_from_file_location('pollerr_cpu_support', HELPER)
H = importlib.util.module_from_spec(spec);spec.loader.exec_module(H)

def expected(name,new):
    b=dict(error='none',code=0,state=4,fd=101,retries=1,gets=1,closes=1,
        sockets=1,sleeps=1,get_flags=0,set_flags=0,new_fd_flags=os.O_APPEND)
    if name=='connected_no_recreation':
        return b|dict(state=6,fd=7,retries=0,closes=0,sockets=0,sleeps=0,new_fd_flags=0)
    if name=='fatal_error_no_recreation':
        return b|dict(error='system_error',state=10,fd=7,retries=0,
            closes=0,sockets=0,sleeps=0,new_fd_flags=0)
    if name=='synchronous_unchanged':return b
    if name=='async_zero_flags':b['new_fd_flags']=0
    if name=='socket_creation_failure':
        return b|dict(fd=-1,error='sys_error' if new else 'none',code=24 if new else 0,
            state=5 if new else 4)
    if not new:return b
    b['get_flags']=1
    if name=='get_flags_failure':return b|dict(error='sys_error',code=5,state=5)
    b['set_flags']=1
    if name=='set_flags_failure':return b|dict(error='sys_error',code=22,state=5)
    return b|dict(new_fd_flags=b['new_fd_flags']|os.O_NONBLOCK)

def check(source,patch3,patch4,compiler):
    assert all(H.sha(source/name)==pin for name,pin in H.PINS.items())
    original=(source/'src/core/bootstrap/socket.cc').read_text()
    error_source=(source/'src/core/errors.cc').read_text()
    errors=error_source[error_source.index('std::string errorToString('):
        error_source.index('MSCCLPP_API_CPP CudaError::')].replace('MSCCLPP_API_CPP ','')
    template_path=HERE/'socket-retry-fixture.cpp.in';template=template_path.read_text()
    assert template.count('/* SOURCE_ERRORS */')==template.count('/* SOURCE_POLL */')==1
    env=dict(os.environ)
    for k in ('LD_PRELOAD','LD_LIBRARY_PATH','ROCP_TOOL_LIBRARIES','ROCP_TOOL_ATTACH'):env.pop(k,None)
    env.update(HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
    rows={};generated={};needed={}
    with tempfile.TemporaryDirectory(prefix='mscclpp-retry-cpu-') as tmp:
        tmp=Path(tmp);target=tmp/'src/core/bootstrap/socket.cc'
        target.parent.mkdir(parents=True);target.write_text(original)
        bodies=[]
        for patch in (patch3,patch4):
            r=subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(patch.resolve())],
                cwd=tmp,check=True,capture_output=True,text=True)
            assert 'offset' not in r.stdout+r.stderr and 'fuzz' not in r.stdout+r.stderr
            bodies.append(H.method(target.read_text()))
        for label,body in zip(('before0004','after0004'),bodies):
            cpp=tmp/(label+'.cpp');exe=tmp/label
            cpp.write_text(template.replace('/* SOURCE_ERRORS */',errors).replace('/* SOURCE_POLL */',body))
            generated[label]=H.sha(cpp)
            subprocess.run([str(compiler),'-std=c++17','-O0','-Wall','-Wextra','-Werror',
                '-I'+str(source/'include'),'-I'+str(source/'src/core/include'),str(cpp),'-o',str(exe)],
                env=env,check=True,capture_output=True,text=True,timeout=60)
            dynamic=subprocess.run([str(H.READELF),'-d',str(exe)],check=True,capture_output=True,text=True).stdout
            needed[label]=[x.split('[')[1].split(']')[0] for x in dynamic.splitlines() if '(NEEDED)' in x]
            assert not any(any(x in n.lower() for x in ('hip','hsa','mscclpp','ibverbs','cuda')) for n in needed[label])
            r=subprocess.run([str(exe)],env=env,check=True,capture_output=True,text=True,timeout=10)
            assert r.stderr==''
            records=[json.loads(x) for x in r.stdout.splitlines()]
            assert len(records)==9 and len({x['case'] for x in records})==9
            for row in records:
                assert {k:v for k,v in row.items() if k!='case'}==expected(row['case'],label=='after0004'),row
            rows[label]=records
    changed=[a['case'] for a,b in zip(rows['before0004'],rows['after0004']) if a!=b]
    assert changed==['async_preserves_flags','abortable_preserves_flags','async_zero_flags',
        'socket_creation_failure','get_flags_failure','set_flags_failure']
    return {'schema':'MSCCLPP_RETRY_DESCRIPTOR_HOST_CPU_V1','status':'PASS','cases_per_version':9,
        'changed_cases':changed,'source':str(source),'source_pins':H.PINS,
        'patches':{str(p.resolve()):H.sha(p) for p in (patch3,patch4)},'zero_fuzz_offset':True,
        'compiler':{'path':str(compiler),'sha256':H.sha(compiler)},
        'test_source_sha256':H.sha(Path(__file__)),'helper_source_sha256':H.sha(HELPER),
        'template_sha256':H.sha(template_path),'generated_fixture_sha256':generated,
        'needed':needed,'rows':rows,'network_GPU_calls':False,
        'scope':'actual pollConnect/declaration/host exceptions; constructor and syscalls stubbed',
        'package_build':'NOT_RUN','live_qualification':'NOT_RUN','original_failure':'PRESERVED',
        'initiating_async_error_cause':'UNASSIGNED'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=H.SOURCE)
    p.add_argument('--patch3',type=Path,default=HERE.parent/'0003-bootstrap-pollerr-so-error.patch')
    p.add_argument('--patch4',type=Path,default=HERE.parent/'0004-bootstrap-retry-nonblocking.patch')
    p.add_argument('--compiler',type=Path,default=H.COMPILER);p.add_argument('--out',type=Path)
    a=p.parse_args();result=check(a.source,a.patch3,a.patch4,a.compiler)
    text=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if a.out:
        with a.out.open('x') as f:f.write(text)
    else:print(text,end='')
