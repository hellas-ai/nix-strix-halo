#!/usr/bin/env python3
"""Compile/run the exact old/new host method with all socket syscalls stubbed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

SOURCE = Path('/nix/store/ipfjfbhlpa730gp1g040mfqsx8fwvmw3-source')
COMPILER = Path('/nix/store/z4c6k0mrlkwl3s4w9ysxc8vq1wylm3ms-gcc-wrapper-15.3.0/bin/c++')
READELF = Path('/nix/store/zyqqinydzjhydl58dr5gpw2s5adqcmsc-binutils-wrapper-2.46/bin/readelf')
PINS = {
    'src/core/bootstrap/socket.cc': '0cd7aa43f7d888ff6f52ad0a739e7ad17b72522eedaa9682e560a0e844784dcd',
    'src/core/include/socket.h': 'ecfeaa03f1362ec6f91caa84bf547cff5d7e6502be5181962dd13533a8cf6517',
    'src/core/errors.cc': '9bd8df5f72c7c1f13f59d035314946d08d7babab41e421cc2fd6e940440f5791',
    'include/mscclpp/errors.hpp': '0ac256131003a82137224a786f203f3f8ede169a6e95b21b60d58a7615628d41',
}

def digest(b): return hashlib.sha256(b).hexdigest()
def sha(p): return digest(p.read_bytes())

def method(text):
    start = text.index('void Socket::pollConnect() {')
    end = text.index('\nvoid Socket::finalizeConnect()', start)
    return text[start:end].strip()

def expected(name, new):
    base = dict(error='none',code=0,state=5,fd=7,retries=0,polls=1,gets=0,
        closes=0,sockets=0,sleeps=0,socket_family=0)
    if name in ('poll_timeout',): return base
    if name == 'poll_syscall_error': return base | dict(error='sys_error',code=4)
    if name in ('hangup_only','invalid_fd_event','unexpected_empty_event','unexpected_return_count'):
        return base | dict(error='internal_error',code=1)
    if name.startswith('pollerr_') and not new:
        return base | dict(error='internal_error',code=1)
    base['gets'] = 1
    if name in ('pollerr_unreachable',): return base | dict(error='system_error',state=10)
    if name == 'pollerr_get_failure': return base | dict(error='sys_error',code=9)
    if name == 'pollerr_inprogress': return base
    if name in ('writable_success','pollerr_zero_so_error'): return base | dict(state=6)
    return base | dict(state=4,fd=101,retries=1,closes=1,sockets=1,sleeps=1,
        socket_family=10 if name.endswith('_ipv6') else 2)

def check(source, patch, compiler):
    assert all(sha(source / name) == pin for name,pin in PINS.items())
    original = (source / 'src/core/bootstrap/socket.cc').read_text()
    old = method(original)
    new = old.replace('/* check socket status */\n  if ((ret == 1 && (pfd.revents & POLLOUT)) == 0)',
        '// A pending connect error can report POLLERR without POLLOUT. Read SO_ERROR\n'
        '  // so it reaches the same retry/error handling as writable completion.\n'
        '  // Other unexpected events retain the existing early error.\n'
        '  if ((ret == 1 && (pfd.revents & (POLLOUT | POLLERR))) == 0)')
    assert old != new
    errors = (source / 'src/core/errors.cc').read_text()
    errors = errors[errors.index('std::string errorToString('):errors.index('MSCCLPP_API_CPP CudaError::')]
    errors = errors.replace('MSCCLPP_API_CPP ', '')
    template_path = Path(__file__).with_name('socket-pollerr-fixture.cpp.in')
    template = template_path.read_text()
    assert template.count('/* SOURCE_ERRORS */') == template.count('/* SOURCE_POLL */') == 1
    env = dict(os.environ)
    for key in ('LD_PRELOAD','LD_LIBRARY_PATH','ROCP_TOOL_LIBRARIES','ROCP_TOOL_ATTACH'):
        env.pop(key, None)
    env.update(HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
    rows = {}; fixture_hashes = {}; needed = {}
    with tempfile.TemporaryDirectory(prefix='mscclpp-pollerr-cpu-') as tmp:
        tmp = Path(tmp); target = tmp / 'src/core/bootstrap/socket.cc'
        target.parent.mkdir(parents=True); target.write_text(original)
        applied = subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(patch.resolve())],
            cwd=tmp, check=True, text=True, capture_output=True)
        assert 'offset' not in applied.stdout + applied.stderr and 'fuzz' not in applied.stdout + applied.stderr
        assert target.read_text() == original.replace(old,new)
        for label,body in (('baseline',old),('patched',new)):
            text = template.replace('/* SOURCE_ERRORS */',errors).replace('/* SOURCE_POLL */',body)
            cpp = tmp / (label+'.cpp'); exe = tmp / label; cpp.write_text(text)
            fixture_hashes[label] = sha(cpp)
            subprocess.run([str(compiler),'-std=c++17','-O0','-Wall','-Wextra','-Werror',
                '-I'+str(source/'include'),'-I'+str(source/'src/core/include'),
                str(cpp),'-o',str(exe)],env=env,check=True,capture_output=True,text=True,timeout=60)
            dynamic = subprocess.run([str(READELF),'-d',str(exe)],check=True,capture_output=True,text=True).stdout
            needed[label] = [x.split('[')[1].split(']')[0] for x in dynamic.splitlines() if '(NEEDED)' in x]
            assert not any(any(x in n.lower() for x in ('hip','hsa','mscclpp','ibverbs','cuda')) for n in needed[label])
            output = subprocess.run([str(exe)],env=env,check=True,capture_output=True,text=True,timeout=10)
            assert output.stderr == ''
            records = [json.loads(x) for x in output.stdout.splitlines()]
            assert len(records) == 17 and len({x['case'] for x in records}) == 17
            for r in records:
                assert {k:v for k,v in r.items() if k!='case'} == expected(r['case'],label=='patched'), r
            rows[label] = records
    changes = [a['case'] for a,b in zip(rows['baseline'],rows['patched']) if a!=b]
    assert changes == ['pollerr_refused','pollerr_timeout','pollerr_unreachable',
        'pollerr_inprogress','pollerr_get_failure','pollerr_zero_so_error','pollerr_refused_ipv6']
    return {'schema':'MSCCLPP_POLLERR_HOST_METHOD_CPU_V1','status':'PASS',
        'cases_per_version':17,'changed_cases':changes,'source':str(source),'source_pins':PINS,
        'patch':{'path':str(patch.resolve()),'sha256':sha(patch),'zero_fuzz_offset':True},
        'compiler':{'path':str(compiler),'sha256':sha(compiler)},
        'test_source_sha256':sha(Path(__file__)),'template_sha256':sha(template_path),
        'generated_fixture_sha256':fixture_hashes,'needed':needed,'rows':rows,
        'scope':'actual old/new method and declaration/host exceptions; constructor and all syscalls stubbed',
        'network_GPU_calls':False,'initiating_live_socket_error':'UNASSIGNED',
        'package_build':'NOT_RUN','live_qualification':'NOT_RUN'}

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--patch',type=Path,default=Path(__file__).resolve().parents[1]/'0003-bootstrap-pollerr-so-error.patch')
    p.add_argument('--compiler',type=Path,default=COMPILER)
    p.add_argument('--out',type=Path)
    a=p.parse_args(); result=check(a.source,a.patch,a.compiler)
    text=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if a.out:
        with a.out.open('x') as f:f.write(text)
    else:print(text,end='')
