#!/usr/bin/env python3
"""Apply package patches and CPU-compile actual Env parser/constructor; no package/GPU/RDMA build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

SOURCE=Path('/nix/store/ipfjfbhlpa730gp1g040mfqsx8fwvmw3-source')
CXX=Path('/nix/store/061ybr50r1m2jnvih4vvpgmgwb37df3r-gcc-wrapper-12.2.0/bin/c++')
PACKAGE=Path(__file__).resolve().parents[1]
PINS={
    'include/mscclpp/env.hpp':'2b225ddddb359734f46d643789b0a29ec9122f0bdfa91f2af0e4ffd49869feaf',
    'include/mscclpp/errors.hpp':'0ac256131003a82137224a786f203f3f8ede169a6e95b21b60d58a7615628d41',
    'include/mscclpp/core.hpp':'9bdf0045e54166d2f300d1a72a0341aea68abc18ee0ecc7ab9940bcaf62b4fc1',
    'src/core/env.cpp':'262c2ce984c713d7462e0de07743913134f373f9950ca28111888aecabf30e11',
    'src/core/ib.cc':'3cc6bd43895778edb92d12479ffa200aa198b38c392c78fcd388ee4c2eb23fb1',
    'src/core/errors.cc':'9bd8df5f72c7c1f13f59d035314946d08d7babab41e421cc2fd6e940440f5791',
    'CMakeLists.txt':'32e03e06ab5c99acddf316b3b762415d2c0224d8379964ab96074a1d92553bff',
    'src/core/gpu_utils.cc':'dbe1c03055302b9596e0da0fbbfa8c6f5d30757005b5baece75d3ebd0e6f00c7',
}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command(argv,**kwargs):return subprocess.run(argv,check=True,text=True,capture_output=True,timeout=60,**kwargs)


def check(source,cxx):
    assert all(sha(source/name)==pin for name,pin in PINS.items()),'Wrong upstream source'
    names=('0001-rocm-host-memory-for-ib-gfx1151.patch','0002-ib-traffic-class-env.patch')
    nix=(PACKAGE/'default.nix').read_text()
    assert all(nix.count('./'+name)==1 for name in names)
    assert nix.index('./'+names[0])<nix.index('./'+names[1])
    with tempfile.TemporaryDirectory(prefix='mscclpp-traffic-class-cpu-') as temporary:
        root=Path(temporary)
        for name in PINS:
            dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
        for name in names:
            result=command(['patch','--batch','--fuzz=0','-p1','-i',str(PACKAGE/name)],cwd=root)
            assert 'fuzz' not in result.stdout.lower() and 'offset' not in result.stdout.lower()
        ib=(root/'src/core/ib.cc').read_text()
        expression='qp_attr.ah_attr.grh.traffic_class = static_cast<uint8_t>(env()->ibTrafficClass);'
        assert ib.count(expression)==1
        assert ib.replace(expression,'qp_attr.ah_attr.grh.traffic_class = 0;')==(source/'src/core/ib.cc').read_text()
        assert 'if (info.linkLayer == IBV_LINK_LAYER_ETHERNET || info.isGrh)' in ib and 'qp_attr.ah_attr.sl = 0;' in ib
        assert (root/'include/mscclpp/core.hpp').read_bytes()==(source/'include/mscclpp/core.hpp').read_bytes()
        # Keep actual generic error methods, avoiding GPU-related error definitions/includes.
        errors=(root/'src/core/errors.cc').read_text()
        generic=errors[errors.index('std::string errorToString'):errors.index('MSCCLPP_API_CPP SysError::SysError')]
        cpu=root/'cpu';cpu.mkdir()
        (cpu/'logger.hpp').write_text('#pragma once\n#define INFO(...) ((void)0)\n')
        translation=cpu/'check.cpp'
        translation.write_text('#include <mscclpp/errors.hpp>\n#define MSCCLPP_API_CPP\nnamespace mscclpp {\n'+generic+
            '\n}\n#include "'+str(root/'src/core/env.cpp')+'"\n#include "'+str(PACKAGE/'tests/ib-traffic-class.cpp')+'"\n')
        binary=cpu/'check'
        command([str(cxx),'-std=c++17','-O2','-Wall','-Wextra','-Werror','-Wno-unused-parameter',
                 '-I',str(root/'include'),'-I',str(cpu),str(translation),'-o',str(binary)])
        parser=json.loads(command([str(binary)]).stdout)
        assert parser==dict(status='PASS_ACTUAL_PARSER',unset=1,all_byte_values=256,leading_zero_cases=4,invalid_cases=28)
        constructors=[]
        for value,expected in [('UNSET',0),('0',0),('106',106),('255',255),('000106',106),('',None),('256',None),('106x',None),(' 106',None)]:
            row=json.loads(command([str(binary),'constructor',value]).stdout)
            assert row==({'status':'REJECT_INVALID_USAGE'} if expected is None else {'status':'ACCEPT','value':expected,'cached_once':True})
            constructors.append(dict(input_class='UNSET' if value=='UNSET' else 'VALID' if expected is not None else 'INVALID',expected_value=expected,**row))
        return dict(status='PASS_PATCH_APPLICATION_AND_ACTUAL_CPU_PARSER',upstream_source=str(source),upstream_pins=PINS,
            patches={name:sha(PACKAGE/name) for name in names},package_source_sha256=sha(PACKAGE/'default.nix'),
            validation_source={p.name:sha(p) for p in [Path(__file__),PACKAGE/'tests/ib-traffic-class.cpp']},
            patched_source={name:sha(root/name) for name in ['include/mscclpp/env.hpp','src/core/env.cpp','src/core/ib.cc']},
            parser=parser,constructor_cases=constructors,compiler=str(cxx),compiler_sha256=sha(cxx),
            actual_Env_cpp_used=True,actual_generic_error_methods_used=True,logger_macro_only_stubbed=True,
            QP_assignment='Source-checked only; no ibv_modify_qp execution',C_ABI='Unchanged',
            CXX_ABI='Env gains a public field; rebuild C++ consumers',GPU_RDMA_or_package_build=False,
            scope='No GPU/RDMA initialization, network, services, package build, or existing runtime fixtures modified.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=SOURCE)
    ap.add_argument('--cxx',type=Path,default=CXX);ap.add_argument('--out',type=Path);args=ap.parse_args()
    if args.out:
        assert not any(p.is_symlink() for p in (args.out,*args.out.parents)) and not args.out.exists()
    value=check(args.source,args.cxx)
    if args.out:
        with args.out.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(value,sort_keys=True))
