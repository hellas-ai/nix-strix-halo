#!/usr/bin/env python3
"""Binary SDK omits ClangConfig.cmake; import its exact assembler tools only."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('sdk',type=Path);a=p.parse_args()
for rel in ('runtime/hsa-runtime/core/runtime/trap_handler/CMakeLists.txt','runtime/hsa-runtime/core/runtime/blit_shaders/CMakeLists.txt','runtime/hsa-runtime/image/blit_src/CMakeLists.txt'):
 path=a.source/rel;s=path.read_text();found=0;lines=[]
 for l in s.splitlines():
  if l.startswith('find_package(Clang REQUIRED') or l.startswith('find_package(LLVM REQUIRED'):found+=1
  else:lines.append(l)
 assert found in (1,2),(path,found)
 targets=''
 for name in ('clang','llvm-objcopy'):
  tool=a.sdk/'lib/llvm/bin'/name;assert tool.exists()
  targets+=f'if(NOT TARGET {name})\n  add_executable({name} IMPORTED GLOBAL)\n  set_target_properties({name} PROPERTIES IMPORTED_LOCATION "{tool}")\nendif()\n'
 path.write_text(targets+'\n'+'\n'.join(lines)+'\n')
