#!/usr/bin/env python3
"""Apply five pinned feature deltas onto exact binary SDK source, no compile/GPU."""
import argparse,hashlib,json,os,re,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE='/nix/store/bnsprmrbzsp72n2mc30d2lwija6mzx44-source'
def added_hunks(patch,filename):
 s=patch.read_text();a=s.index('diff --git a/'+filename+' ');b=s.find('diff --git ',a+12);section=s[a:b if b>=0 else None]
 return ['\n'.join(l[1:] for l in h.splitlines() if l.startswith('+') and not l.startswith('+++'))+'\n' for h in re.split(r'(?m)^@@.*\n',section)[1:]]
def replace_once(path,old,new):
 s=path.read_text();assert s.count(old)==1,(str(path),old,s.count(old));path.write_text(s.replace(old,new))
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path(BASE));p.add_argument('--destination',type=Path,required=True);a=p.parse_args();assert not a.destination.exists(),'use fresh destination'
 for name in ('clr','rocr-runtime','hip'):
  dst=a.destination/'projects'/name;shutil.copytree(a.base/'projects'/name,dst,symlinks=True)
  for x in dst.rglob('*'):
   if not x.is_symlink():x.chmod(x.stat().st_mode|0o200)
 env={**os.environ,'GIT_CEILING_DIRECTORIES':str(a.destination.parent)}
 def apply(name,exclude=()):
  cmd=['git','apply','--recount',*[f'--exclude={x}' for x in exclude],'--include=projects/clr/**','--include=projects/rocr-runtime/**',str(HERE/name)]
  subprocess.run(cmd,cwd=a.destination,env=env,check=True)
 apply('e4d6d3018097-sdk-context.patch');apply('fda32a1c2087.patch')
 excluded=('projects/clr/hipamd/src/hrr/**','projects/rocr-runtime/rocrtst/suites/test_common/**','projects/rocr-runtime/runtime/hsa-runtime/hsacore.dll.def','projects/clr/rocclr/device/rocm/rocvirtual.cpp','projects/clr/rocclr/device/rocm/rocvirtual.hpp','projects/clr/rocclr/platform/command.hpp')
 apply('91155796e8e1-sdk-context.patch',excluded)
 patch=HERE/'91155796e8e1.patch'
 file='projects/clr/rocclr/device/rocm/rocvirtual.cpp';h=added_hunks(patch,file);assert len(h)==1
 path=a.destination/file;s=path.read_text();marker='// Publish one metadata-prefetch packet';i=s.index(marker);i=s.rfind('// ================================================================================================',0,i);path.write_text(s[:i]+h[0]+s[i:])
 file='projects/clr/rocclr/device/rocm/rocvirtual.hpp';h=added_hunks(patch,file);assert len(h)==1
 path=a.destination/file;old='  template <typename AqlPacket> bool dispatchGenericAqlPacket(';replace_once(path,old,h[0]+old)
 file='projects/clr/rocclr/platform/command.hpp';h=added_hunks(patch,file);assert len(h)==2
 path=a.destination/file;old='  bool owns_hw_events_ = true;\n';replace_once(path,old,old+h[0]);old='  void setOwnsHwEvents(bool owns) { owns_hw_events_ = owns; }\n';replace_once(path,old,old+'\n'+h[1])
 apply('b3f76e347fc4.patch');apply('7dda3ac6cfe6.patch')
 # Qualification-only diagnostics distinguish creation from actual replay.
 path=a.destination/'projects/clr/hipamd/src/hip_graph_internal.cpp'
 anchor='          usedPm4 = batchStatus;\n        }\n      }\n      if (!usedPm4)'
 replacement='          usedPm4 = batchStatus;\n        }\n        ClPrint(amd::LOG_INFO, amd::LOG_CODE, \"[hipGraph][PM4-GATE] packets=%zu prepared=%u replayed=%u\", flatHdrs->size(), static_cast<unsigned>(retained != nullptr), static_cast<unsigned>(usedPm4));\n      }\n      if (!usedPm4)'
 replace_once(path,anchor,replacement)
 # SDK has direct queue reservation, not fork's newer barrier-elision helper.
 path=a.destination/'projects/clr/rocclr/device/rocm/rocvirtual.cpp'
 old='  const bool requested_barrier = (header & kBarrierBit) != 0;\n  AqlSlotReservation reservation = ReserveAqlSlots(1);\n  const uint64_t index = reservation.start_slot;\n  if (requested_barrier) {\n    OptimizeStreamOrderingBarrier(header, reservation);\n  }\n  RecordAqlPacketHeader(reservation, 0, header);\n  CompleteAqlSubmission(reservation);'
 new='  // SDK-compatible conservative submission: retain materialized barrier and\n  // system acquire/release header; do not apply newer barrier-elision helpers.\n  const uint64_t index = Hsa::queue_add_write_index_screlease(gpu_queue_, 1);'
 replace_once(path,old,new)
 replace_once(path,'  writePacketToRingBuffer(slot, &packet, header, vendor_header, index);','  writePacketToRingBuffer(slot, &packet, header, vendor_header, index & queue_mask);')
 # Build shared source remains the exact SDK base, not fork/develop.
 (a.destination/'shared').symlink_to(a.base/'shared',target_is_directory=True)
 # Pin generated HIP runtime version to the installed SDK's public version.
 version=a.destination/'projects/hip/VERSION';assert [int(x) for x in version.read_text().splitlines() if x.isdigit()]==[7,15,0]
 version.write_text('#HIP_VERSION_MAJOR\n7\n#HIP_VERSION_MINOR\n15\n#HIP_VERSION_PATCH\n26333\n')
 manifest=[]
 for x in (a.destination/'projects').rglob('*'):
  if x.is_file():
   rel=x.relative_to(a.destination);base=a.base/rel
   if not base.exists() or x.read_bytes()!=base.read_bytes():manifest.append(dict(path=str(rel),sha256=hashlib.sha256(x.read_bytes()).hexdigest(),base_sha256=hashlib.sha256(base.read_bytes()).hexdigest() if base.exists() else None))
 (a.destination/'port-manifest.json').write_text(json.dumps(dict(base_commit='6b0e43f341195e203754e08f850e437ff2fc09f9',feature_head='7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1',files=manifest),indent=2));print(json.dumps(dict(changed_files=len(manifest),destination=str(a.destination)),indent=2))
if __name__=='__main__':main()
