#include <cassert>
#include <cstdio>
#include <limits>
#include "core/inc/amd_graph_command_encoder.h"
#include "validation.inc"
using namespace rocr::graph;
unsigned writes(const std::vector<uint32_t>& w) {
 unsigned count=0;
 for(size_t i=0;i<w.size();) {
  assert((w[i]>>30)==3);unsigned n=((w[i]>>16)&0x3fff)+1,op=(w[i]>>8)&0xff;assert(i+1+n<=w.size());
  if(op==0x76 && w[i+1]==0x218){++count;assert(n==2);}
  i+=1+n;
 }
 return count;
}
void test(unsigned count,unsigned scratch_at,unsigned scratch_again) {
 Gfx11CommandEncoder e;Gfx11KernelImage image{0x10000,0x11,0x22,0x33,0x408,0};
 for(unsigned i=0;i<count;++i){hsa_kernel_dispatch_packet_t p{};p.workgroup_size_x=256;p.workgroup_size_y=p.workgroup_size_z=1;p.grid_size_x=256;p.grid_size_y=p.grid_size_z=1;p.kernarg_address=reinterpret_cast<void*>(0x20000);p.private_segment_size=(i==scratch_at||i==scratch_again)?736:0;assert(e.Append(p,image,0,HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));}
 e.Finish();bool scratch=scratch_at<count||scratch_again<count;assert(e.dispatch_count()==count);assert(writes(e.words())==(scratch?1:0));assert(valid(e.words(),e.tmpring_patch_dword(),scratch,HSA_VEN_AMD_GRAPH_ENCODER_GFX11));
 if(scratch){auto w=e.words();assert(e.tmpring_patch_dword()<w.size());w[e.tmpring_patch_dword()]=0x12345001;assert(writes(w)==1);assert(w[e.tmpring_patch_dword()]==0x12345001);}
 else {assert(e.tmpring_patch_dword()==std::numeric_limits<size_t>::max());assert(!valid(e.words(),0,false,HSA_VEN_AMD_GRAPH_ENCODER_GFX11));assert(!valid(e.words(),e.tmpring_patch_dword(),true,HSA_VEN_AMD_GRAPH_ENCODER_GFX11));assert(!valid(e.words(),e.tmpring_patch_dword(),false,HSA_VEN_AMD_GRAPH_ENCODER_GFX12));}
}
int main(){test(1,99,99);test(16,99,99);test(16,0,15);test(16,7,15);test(16,15,99);puts("PASS: scratchless1/16 encode and pass actual Prepare validation; mixed first/middle/last preserve exactly one queue relocation");}
