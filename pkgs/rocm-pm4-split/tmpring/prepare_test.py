"""Extract actual patched Prepare validation for CPU compile, not a mirror."""
from pathlib import Path
import sys
source=Path(sys.argv[1]).read_text();start=source.index('  const bool requires_scratch = private_wave32');end=source.index('\n  const size_t bytes',start);block=source[start:end]
block=block.replace('private_wave32 != 0 || private_wave64 != 0','scratch').replace('capabilities.encoder_family','family').replace('return HSA_STATUS_ERROR_INVALID_ARGUMENT;','return false;')
Path('validation.inc').write_text('bool valid(const std::vector<uint32_t>& words,size_t tmpring_patch_dword,bool scratch,hsa_ven_amd_graph_encoder_family_t family) {\n'+block+'\nreturn true;\n}\n')
# Scratchless Materialize retains its existing IB without queue scratch calls.
materialize=source[source.index('hsa_status_t Materialize('):]
assert materialize.index('void* ib = command_list->ib();')<materialize.index('if (command_list->requires_scratch()) {')<materialize.index('aql_queue->GetGraphScratchState(')<materialize.index('command_list->QueueIb(')
print('PASS: actual Materialize scratchless branch remains direct retained IB; hardware execution still unqualified')
