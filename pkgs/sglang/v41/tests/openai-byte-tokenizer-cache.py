"""CPU-only ByteLevel/SentencePiece and OpenAI logprob-byte regression."""
import gc
import ast
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import weakref

from tokenizers import Tokenizer, decoders, models

from sglang.srt.entrypoints.openai import utils

# Execute the installed serving_chat method body without importing its GPU
# scheduler dependencies in this CPU package check.
source = utils.__file__.replace('utils.py', 'serving_chat.py')
tree = ast.parse(open(source).read(), filename=source)
owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OpenAIServingChat')
method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_build_token_logprobs_from_raw')
method.decorator_list = []
ns = {}

@dataclass
class TopLogprob:
    token: str
    bytes: list[int]
    logprob: float

@dataclass
class ChatCompletionTokenLogprob:
    token: str
    bytes: list[int]
    logprob: float
    top_logprobs: list[TopLogprob]

exec(compile(ast.Module(body=[method], type_ignores=[]), source, 'exec'),
     {'Any': Any, 'ChatCompletionTokenLogprob': ChatCompletionTokenLogprob,
      'TopLogprob': TopLogprob, 'token_id_to_bytes': utils.token_id_to_bytes}, ns)


class BackendWrapper:
    def __init__(self, backend):
        self.backend_tokenizer = backend
        self.vocab_calls = 0

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def added_tokens_decoder(self):
        return self.backend_tokenizer.get_added_tokens_decoder()

    def convert_ids_to_tokens(self, token_id):
        return self.backend_tokenizer.id_to_token(token_id)

    def get_vocab(self):
        self.vocab_calls += 1
        return self.backend_tokenizer.get_vocab(with_added_tokens=True)


def logprobs(tokenizer, output, top):
    owner = SimpleNamespace(tokenizer_manager=SimpleNamespace(tokenizer=tokenizer))
    return ns['_build_token_logprobs_from_raw'](owner, output, top)


backend = Tokenizer(models.BPE(vocab={'Ã': 0, '©': 1}, merges=[]))
backend.decoder = decoders.ByteLevel()
backend.add_special_tokens(['é'])
byte = BackendWrapper(backend)
rows = logprobs(byte, [(-0.1, 0, '�')], [[(-0.2, 1, '�'), (-0.3, 2, 'é')]])
assert rows[0].bytes == [0xC3]
assert rows[0].top_logprobs[0].bytes == [0xA9]
assert rows[0].top_logprobs[1].bytes == [0xC3, 0xA9]
assert byte.vocab_calls == 0
assert utils._is_byte_level_tokenizer(byte) is True
# Added-token membership follows backend changes without stale ID caching.
backend.add_special_tokens(['à'])
assert utils.token_id_to_bytes(byte, 3) is None
byte_backend = Tokenizer(models.BPE(vocab={'é': 0}, merges=[]))
byte_backend.decoder = decoders.ByteLevel()
promoted = BackendWrapper(byte_backend)
assert utils.token_id_to_bytes(promoted, 0) == [0xE9]
byte_backend.add_special_tokens(['é'])
assert len(promoted) == 1
assert utils.token_id_to_bytes(promoted, 0) is None

sp_backend = Tokenizer(models.Unigram(vocab=[('▁é', 0.0)], unk_id=0))
sp_backend.decoder = decoders.Metaspace()
sp = BackendWrapper(sp_backend)
sp_row = logprobs(sp, [(-0.1, 0, 'é')], [[(-0.2, 0, 'é')]])[0]
assert sp_row.bytes == [0xC3, 0xA9]
assert sp_row.top_logprobs[0].bytes == [0xC3, 0xA9]
assert sp.vocab_calls == 0
assert utils._is_byte_level_tokenizer(sp) is False
sp_seq_backend = Tokenizer(models.Unigram(vocab=[('é', 0.0)], unk_id=0))
sp_seq_backend.decoder = decoders.Sequence([decoders.Metaspace()])
sp_seq = BackendWrapper(sp_seq_backend)
assert utils._is_byte_level_tokenizer(sp_seq) is False
assert logprobs(sp_seq, [(-0.1, 0, 'é')], None)[0].bytes == [0xC3, 0xA9]

class Legacy:
    def __init__(self):
        self.vocab_calls = 0
    def get_vocab(self):
        self.vocab_calls += 1
        return {'▁é': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    def convert_ids_to_tokens(self, token_id):
        return ['▁é', 'a', 'b', 'c', 'd', 'e'][token_id]

legacy = Legacy()
assert utils._is_byte_level_tokenizer(legacy) is False
assert utils._is_byte_level_tokenizer(legacy) is False
assert legacy.vocab_calls == 1
ref = weakref.ref(legacy)
del legacy
gc.collect()
assert ref() is None
assert len(utils._BYTE_LEVEL_TOKENIZERS) >= 2
print('ByteLevel fragments, added Unicode promotion, nested Metaspace, legacy negative cache and GC: PASS')
