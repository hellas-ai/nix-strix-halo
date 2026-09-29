"""Serve a tiny local model and compare greedy API output with PyTorch."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def request(base, route, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        base + route, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix="vllm-metal-serve-") as directory:
        root = Path(directory)
        vocabulary = {f"token{i}": i for i in range(64)}
        tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="token0"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="token0", bos_token="token1", eos_token="token2", pad_token="token0",
        )
        tokenizer.save_pretrained(root)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=64, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=256, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )).eval()
        model.save_pretrained(root)
        token_ids = [3, 4, 5]
        expected_ids = []
        with torch.no_grad():
            for _ in range(8):
                token = int(model(torch.tensor([token_ids])).logits[0, -1].argmax())
                expected_ids.append(token)
                token_ids.append(token)
        expected = tokenizer.convert_ids_to_tokens(expected_ids)

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        with (root / "server.log").open("w+") as log:
            process = subprocess.Popen(
                [
                    "vllm", "serve", str(root), "--served-model-name", "smoke",
                    "--host", "127.0.0.1", "--port", str(port),
                    "--max-model-len", "128", "--max-num-seqs", "1",
                    "--gpu-memory-utilization", "0.05", "--dtype", "float16",
                    "--no-enable-prefix-caching", "--enforce-eager",
                ],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            )
            try:
                deadline = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"vLLM exited with {process.returncode}")
                    try:
                        models = request(base, "/v1/models")
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() > deadline:
                            raise TimeoutError("vLLM did not become ready within 180 seconds")
                        time.sleep(1)
                assert models["data"][0]["id"] == "smoke", models
                payload = {
                    "model": "smoke", "prompt": "token3 token4 token5",
                    "temperature": 0, "max_tokens": 8, "ignore_eos": True,
                    "skip_special_tokens": False, "seed": 42,
                }
                for _ in range(2):
                    response = request(base, "/v1/completions", payload)
                    actual = response["choices"][0]["text"].split()
                    assert actual == expected, {"expected": expected, "actual": actual}
                    assert response["usage"]["completion_tokens"] == 8, response
                print(json.dumps({
                    "model": "synthetic two-layer Llama", "requests": 2,
                    "reference": "PyTorch greedy", "token_ids": expected_ids,
                    "verified": True,
                }, sort_keys=True))
            except BaseException:
                log.flush()
                log.seek(0)
                print(log.read(), flush=True)
                raise
            finally:
                # vLLM starts worker processes. Reap the whole test-owned group,
                # including when startup, a request, or an assertion fails.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    main()
