# GLM C8 HTTP acceptance

`client.py` uses Python3.10+ standard-library HTTP and threads. It sends only
generated fixtures: arithmetic at concurrency1/2/4/8, two longer C8 groups
(uniform then mixed prompt lengths), and a streamed function call followed by
a tool-result continuation. The tool is simulated; no file is read. All
requests stream with returned usage enabled. No service, cache, configuration,
thermal setting or fleet state is changed by this client.

Run only when the separate campaign owns the workers and has admitted the
hardware. This client neither acquires that campaign's lease nor manages
thermal aborts. Requests are bounded by output limits and HTTP timeout; the
client records incomplete/failed responses without retrying them. A controller
can terminate the process for a campaign-wide deadline.

```sh
python3 pkgs/sglang/glm53/qualification/c8/test_client.py

python3 pkgs/sglang/glm53/qualification/c8/client.py \
  --base-url http://worker:30053/v1 \
  --worker-url http://worker:30053 \
  --model glm-5.3-flash-uncensored \
  --out /tmp/glm-c8-direct-unique

# Through trex: keep telemetry on the actual selected worker.
python3 pkgs/sglang/glm53/qualification/c8/client.py \
  --base-url http://trex:PORT/v1 \
  --worker-url http://worker:30053 \
  --model glm-5.3-flash-uncensored \
  --out /tmp/glm-c8-gateway-unique
```

An optional bearer token comes from `OPENAI_API_KEY` (override the environment
variable name with `--token-env`). Tokens are never written to the report.
Do not put credentials in endpoint URLs. Output directories must be new.
`--expected-revision REV` additionally requires matching server revision
readback. Servers that expose no checkpoint revision remain unverified and
fail that optional gate; a caller-supplied revision alone is never proof.

`report.json` is updated after each group. It records worker configuration
through `/get_server_info` or `/server_info`, generated response contents,
request hashes, errors, completion counts, time to first generated event and
group end-to-end throughput. Group throughput includes prefill and uses actual
returned completion counts, including reasoning tokens. It is not isolated
decode throughput. Length-capped long responses are retained; arithmetic and
tool continuation must stop naturally with exact answers.

PASS requires server readback explicitly reporting `speculative_algorithm:
null`, all fixture/protocol checks and observed worker occupancy at
the requested maximum concurrency for at least three consecutive samples
spanning one second in every long group. Gaps longer than one second break
the interval. `/v1/loads?include=core` is preferred, with `/metrics` as fallback.
Counts from multiple ranks/replicas are never added. Missing telemetry gives
`FAIL_OR_INCONCLUSIVE`; eight client threads do not prove C8. Occupancy is a
scheduler observation: attach actual decode batch/graph logs separately to
establish that eight requests decoded in the same native batch. The report
does not claim numerical equivalence, stable thermal performance or general
coding quality. Gateway routing must be fixed to the observed worker for the
report to describe the same requests.

Defaults: request timeout180s, long output cap384 tokens, two long groups.
Bounds: timeout5..600s, output64..1024 tokens, repeats1..4, concurrency1..8.
Choose fresh output paths for repeats rather than appending evidence.
