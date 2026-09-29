# Redis vector transition probe

`make benchmark-redis-transition` runs a focused Python probe against a **dedicated, empty** Redis Search instance. It does not build the Rust benchmark. Install Python `numpy` and `redis` in the selected run environment. Use one host, one `.npy` corpus, one `N`, and the same K10 query setup for all cases. The runner refuses a server with any keys or indexes before changing global settings.

```sh
make benchmark-redis-transition TRANSITION_ARGS="--cases scripts/redis_transition_arm.json --vectors /path/to/vectors.npy --n 10240 --dtype FLOAT32 --metric COSINE --host 127.0.0.1 --port 14960 --ack-dedicated-empty-instance --output-dir /path/to/new-result-directory --repetitions 3"
```

The included Arm case file compares plain HNSW, SQ8 with threshold 0, trained SQ8, and SVS with requested LVQ8. It uses threshold 10240 and expects this development server to report SVS compression as GlobalSQ8; debug output must identify its effective scalar quantization. Treat it as a pinned-runtime example: on a server reporting another compression, inspect the actual mode before changing the expected readback. HNSW M=32 and SVS graph degree 64 are starting construction settings, not a claim of equal graph structure or search effort. See the existing [SVS calibration configurations](../experiments/configurations/dbpedia-calibration.json) for steady-state comparisons. Query speed comparisons need matched recall on the full corpus; these transition probes do not measure recall.

The case file declares algorithm-specific `FT.CREATE` vector attributes and server-reported `FT.INFO` values. Shared `TYPE`, `DIM`, and `DISTANCE_METRIC` come from the dtype, NPY shape, and metric arguments. `expected_schema` keys are lowercase `FT.INFO` vector attribute names; values are compared as strings. Include every algorithm-specific attribute whose application matters. For example:

```json
{
  "cases": [
    {
      "name": "hnsw-sq8-threshold-10240",
      "algorithm": "HNSW",
      "create_params": {
        "M": 32, "EF_CONSTRUCTION": 200,
        "COMPRESSION": "SQ8", "TRAINING_THRESHOLD": 10240
      },
      "expected_schema": {
        "m": 32, "ef_construction": 200,
        "compression": "SQ8", "training_threshold": 10240
      },
      "expected_pre": "accumulating",
      "query_param": {"name": "EF_RUNTIME", "value": 400}
    }
  ]
}
```

Use `expected_pre: "ready"` for plain HNSW and SQ8 threshold 0. An accumulating case must set `TRAINING_THRESHOLD` equal to CLI `--n`; at N−1 the runner requires frontend N−1 and backend 0. A ready case must reach frontend 0, backend N−1, background 0 before timing. Full completion after the single trigger HSET requires frontend 0, backend N, background 0. For SVS-VAMANA, use `GRAPH_MAX_DEGREE` and `CONSTRUCTION_WINDOW_SIZE` in `create_params`, and `SEARCH_WINDOW_SIZE` in `query_param`. Set `expected_schema.compression` to the **reported** `FT.INFO` value, which may differ from the requested value. The raw debug samples include `BACKEND_INDEX.QUANT_BITS` to show effective SVS quantization.

The runner alternates worker and case order across repetitions. It sets `search-workers` to 0 or 4, checks the readback, sets `search-on-timeout` to `FAIL`, and restores both original settings after all probe threads stop. It creates a unique index/key prefix per case and drops only that index and its keys. Each run uploads N−1 vectors, validates the server state, starts independent PING and KNN connections, obtains at least one successful sample from each during a one-second warmup, issues one final HSET, polls `_FT.DEBUG VECSIM_INFO` through backend completion, and samples for another second. Every KNN response must contain ten distinct, existing IDs with finite scores. This corpus is a prefix of the NPY file, so **no recall claim** follows from these probes.

`transition.json` retains monotonic start/end timestamps for every probe command, the trigger HSET round trip, all VECSIM_INFO poll intervals and values, the first **observed** completed state, schema and final document/error checks, config readbacks, server/module metadata, and dataset/script SHA256. Probe sampling is closed-loop: its interval includes command service time. An observed gap or p99 is a sample statistic, not a latency bound. Observed build completion is at the end of the first successful poll; its uncertainty includes the poll interval and command time. Trigger RTT is separate from build completion. A timeout, malformed KNN result, indexing error, schema mismatch, incomplete transition, server restart, or failed config restoration exits nonzero and leaves a failure artifact. Pin the actual Redis/Search binary hashes and host CPU details alongside the artifact when reporting results.
