# HNSW SQ8 end-to-end benchmarks

Use `experiments/configurations/redis-hnsw-sq8.json` with a Redis Search build
that supports HNSW `COMPRESSION SQ8`. The Rust Redis adapter reads compression
and training threshold from `collection_params.hnsw_config`, sends them to
`FT.CREATE`, and verifies the server's `FT.INFO` attributes before uploading.
Unsupported servers fail explicitly.

## Comparison

The six configurations compare uncompressed HNSW, SQ8 with threshold 0, and SQ8
with learned-mean threshold 10240, separately for FLOAT32 and FLOAT16 input.
All use M=32, EF_CONSTRUCTION=200, upload parallelism 8 and batches of 64.
The primary file explicitly requests k=100; `redis-hnsw-sq8-k10.json` provides
matching k=10 cases for narrower ground truth. Both sweep ef=100/200/400/800/1600, and use
1/8/100 clients. These are starting settings, not claimed optimal settings.

This follows the published SVS benchmark's comparison of index memory, total
Redis memory, ingestion, throughput, latency and accuracy. It does not reproduce
the original host or every original parameter:
https://redis.io/blog/tech-dive-comprehensive-compression-leveraging-quantization-and-dimensionality-reduction/

## Dataset stages

1. First real-data check: `dbpedia-openai-100K-1536-angular`, the full registered
   100K corpus and its own ground truth (10 neighbors per query; use the k10 file).
2. Representative runs: `cohere-768-1M` and
   `dbpedia-openai-1M-1536-angular-100neighbors`.
3. The ticket's 64D and 128D datasets and additional sizes still need selection
   and registration. Synthetic low-dimensional tests must be labelled synthetic.

The registered 100K and 1M DBpedia families use different embedding models.
Do not treat their measurements as scaling results for the same corpus.
Never use `--upload-end-idx` to shrink a corpus while retaining its full-corpus
nearest-neighbor ground truth. Do not truncate or pad vectors to change dimension.
For FLOAT16, report accuracy against the dataset's original ground truth as
end-to-end quality including input conversion, and compare SQ8 with the FLOAT16
baseline separately from FLOAT32.

## Execution

Run on a dedicated Redis instance. Pin the harness commit, Redis and Search
binaries (SHA256), datasets (SHA256), CPU, compiler and worker configuration.
Use a fresh result directory per independent run. Keep only one configuration's
index/data resident when measuring total Redis memory.

Example first upload (from the repository root):

```sh
export REDIS_PORT=14960
export REDIS_QUERY_TIMEOUT=90000
./target/release/vector-db-benchmark \
  --engines-file experiments/configurations/redis-hnsw-sq8-k10.json \
  --engines redis-hnsw-float32-sq8-trained-k10 \
  --datasets dbpedia-openai-100K-1536-angular \
  --skip-search --keep-data --skip-if-exists false
```

Before search, inspect `FT.INFO idx:<config-name>` and
`_FT.DEBUG VECSIM_INFO idx:<config-name> vector` on the dedicated server:

- exact document count and no indexing errors;
- requested compression and threshold;
- `FRONTEND_INDEX.INDEX_SIZE=0`, `BACKEND_INDEX.INDEX_SIZE=dataset size`, and
  `BACKGROUND_INDEXING=0` for the completed steady-state workload;
- record `vector_index_sz_mb`, total `used_memory`, and the wait duration.

The harness's existing `wait_for_indexing` checks document indexing, not the
complete tiered vector migration. Its timeout also returns success. Therefore
its upload time alone is not proof that the SQ8 backend finished building.
Record upload plus the additional backend-drain duration as a separate quantity.
Do not time dataset downloads as ingestion.

Then run search on the validated corpus:

```sh
./target/release/vector-db-benchmark \
  --engines-file experiments/configurations/redis-hnsw-sq8-k10.json \
  --engines redis-hnsw-float32-sq8-trained-k10 \
  --datasets dbpedia-openai-100K-1536-angular \
  --skip-upload --keep-data --skip-if-exists false \
  --fail-on-dropped-queries --repetitions 1 --search-duration 30
```

Set the dedicated server's `search-on-timeout` to `FAIL`. Drop the completed
configuration's index with `FT.DROPINDEX idx:<config-name> DD` before the next
configuration, after copying results and telemetry. Do not use `FLUSHALL` on a
shared instance.

## Interpretation and regression checks

Retain each independent run, not just a best-of result. Alternate baseline/SQ8
order and run an identical-binary baseline A/A control before attributing small
throughput differences. Repeat graph construction as well as searches.

Use `mean_recall`, not `mean_precision_at_returned`, for recall@k. Require at
least k valid ground-truth neighbors in every measured query. In particular,
use k=10 for the first DBpedia 100K dataset and k=100 only with sufficiently
wide ground truth. Reject a
measurement with dropped queries, partial corpus, schema mismatch, indexing
errors or incomplete backend migration. Report both matching-parameter results
and the best measured QPS meeting each recall target (for example 0.95 and 0.99);
report an unreachable target rather than extrapolating a crossing.

Compute vector-index compression as baseline `vector_index_sz_mb` divided by
SQ8 `vector_index_sz_mb`. Report total Redis memory separately: stored HASH
vectors remain full precision. Report ingestion through backend completion,
QPS, p50/p95/p99 latency and recall@100 for each configuration.

Performance regression thresholds must be calibrated from repeated controls;
the first smoke run does not establish them. The separate transition workload
must observe below-threshold accumulation, threshold crossing and backend drain
while querying. A completed above-threshold index alone does not satisfy that
transition measurement.

## Validation

Run `make agent-check`, then `REDIS_TEST_PORT=<dedicated-port> make integration-test-no-docker`
and `REDIS_TEST_PORT=<dedicated-port> make integration-test-redis-sq8` on the
selected build host. The integration suites destructively own the dedicated
test instance; run them before loading benchmark data. The SQ8 test is a
separate opt-in because the default Redis image may predate HNSW SQ8 support.
It asserts server-reported compression and thresholds 0 and 4, and uploads
enough vectors to cross the learned threshold.
