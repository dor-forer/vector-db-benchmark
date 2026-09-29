#!/usr/bin/env python3
"""Measure one-vector threshold transitions on an otherwise empty Redis Search server."""

import argparse
import hashlib
import json
import math
import platform
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import redis
from redis.backoff import NoBackoff
from redis.retry import Retry


def pairs(value):
    """Decode RESP2 flat maps and RESP3 maps, including nested VECSIM_INFO maps."""
    def key(v):
        return (v.decode() if isinstance(v, bytes) else str(v)).lower()

    if isinstance(value, dict):
        return {key(k): pairs(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) % 2 == 0 and all(isinstance(value[i], (str, bytes)) for i in range(0, len(value), 2)):
            return {key(value[i]): pairs(value[i + 1]) for i in range(0, len(value), 2)}
        return [pairs(v) for v in value]
    return value.decode() if isinstance(value, bytes) else value


def integer(obj, *path):
    for key in path:
        obj = obj[key]
    return int(obj)


def vector_state(conn, index):
    raw = pairs(conn.execute_command("_FT.DEBUG", "VECSIM_INFO", index, "vector"))
    # Require all three server-reported counters; no default counts.
    return {
        "frontend": integer(raw, "frontend_index", "index_size"),
        "backend": integer(raw, "backend_index", "index_size"),
        "background": integer(raw, "background_indexing"),
        "raw": raw,
    }


def ft_info(conn, index, docs):
    info = pairs(conn.execute_command("FT.INFO", index))
    assert integer(info, "num_docs") == docs, f"FT.INFO num_docs != {docs}: {info.get('num_docs')}"
    failures = info.get("hash_indexing_failures", info.get("indexing_failures"))
    assert failures is not None and int(failures) == 0, f"indexing failures unavailable or nonzero: {failures}"
    return info


def schema(info, expected):
    attrs = info["attributes"]
    field = next((a for a in attrs if a.get("identifier") == "vector"), None)
    assert field is not None, f"FT.INFO vector attribute absent: {attrs}"
    for key, wanted in expected.items():
        actual = field.get(key.lower())
        assert actual is not None and str(actual).upper() == str(wanted).upper(), (
            f"FT.INFO vector {key}={actual!r}, expected {wanted!r}"
        )
    return field


def await_state(conn, index, docs, expected, deadline_s, samples, interval_s=0.05):
    deadline = time.monotonic() + deadline_s
    while True:
        start = time.monotonic_ns()
        state = vector_state(conn, index)
        end = time.monotonic_ns()
        samples.append({"start_ns": start, "end_ns": end, **state})
        if all(state[k] == v for k, v in expected.items()):
            return end
        if time.monotonic() >= deadline:
            raise TimeoutError(f"vector state did not reach {expected}; last={state}")
        time.sleep(interval_s)


def query(conn, index, blob, query_param, prefix, docs):
    extra = f" {query_param['name']} {int(query_param['value'])}" if query_param else ""
    expression = f"*=>[KNN 10 @vector $vec{extra} AS score]"
    result = conn.execute_command(
        "FT.SEARCH", index, expression, "PARAMS", 2, "vec", blob,
        "SORTBY", "score", "RETURN", 1, "score", "LIMIT", 0, 10,
        "TIMEOUT", 0, "DIALECT", 2,
    )
    assert len(result) == 21 and int(result[0]) >= 10, f"unexpected K10 shape: {result[:3]}"
    seen = set()
    for i in range(1, len(result), 2):
        identifier = result[i]
        assert identifier.startswith(prefix) and identifier not in seen, f"invalid or duplicate id: {identifier}"
        ordinal = int(identifier[len(prefix):])
        assert 0 <= ordinal < docs, f"id outside uploaded corpus: {identifier}"
        seen.add(identifier)
        fields = pairs(result[i + 1])
        assert math.isfinite(float(fields["score"])), f"non-finite score: {fields}"


def probe(conn, label, stop, records, errors, index, blob, query_param, prefix, docs, interval_s):
    try:
        while not stop.is_set():
            start = time.monotonic_ns()
            try:
                if label == "ping":
                    assert conn.ping() is True
                else:
                    query(conn, index, blob, query_param, prefix, docs)
                error = None
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            end = time.monotonic_ns()
            records.append({"kind": label, "start_ns": start, "end_ns": end, "error": error})
            if error:
                errors.append({"kind": label, "error": error})
                stop.set()
                break
            stop.wait(interval_s)  # closed-loop sampling; delays include service time
    finally:
        conn.close()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def validate_cases(path, n):
    cases = json.loads(path.read_text())["cases"]
    assert cases and len({c["name"] for c in cases}) == len(cases), "case names must be unique"
    for case in cases:
        assert case["algorithm"] in ("HNSW", "SVS-VAMANA")
        assert case["expected_pre"] in ("accumulating", "ready")
        assert isinstance(case["create_params"], dict) and isinstance(case["expected_schema"], dict)
        assert case["expected_schema"], "expected_schema is required for server readback"
        attrs = {k.upper(): v for k, v in case["create_params"].items()}
        assert not {"TYPE", "DIM", "DISTANCE_METRIC"} & attrs.keys(), "common attributes belong to CLI"
        threshold = attrs.get("TRAINING_THRESHOLD")
        if case["expected_pre"] == "accumulating":
            assert threshold == n, "accumulating case threshold must equal N"
        if threshold is not None:
            assert case["expected_schema"].get("training_threshold") is not None
        param = case.get("query_param")
        assert param is None or (param["name"] in ("EF_RUNTIME", "SEARCH_WINDOW_SIZE") and int(param["value"]) > 0)
    return cases


def run_case(controller, factory, case, vectors, n, dtype, metric, prefix, deadline_s,
             warmup_s, post_s, interval_s, result):
    index = "idx:" + prefix
    key_prefix = prefix + ":"
    case_result = {"case": case["name"], "index": index, "key_prefix": key_prefix,
                   "samples": [], "probes": [], "phase": "create"}
    result["runs"].append(case_result)
    attrs = ["TYPE", dtype, "DIM", str(vectors.shape[1]), "DISTANCE_METRIC", metric]
    for key, value in case["create_params"].items():
        attrs.extend((key, str(value)))
    expected_schema = {"type": "VECTOR", "algorithm": case["algorithm"],
                       "data_type": dtype, "dim": vectors.shape[1],
                       "distance_metric": metric, **case["expected_schema"]}
    created = False
    try:
        controller.execute_command("FT.CREATE", index, "ON", "HASH", "PREFIX", 1, key_prefix,
                                   "SCHEMA", "vector", "VECTOR", case["algorithm"], len(attrs), *attrs)
        created = True
        case_result["schema"] = schema(pairs(controller.execute_command("FT.INFO", index)),
                                       expected_schema)
        case_result["phase"] = "upload"
        for i in range(n - 1):
            controller.hset(f"{key_prefix}{i}", "vector", vectors[i].tobytes())
        # Wait for normal document indexing before asserting the vector layout.
        deadline = time.monotonic() + deadline_s
        while True:
            try:
                info = ft_info(controller, index, n - 1)
                break
            except AssertionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
        schema(info, expected_schema)
        expected_pre = ({"frontend": n - 1, "backend": 0}
                        if case["expected_pre"] == "accumulating" else
                        {"frontend": 0, "backend": n - 1, "background": 0})
        case_result["phase"] = "pre_state"
        case_result["pre_complete_ns"] = await_state(controller, index, n - 1, expected_pre,
                                                       deadline_s, case_result["samples"])
        # Every timed socket is independent and created before the trigger.
        stop = threading.Event()
        errors = []
        probes = []
        connections = []
        blob = vectors[1].tobytes()
        try:
            for label in ("ping", "knn"):
                conn = factory()
                connections.append(conn)
                assert conn.ping(), "probe connection did not establish"
                thread = threading.Thread(target=probe, args=(conn, label, stop, case_result["probes"],
                                                              errors, index, blob, case.get("query_param"),
                                                              key_prefix, n, interval_s), daemon=True)
                thread.start()
                probes.append(thread)
            case_result["phase"] = "warmup"
            case_result["warmup_start_ns"] = time.monotonic_ns()
            time.sleep(warmup_s)
            assert not errors, f"warmup probe failed: {errors}"
            assert any(p["kind"] == "ping" for p in case_result["probes"])
            assert any(p["kind"] == "knn" for p in case_result["probes"])
            case_result["phase"] = "trigger"
            start = time.monotonic_ns()
            controller.hset(f"{key_prefix}{n - 1}", "vector", vectors[n - 1].tobytes())
            end = time.monotonic_ns()
            case_result["trigger"] = {"start_ns": start, "end_ns": end, "rtt_ns": end - start}
            assert not errors, f"probe failed during trigger: {errors}"
            case_result["phase"] = "drain"
            complete = await_state(controller, index, n, {"frontend": 0, "backend": n,
                                                            "background": 0}, deadline_s,
                                   case_result["samples"])
            case_result["observed_completion_ns"] = complete
            case_result["phase"] = "post"
            time.sleep(post_s)
            case_result["post_end_ns"] = time.monotonic_ns()
            assert not errors, f"probe failed: {errors}"
        finally:
            stop.set()
            try:
                for thread in probes:
                    thread.join()
            finally:
                for conn in connections:
                    conn.close()
            assert all(not thread.is_alive() for thread in probes), "probe thread did not stop"
        case_result["phase"] = "validation"
        case_result["final_info"] = ft_info(controller, index, n)
        case_result["final_state"] = vector_state(controller, index)
        case_result["effective_backend_quant_bits"] = (
            case_result["final_state"]["raw"].get("backend_index", {}).get("quant_bits")
        )
        assert all(case_result["final_state"][k] == v for k, v in
                   {"frontend": 0, "backend": n, "background": 0}.items())
        assert not any(p["error"] for p in case_result["probes"]), "probe errors in raw log"
        case_result["phase"] = "complete"
    finally:
        try:
            if created:
                controller.execute_command("FT.DROPINDEX", index, "DD")
        finally:
            # DD can leave hashes rejected by indexing; only this run's prefix is owned.
            while True:
                keys = list(controller.scan_iter(match=key_prefix + "*", count=1000))
                if not keys:
                    break
                for key in keys:
                    assert key.startswith(key_prefix), f"SCAN returned another run's key: {key}"
                    controller.delete(key)
            assert controller.dbsize() == 0 and not controller.execute_command("FT._LIST"), (
                "dedicated instance not empty after owned-index cleanup"
            )
            case_result["cleanup_ns"] = time.monotonic_ns()


def main():
    if not __debug__:
        raise RuntimeError("optimized Python disables benchmark validation assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True, help="2D .npy corpus; first N rows are uploaded")
    parser.add_argument("--n", type=int, required=True, help="one trigger HSET inserts vector N")
    parser.add_argument("--dtype", choices=("FLOAT32", "FLOAT16"), required=True)
    parser.add_argument("--metric", choices=("L2", "COSINE", "IP"), default="COSINE")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ack-dedicated-empty-instance", action="store_true", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", choices=(0, 4), default=[0, 4])
    parser.add_argument("--deadline-s", type=float, default=180.0)
    parser.add_argument("--probe-interval-ms", type=float, default=10.0)
    parser.add_argument("--warmup-s", type=float, default=1.0)
    parser.add_argument("--post-s", type=float, default=1.0)
    args = parser.parse_args()
    assert args.n >= 11 and args.repetitions > 0 and args.deadline_s > 0
    assert args.warmup_s >= 1 and args.post_s >= 1 and args.probe_interval_ms >= 0
    cases = validate_cases(args.cases, args.n)
    raw = np.load(args.vectors, mmap_mode="r", allow_pickle=False)
    assert raw.ndim == 2 and raw.shape[0] >= args.n and np.issubdtype(raw.dtype, np.floating)
    dtype = np.dtype("<f4" if args.dtype == "FLOAT32" else "<f2")
    vectors = np.ascontiguousarray(raw[:args.n], dtype=dtype)
    assert np.isfinite(vectors).all(), "converted corpus has non-finite values"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset_hash = hashlib.sha256()
    with args.vectors.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            dataset_hash.update(chunk)
    dataset_sha256 = dataset_hash.hexdigest()
    data = {"status": "running", "arguments": vars(args), "cases": cases, "runs": [],
            "started_wall_utc": time.time(), "monotonic_origin_ns": time.monotonic_ns(),
            "dataset_sha256": dataset_sha256,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "client_host": platform.uname()._asdict(), "numpy_version": np.__version__,
            "redis_py_version": redis.__version__}
    artifact = args.output_dir / "transition.json"
    dump(artifact, data)

    def factory():
        return redis.Redis(host=args.host, port=args.port, protocol=2, decode_responses=True,
                           socket_timeout=args.deadline_s + 10, socket_connect_timeout=5,
                           retry=Retry(NoBackoff(), 0), retry_on_timeout=False,
                           single_connection_client=True)

    conn = factory()
    original = {}
    try:
        assert conn.ping() and conn.dbsize() == 0, "dedicated server must start with zero keys"
        assert not conn.execute_command("FT._LIST"), "dedicated server must start with zero indexes"
        server = conn.info("server")
        run_id = server["run_id"]
        data["server_info"] = server
        data["module_list"] = pairs(conn.execute_command("MODULE", "LIST"))
        for key in ("search-workers", "search-on-timeout"):
            config = conn.config_get(key)
            assert len(config) == 1, f"CONFIG GET {key} returned {config}"
            original[key] = next(iter(config.values()))
        data["original_config"] = original.copy()
        conn.config_set("search-on-timeout", "FAIL")
        assert str(next(iter(conn.config_get("search-on-timeout").values()))).upper() == "FAIL"
        order = []
        for rep in range(args.repetitions):
            workers = args.workers if rep % 2 == 0 else list(reversed(args.workers))
            for worker in workers:
                case_order = cases if (rep + worker // 4) % 2 == 0 else list(reversed(cases))
                order.extend((rep, worker, case) for case in case_order)
        data["order"] = [{"repetition": rep, "workers": worker, "case": case["name"]}
                         for rep, worker, case in order]
        for rep, worker, case in order:
            assert conn.dbsize() == 0 and not conn.execute_command("FT._LIST"), "server not empty between cases"
            assert conn.info("server")["run_id"] == run_id, "server restarted"
            conn.config_set("search-workers", worker)
            actual = next(iter(conn.config_get("search-workers").values()))
            assert int(actual) == worker, f"search-workers readback {actual} != {worker}"
            prefix = "transition:" + uuid.uuid4().hex
            data["current_run"] = {"repetition": rep, "workers": worker, "case": case["name"]}
            run_case(conn, factory, case, vectors, args.n, args.dtype, args.metric, prefix,
                     args.deadline_s, args.warmup_s, args.post_s,
                     args.probe_interval_ms / 1000, data)
            data["runs"][-1].update(repetition=rep, workers=worker)
            data.pop("current_run")
            assert conn.info("server")["run_id"] == run_id, "server restarted"
            dump(artifact, data)
        data["status"] = "complete"
    except BaseException as exc:
        data["status"] = "failed"
        data["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        restore_errors = []
        restored = {}
        for key, value in reversed(list(original.items())):
            try:
                conn.config_set(key, value)
                restored[key] = next(iter(conn.config_get(key).values()))
                if str(restored[key]).upper() != str(value).upper():
                    raise RuntimeError(f"{key} restored as {restored[key]!r}, expected {value!r}")
            except Exception as exc:
                restore_errors.append(f"{key}: {type(exc).__name__}: {exc}")
        data["restored_config"] = restored
        if restore_errors:
            data["status"] = "failed"
            data["restore_errors"] = restore_errors
        data["finished_wall_utc"] = time.time()
        try:
            dump(artifact, data)
        finally:
            conn.close()
        if restore_errors:
            raise RuntimeError(f"configuration restore failed: {restore_errors}")


if __name__ == "__main__":
    main()
