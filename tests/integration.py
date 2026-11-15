#!/usr/bin/env python3
"""Run the inspected public env/scorer directly; never invoke the official CLI.

Only the test parent knows fixture/reference paths. The bot receives the brief
and env URL alone. Every env binds port 0 and writes into a new temporary tree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KIT = Path("/tmp/makefun/free-model-aggregator-01-kit")
BASE_ENV = {"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}


def run_checked(command, env, timeout=30):
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name} failed ({result.returncode}): " + result.stderr.decode()[:1000])
    return result.stdout


def stop_owned(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def wait_ready(process, ready):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("env exited before ready")
        if ready.exists():
            info = json.loads(ready.read_text())
            url = f'http://127.0.0.1:{info["port"]}'
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url + "/work", timeout=2) as response:
                assert isinstance(json.load(response)["done"], bool)
            return url
        time.sleep(0.02)
    raise RuntimeError("env readiness deadline exceeded")


def one_pass(kit, fixture, target):
    target.mkdir(parents=True)
    state = target / "state"
    env = {**BASE_ENV, "MAKEFUN_ENV_HOST": "127.0.0.1", "MAKEFUN_ENV_PORT": "0",
           "MAKEFUN_ENV_DATA_DIR": str(fixture / "input/env"), "MAKEFUN_STATE_DIR": str(state)}
    with (target / "env.log").open("wb") as log:
        process = subprocess.Popen(["node", str(kit / "program/env/server.mjs")], cwd=ROOT,
                                   env=env, stdout=log, stderr=log)
        try:
            url = wait_ready(process, state / "ready")
            bot_env = {**BASE_ENV, "MAKEFUN_ENV_URL": url, "MAKEFUN_PROMPT_FILE": str(fixture / "input/prompt.txt")}
            catalog_bytes = run_checked([sys.executable, "main.py", "--list"], bot_env)
            catalog = json.loads(catalog_bytes)
            assert "key" not in {k for p in catalog["providers"] for k in p}
            run_checked([sys.executable, "-P", "-s", "main.py"], bot_env, timeout=150)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url + "/work", timeout=2) as response:
                assert json.load(response) == {"done": True}, "entry exited before work finished"
            again = run_checked([sys.executable, "main.py", "--list"], bot_env)
            assert again == catalog_bytes, "--list changed after serving work"
        finally:
            stop_owned(process)
    assert process.returncode == 0, "env export failed"
    exported = state / "export/state.json"
    document = json.loads(exported.read_text())
    assert document["catalog"] == catalog, "registered catalog differs from --list"
    # Also check exact compact serialization, independently of env normalization.
    compact = json.dumps(catalog, ensure_ascii=False, separators=(",", ":")) + "\n"
    assert compact.encode() == catalog_bytes
    scoring_env = {**BASE_ENV, "MAKEFUN_OUTPUT_DIR": str(exported.parent),
                   "MAKEFUN_REFERENCE_DIR": str(fixture / "reference"), "MAKEFUN_SCORES_DIR": str(target / "scores")}
    run_checked(["node", str(kit / "program/scorer/score.mjs")], scoring_env)
    score = json.loads((target / "scores/scores.json").read_text())
    return score, exported.read_bytes()


def generate(kit, parent, label):
    seed_dir, fixture = parent / (label + "-seed"), parent / label
    seed_dir.mkdir()
    # Public, reproducible extra practice seeds. Never passed to the entry.
    seed = hashlib.sha256(("free-router-public-regression/" + label).encode()).hexdigest()
    (seed_dir / "seed.json").write_text(json.dumps({"schema": "agent-fun/instance-seed/v1", "seed": seed, "purpose": "public practice"}))
    run_checked(["node", str(kit / "program/generator/draw.mjs")],
                {**BASE_ENV, "MAKEFUN_INPUT_DIR": str(seed_dir), "MAKEFUN_OUTPUT_DIR": str(fixture)})
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--extra", type=int, default=0)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1 or args.extra < 0:
        parser.error("repeat must be positive and extra nonnegative")
    kit = args.kit.resolve()
    for program in ("env/server.mjs", "scorer/score.mjs", "generator/draw.mjs"):
        assert (kit / "program" / program).is_file()
    parent = Path(tempfile.mkdtemp(prefix="router-tests-", dir="/tmp/makefun"))
    passed = False
    results = []
    try:
        fixtures = [(f"trial-{i}", kit / f"practice/trial-{i}") for i in range(3)]
        fixtures += [(f"extra-{i}", generate(kit, parent, f"extra-{i}")) for i in range(args.extra)]
        for name, fixture in fixtures:
            baseline, first_score = None, None
            for repeat in range(args.repeat):
                score, state = one_pass(kit, fixture, parent / f"{name}-run-{repeat}")
                if baseline is None:
                    baseline, first_score = state, score
                else:
                    assert state == baseline, f"{name}: state export changed on repeat {repeat}"
                    assert score == first_score
            digest = hashlib.sha256(baseline).hexdigest()
            result = {"instance": name, **first_score, "identical_runs": args.repeat, "state_sha256": digest}
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        (parent / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        passed = True
    finally:
        if args.keep or not passed:
            print("Artifacts: " + str(parent), flush=True)
        else:
            shutil.rmtree(parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
