PHASE1_RUNS = [  # (script and args, output file, needs the GPU)
    ("toy/phase1_tokens.py", "tokens.json", False),
    ("toy/phase1_params.py", "params.json", False),
    ("toy/phase1_gguf.py", "gguf.json", False),
    ("toy/phase1_greedy.py --device cpu --repeats 1", "greedy-cpu.json", False),
    ("toy/phase1_greedy.py --device cuda --repeats 1", "greedy-cuda.json", True),
    ("toy/phase1_sample.py", "sample.json", False),
]


def first_difference(a, b, path="deterministic"):
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: missing on one side"
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
    elif a != b:
        return f"{path}: {str(a)[:40]} vs {str(b)[:40]}"
    return None


@phase("1")
def check_phase1():
    res = ROOT / "results/1"
    tracked = set(git("ls-files", "results/1").splitlines())
    for _, out, _ in PHASE1_RUNS:
        report(f"results/1/{out} committed", f"results/1/{out}" in tracked)
    for dev in ("cpu", "cuda"):
        f = res / f"greedy-{dev}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())["deterministic"]
        ids = d["ids"]
        same = d["all_three_equal"] and ids["nocache"] == ids["cached"] == ids["generate"] and len(ids["cached"]) > 0
        report(f"1.2: no-cache, cached and generate() give identical ids ({dev})", same,
               f"{len(ids['cached'])} tokens each")
    tmp = Path(tempfile.mkdtemp(prefix="ff-check1-"))
    wt = tmp / "checkout"
    p = run(["git", "worktree", "add", "--detach", str(wt), "HEAD"])
    if p.returncode:
        report("clean checkout for the re-run", False, p.stderr.strip()); return
    try:
        (wt / "models").symlink_to(ROOT / "models")     # git-ignored input, documented in plan.md 0.4
        env = load_env()
        env["HF_HUB_OFFLINE"] = "1"
        env.pop("FF_MODEL", None)                        # use the checkout's own default path
        for i, (cmd, out, gpu) in enumerate(PHASE1_RUNS):
            od = tmp / f"out{i}"
            argv = [str(ROOT / ".venv/bin/python"), *cmd.split(), "--out", str(od)]
            if gpu:
                argv = ["flock", "/tmp/flashference-gpu.lock", *argv]
            p = run(argv, env=env, cwd=wt, timeout=3600)
            name = f"{cmd.split()[0]} re-run from a clean checkout (HF_HUB_OFFLINE=1)"
            if p.returncode or not (od / out).exists():
                report(name, False, (p.stderr or p.stdout).strip()[-300:]); continue
            new = json.loads((od / out).read_text())["deterministic"]
            old = json.loads((res / out).read_text())["deterministic"] if (res / out).exists() else None
            d = first_difference(old, new) if old is not None else "no committed output"
            report(name + f": {out} deterministic part identical", d is None, d or "")
    finally:
        run(["git", "worktree", "remove", "--force", str(wt)])
        shutil.rmtree(tmp, ignore_errors=True)
