#!/usr/bin/env python3
"""check.py: machine checks for the autopilot run (autoplan.md §5.1).

  python autopilot/check.py --phase N        global checks, then phase N's checks (N = B, 0, 1, ...)
  python autopilot/check.py --all            global checks, then every phase's checks
  python autopilot/check.py --predictions    prediction-order check only (§5.2)
  python autopilot/check.py --tables         generated-table check only (§4.4)
  python autopilot/check.py --update-tables  rewrite every generated table from its own command

Prints one PASS / FAIL / SKIP line per check. Exits 1 if any check fails.
"""
import argparse, gzip, hashlib, json, os, re, shlex, shutil, subprocess, sys, tempfile, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULE5 = ["plan.md", "autoplan.md", ".claude/settings.json", "autopilot/guard.sh"]
GITIGNORE = ("models/\nllama.cpp/\nref/\n.venv/\n__pycache__/\n*.pyc\nresults/_incomplete/\n"
             "results/_jobs/\nautopilot/logs/\nprof/\n*.nsys-rep\n*.sqlite\n")
LLAMA_COMMIT = "63412d873"
PLAN_SHA256 = "AD1549EEDC613064971DCBBBFAB6C9B7990984D1C9AB38F792C6F2EC1207BBC2"   # plan.md 0.4
FAILED = []


def run(cmd, env=None, timeout=None, cwd=ROOT, shell=False, inp=None):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout, shell=shell, input=inp)


def git(*args, binary=False):
    p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)
    return p.stdout if binary else p.stdout.decode(errors="replace")


def load_env():
    """The environment autopilot/env.sh gives, so binaries resolve the same way."""
    out = subprocess.run(["bash", "-c", f"source '{ROOT}/autopilot/env.sh' && env -0"],
                         capture_output=True).stdout.decode()
    return dict(kv.split("=", 1) for kv in out.split("\0") if "=" in kv)


def report(name, ok, why=""):
    tag = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    print(f"{tag}  {name}" + (f": {why}" if why else ""))
    if ok is False:
        FAILED.append(name)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_commit():
    for line in git("log", "--reverse", "--format=%H %s", "HEAD").splitlines():
        h, _, subj = line.partition(" ")
        if subj == "bootstrap":
            return h
    return None


def is_ancestor(a, b):
    return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", a, b]).returncode == 0


# ---------------------------------------------------------------- global checks

GEN_RE = re.compile(r"<!-- gen: (.+?) -->\n(.*?)<!-- /gen -->", re.S)


def fenced_ranges(text):
    """Character ranges inside ``` code fences; gen markers there are examples, not tables."""
    out, start = [], None
    for m in re.finditer(r"^```.*$", text, re.M):
        if start is None:
            start = m.start()
        else:
            out.append((start, m.end())); start = None
    return out


def tracked_markdown():
    return [ROOT / f for f in git("ls-files", "*.md").splitlines() if (ROOT / f).exists()]


def gen_output(cmd, env):
    p = run(["bash", "-c", cmd], env=env, timeout=600)
    return p.returncode, p.stdout, p.stderr


def check_tables(update=False):
    env = load_env()
    n = bad = 0
    for f in tracked_markdown():
        text = f.read_text()
        fences = fenced_ranges(text)
        def repl(m):
            nonlocal n, bad
            if any(a <= m.start() < b for a, b in fences):
                return m.group(0)
            n += 1
            rc, out, err = gen_output(m.group(1), env)
            if rc != 0:
                bad += 1
                report(f"table {f.relative_to(ROOT)} `{m.group(1)}`", False, f"command failed: {err.strip()[-300:]}")
                return m.group(0)
            if update:
                return f"<!-- gen: {m.group(1)} -->\n{out.rstrip()}\n<!-- /gen -->"
            if out.strip() != m.group(2).strip():
                bad += 1
                report(f"table {f.relative_to(ROOT)} `{m.group(1)}`", False, "committed table differs from command output")
            return m.group(0)
        new = GEN_RE.sub(repl, text)
        if update and new != text:
            f.write_text(new)
            print(f"updated tables in {f.relative_to(ROOT)}")
    if not update:
        report("generated tables regenerate", bad == 0 if n else None, f"{n - bad}/{n} match" if n else "no tables yet")


PRED_RE = re.compile(r"^(P-\d{3}) \| (\d{4}-\d{2}-\d{2}) \| ([^|]+?) \| (.+?) \| evidence: ([^|]+?) \| "
                     r"got (.+?) \| error (.+?) \| learned (.*)$")
ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2}(?:\.\d+)?)(Z|[+-]\d{2}:?\d{2})?\b")
# Git commit times and evidence stamps both come from CLOCK_REALTIME, which this machine steps back:
# by up to 0.631 s on 2026-10-31 (results/0/clock-steps.json) and by about 1.6 s per 32 s poll after the
# 2026-11-01 restart (timesyncd offset in results/env/timesync-2026-11-01.txt). A run stamped less than
# 1 s (git's resolution) + STEP_MARGIN_S after its prediction commit might have started before it, so
# its order counts as unprovable (reviews/phase-0-r3.md N2b).
STEP_MARGIN_S = 4
TIME_KEY_RE = re.compile(r"time|epoch|realtime|utc|stamp|(^|_)ts($|_)", re.I)
UNIT_SUFFIX = (("_ns", 1e-9), ("_us", 1e-6), ("_ms", 1e-3), ("_s", 1.0))


def iso_times(text):
    """Every ISO-8601 date-time in a string, as epoch seconds (no zone means UTC, this machine's zone)."""
    out = []
    for d, t, z in ISO_RE.findall(text):
        z = "+00:00" if z in ("", "Z") else (z if ":" in z else z[:3] + ":" + z[3:])
        try:
            out.append(datetime.fromisoformat(f"{d}T{t}{z}").timestamp())
        except ValueError:
            pass
    return out


def strings_in(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from strings_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings_in(v)
    elif isinstance(obj, str):
        yield obj


def epoch_times(obj, key=""):
    """Epoch numbers under time-named JSON keys (reviews/phase-0-r3.md N2a). The unit comes from the
    key's suffix (_ns, _us, _ms, _s); without one, s, ms, us and ns are tried in turn. A number counts
    only if it lands between 2020-01-01 and tomorrow, so durations and boot-relative clocks drop out."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from epoch_times(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from epoch_times(v, key)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and TIME_KEY_RE.search(key):
        lo, hi = 1577836800, time.time() + 86400
        scales = [s for suf, s in UNIT_SUFFIX if key.lower().endswith(suf)] or [1.0, 1e-3, 1e-6, 1e-9]
        for sc in scales:
            if lo <= obj * sc <= hi:
                yield obj * sc
                break


def evidence_times(path):
    """Every time recorded in an evidence file: ISO times in any JSON string value and epoch numbers
    under time-named JSON keys (JSON / JSONL, optionally .gz), or ISO times anywhere in a text file.
    None if the file can't be read."""
    p = ROOT / path
    try:
        raw = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()
    except (OSError, UnicodeDecodeError):
        return None
    docs = []
    try:
        docs = [json.loads(raw)]
    except ValueError:
        for line in raw.splitlines():
            try:
                docs.append(json.loads(line))
            except ValueError:
                docs = []
                break
    if not docs:
        return iso_times(raw)
    return [t for d in docs for x in strings_in(d) for t in iso_times(x)] + [t for d in docs for t in epoch_times(d)]


def notebook_versions():
    """(commit, NOTEBOOK.md text) for every commit that touched NOTEBOOK.md, oldest first."""
    return [(c, git("show", f"{c}:NOTEBOOK.md")) for c in git("log", "--reverse", "--format=%H", "--", "NOTEBOOK.md").split()]


def prediction_problems(pid, pred, evid, got):
    """Returns (status, problems, ties, prediction commit). status is pending | checked."""
    problems, ties = [], []
    log = git("log", "--reverse", "--format=%H %ct", "-S", f"{pid} |", "--", "NOTEBOOK.md").split()
    if not log:
        return ("pending", [], [], None) if got.strip() == "__" else \
            ("checked", ["result filled in but the prediction was never committed without it"], [], None)
    pc, pt = log[0], int(log[1])
    old = next((l for l in git("show", f"{pc}:NOTEBOOK.md").splitlines() if l.startswith(pid + " |")), "")
    m0 = PRED_RE.match(old)
    if not m0 or m0.group(6).strip() != "__":
        return "checked", ["first committed with a result already filled in"], [], pc
    if m0.group(4) != pred or m0.group(5) != evid:
        return "checked", ["prediction text or evidence path changed after it was committed"], [], pc
    pending = 0
    for path in [x.strip() for x in evid.split(",")]:
        elog = git("log", "--reverse", "--diff-filter=A", "--format=%H %ct", "--", path).split()
        if not elog:
            if got.strip() == "__":
                pending += 1
            else:
                problems.append(f"result filled in but evidence {path} is not committed")
            continue
        ec, et = elog[0], int(elog[1])
        if ec == pc or not is_ancestor(pc, ec) or et <= pt:
            problems.append(f"evidence {path} ({ec[:7]}) is not strictly after the prediction ({pc[:7]})"); continue
        touched = git("log", "--format=%H", "--", path).split()
        if len(touched) != 1 or run(["git", "diff", "--quiet", "HEAD", "--", path]).returncode:
            problems.append(f"evidence {path} changed after its first commit"); continue
        times = evidence_times(path)
        if not times:
            problems.append(f"evidence {path} records no timestamp"); continue
        # git stores whole seconds and the wall clock steps back (STEP_MARGIN_S), so a run counts as
        # later only from 1 + STEP_MARGIN_S seconds after the prediction commit's second
        early = [t for t in times if t < pt]
        tie = [t for t in times if pt <= t < pt + 1 + STEP_MARGIN_S]
        if early:
            problems.append(f"{path} records {len(early)} timestamp(s) before the prediction commit")
        elif tie:
            ties.append(f"{path} records {len(tie)} timestamp(s) less than {1 + STEP_MARGIN_S} s after the prediction commit")
    if pending and got.strip() == "__":
        return "pending", problems, ties, pc
    return "checked", problems, ties, pc


def result_rewrites(pid, versions):
    """got and error may each be filled in once, from __, and never changed after that
    (reviews/phase-0-r3.md N2c). Corrections go in `learned` or in the notebook entry."""
    seen = {}
    for c, txt in versions:
        line = next((l for l in txt.splitlines() if l.startswith(pid + " |")), None)
        m = PRED_RE.match(line) if line else None
        if not m:
            continue
        for name, val in (("got", m.group(6).strip()), ("error", m.group(7).strip())):
            if name in seen and val != seen[name][1]:
                where = c if c == "the working tree" else c[:7]
                return f"{name} changed in {where} after it was filled in at {seen[name][0][:7]}"
            if val != "__" and name not in seen:
                seen[name] = (c, val)
    return None


def check_predictions():
    nb = ROOT / "NOTEBOOK.md"
    if not nb.exists():
        report("predictions", None, "no NOTEBOOK.md"); return
    lines = [l for l in nb.read_text().splitlines() if re.match(r"^P-\d{3} \|", l)]
    versions = notebook_versions()
    ever = {m.group(1) for _, txt in versions for m in re.finditer(r"^(P-\d{3}) \|", txt, re.M)}
    now_ids = [l[:5] for l in lines]
    want = [f"P-{i:03d}" for i in range(1, len(now_ids) + 1)]
    gone = sorted(ever - set(now_ids))
    report("prediction ids are consecutive and none was ever deleted", not gone and now_ids == want,
           f"deleted: {gone}" if gone else ("" if now_ids == want else f"ids {now_ids} are not P-001..P-{len(now_ids):03d}"))
    if not lines:
        report("predictions committed before their evidence", None, "no predictions yet"); return
    bad, done, pending, void = [], 0, 0, []
    for line in lines:
        m = PRED_RE.match(line)
        if not m:
            bad.append(f"unparsable: {line[:50]}"); continue
        pid, pred, evid, got = m.group(1), m.group(4), m.group(5), m.group(6)
        status, problems, ties, pc = prediction_problems(pid, pred, evid, got)
        rw = result_rewrites(pid, versions + [("the working tree", nb.read_text())])
        if rw:
            problems = problems + [rw]
        if status == "pending" and not problems:
            pending += 1; continue
        if got.startswith("VOID"):
            # VOID is for a run whose order can't be proved only because it began less than
            # 1 + STEP_MARGIN_S s after the prediction commit. Any other problem still fails, and the mark
            # must come in at or after the evidence commit, so a miss can't be voided ahead of time or
            # after the fact.
            first_void = next((c for c, txt in versions
                               if re.search(rf"^{pid} \|.*\| got VOID", txt, re.M)), None)
            ecs = [git("log", "--reverse", "--diff-filter=A", "--format=%H", "--", x.strip()).split()[:1]
                   for x in evid.split(",")]
            after = first_void and all(e and is_ancestor(e[0], first_void) for e in ecs)
            if problems:
                bad += [f"{pid} (VOID): {x}" for x in problems]
            elif not ties:
                bad.append(f"{pid}: marked VOID but its order is provable; a miss can't be voided")
            elif not after:
                bad.append(f"{pid}: VOID mark first appears before its evidence was committed")
            else:
                void.append(pid)
            continue
        if ties:
            bad += [f"{pid}: {x}; order unprovable (git's 1 s resolution plus clock steps; mark VOID and retest)"
                    for x in ties]
        bad += [f"{pid}: {x}" for x in problems]
        if not problems and not ties:
            done += 1
    for b in bad:
        report("prediction order", False, b)
    report("predictions committed before their evidence", not bad,
           f"{len(lines)} predictions, {done} checked in order, {pending} pending"
           + (f", void (not credited, tie verified): {', '.join(void)}" if void else ""))


def check_test_ledger():
    led = ROOT / "results" / "test-ledger.jsonl"
    if not led.exists():
        report("test ledger lists only frozen configs", None, "no test runs yet"); return
    seen, bad = set(), []
    for i, line in enumerate(led.read_text().splitlines(), 1):
        try:
            e = json.loads(line)
        except ValueError:
            bad.append(f"line {i} is not JSON"); continue
        key = (e.get("config"), e.get("frozen_commit"))
        if key in seen:
            bad.append(f"line {i}: second test run of {key[0]}")
        seen.add(key)
        if not e.get("frozen_commit") or not is_ancestor(e["frozen_commit"], "HEAD") or \
           run(["git", "cat-file", "-e", f"{e['frozen_commit']}:{e.get('config')}"]).returncode:
            bad.append(f"line {i}: config {e.get('config')} was not committed at {e.get('frozen_commit')}")
    report("test ledger lists only frozen configs, one run each", not bad, "; ".join(bad) or f"{len(seen)} runs")


def check_test_prompts():
    shafile, tdir = ROOT / "prompts" / "test.sha256", ROOT / "prompts" / "test"
    files = sorted(p for p in tdir.rglob("*") if p.is_file()) if tdir.exists() else []
    if not shafile.exists():
        report("prompts/test matches prompts/test.sha256", None if not files else False,
               "test set not written yet" if not files else "prompts/test has files but no test.sha256")
        return
    listed = {}
    for line in shafile.read_text().splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            listed[name.lstrip("*")] = h
    actual = {str(p.relative_to(ROOT)): sha256_file(p) for p in files}   # hashes only; never printed
    ok = listed == actual
    report("prompts/test matches prompts/test.sha256", ok, f"{len(actual)} files" if ok else "hash or file list differs")


def check_rule5():
    b = bootstrap_commit()
    if not b:
        report("rule-5 files unchanged since bootstrap", None, "no bootstrap commit yet"); return
    bad = []
    for f in RULE5:
        if not (ROOT / f).exists():
            bad.append(f"{f} missing"); continue
        cur = run(["git", "hash-object", f]).stdout.strip()
        if cur != git("rev-parse", f"{b}:{f}").strip() or cur != git("rev-parse", f"HEAD:{f}").strip():
            bad.append(f"{f} differs from bootstrap {b[:7]}")
    report("rule-5 files unchanged since bootstrap", not bad, "; ".join(bad) or f"vs {b[:7]}")


def global_checks():
    check_tables()
    check_predictions()
    check_test_ledger()
    check_test_prompts()
    check_rule5()


# ---------------------------------------------------------------- phase checks

PHASES = {}


def phase(n):
    def deco(fn):
        PHASES.setdefault(n, []).append(fn)
        return fn
    return deco


GUARD_CASES = [  # (command, expected exit code of guard.sh)
    ("sudo true", 2), ("sudo -n true", 2), ("LLAMA_MTP_ADAPT=1 ./build/bin/llama-server --help", 0),
    ("echo adapt; grep -n sudo autoplan.md", 0), ("apt list", 2), ("ls && sudo apt-get install x", 2),
    ("echo $(sudo true)", 2), ("git push origin op", 0), ("git push", 2), ("git push origin main", 2),
    ("git push -f origin op", 2), ("git push --force origin op", 2), ("git push origin op --tags", 2),
    ("git -C llama.cpp push origin op", 2), ("git tag -l", 0), ("git tag v9", 2), ("git tag -d v1.0", 2),
    ("git reset --hard HEAD", 2), ("git clean -fdx", 2), ("git branch -D foo", 2), ("git checkout main", 2),
    ("git switch v2", 2), ("git show v2:plan.md > plan.md", 0), ("cp x /mnt/c/Users/x", 2),
    ("echo hi > /mnt/c/x.txt", 2), ("ls /mnt/c/Users", 0), ("wsl.exe --shutdown", 2),
    ("rm -rf toy/__pycache__", 0), ("rm -rf /tmp/foo", 2), ("rm -rf ~/opt/tmp", 0), ("rm -rf ~/opt", 2),
    ("rm -rf .", 2), ("cd /tmp && rm -rf foo", 2), ("bash -c 'sudo true'", 2),
    ("cat > f <<'EOF'\nsudo apt install x\ngit push --force\nEOF\necho ok", 0),
    ("git commit -m \"never sudo; rm -rf / is bad\"", 0),
]


@phase("B")
def check_bootstrap():
    gi = (ROOT / ".gitignore").read_text() if (ROOT / ".gitignore").exists() else ""
    report(".gitignore is exactly autoplan.md B2's list", gi == GITIGNORE)
    links = [l for l in git("ls-files", "-s").splitlines() if l.startswith("160000")]
    report("no gitlinks in the index", not links, "; ".join(links))
    report("plan.md is byte-identical to v2:plan.md",
           (ROOT / "plan.md").read_bytes() == git("show", "v2:plan.md", binary=True))
    ref = run(["git", "-C", str(ROOT / "ref/flashference"), "rev-parse", "HEAD"]).stdout.strip()
    report("ref/flashference is at v1.0", ref and ref == git("rev-parse", "v1.0^{commit}").strip(), ref[:9])
    same = all((ROOT / "prompts/v1" / p.name).read_bytes() == p.read_bytes()
               for p in (ROOT / "ref/flashference/prompts").glob("*.json"))
    report("prompts/v1 equals ref/flashference/prompts", same)
    bad = []
    for cmd, exp in GUARD_CASES:
        p = run([str(ROOT / "autopilot/guard.sh")], inp=json.dumps({"tool_input": {"command": cmd}, "cwd": str(ROOT)}))
        if p.returncode != exp:
            bad.append(f"{cmd!r} -> {p.returncode} (want {exp})")
    report(f"guard.sh blocks/allows {len(GUARD_CASES)} cases as expected", not bad, "; ".join(bad))
    s = json.loads((ROOT / ".claude/settings.json").read_text()) if (ROOT / ".claude/settings.json").exists() else {}
    hooks = [h["command"] for e in s.get("hooks", {}).get("PreToolUse", []) if e.get("matcher") == "Bash"
             for h in e.get("hooks", [])]
    report(".claude/settings.json registers guard.sh on Bash", any("autopilot/guard.sh" in h for h in hooks))
    for f in ("STATE.md", "NOTEBOOK.md", "autopilot/review-brief.md"):
        report(f"{f} exists", (ROOT / f).exists())
    p = run([sys.executable, str(ROOT / "autopilot/test_check_predictions.py")], timeout=900)
    report("check.py --predictions self-test: broken histories fail, clean ones pass", p.returncode == 0,
           f"{p.stdout.count('ok  ')} cases ok" if p.returncode == 0 else (p.stdout + p.stderr).strip()[-300:])


def state_field(name):
    st = ROOT / "STATE.md"
    m = re.search(rf"^{name}:\s*(\S+)", st.read_text(), re.M) if st.exists() else None
    return m.group(1) if m else None


PHASE1_RUNS = [  # (script and args, output file); greedy uses --repeats 0 (warm-up only)
    ("toy/phase1_tokens.py", "tokens.json"),
    ("toy/phase1_params.py", "params.json"),
    ("toy/phase1_gguf.py", "gguf.json"),
    ("toy/phase1_greedy.py --device cpu --repeats 0", "greedy-cpu.json"),
    ("toy/phase1_greedy.py --device cuda --repeats 0", "greedy-cuda.json"),
    ("toy/phase1_sample.py", "sample.json"),
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
    res = Path(os.environ.get("FF_P1_RESULTS", "results/1"))
    if not res.is_absolute():
        res = ROOT / res

    # (c) 1.2's three methods give identical ids. A missing results file is not a failure yet.
    for dev in ("cpu", "cuda"):
        f = res / f"greedy-{dev}.json"
        name = f"1.2: no-cache, cached and generate() give identical ids ({dev})"
        if not f.exists():
            report(name, None, f"{f.name} not found")
            continue
        try:
            d = json.loads(f.read_text())["deterministic"]
            ids = d["ids"]
            same = (bool(d.get("all_three_equal"))
                    and ids.get("nocache") == ids.get("cached") == ids.get("generate")
                    and len(ids.get("cached") or []) > 0)
            report(name, same, f"{len(ids.get('cached') or [])} tokens each")
        except (ValueError, KeyError, TypeError) as e:
            report(name, False, f"unreadable: {e}")

    # (a) every script runs from a clean copy of HEAD, with HF_HUB_OFFLINE=1 and under the GPU lock;
    # (b) each fresh deterministic section equals the committed one.
    tmp = Path(tempfile.mkdtemp(prefix="ff-check1-"))
    wt = tmp / "checkout"
    wt.mkdir()
    try:
        p = run(["bash", "-o", "pipefail", "-c",
                 f"git -C {shlex.quote(str(ROOT))} archive HEAD | tar -x -C {shlex.quote(str(wt))}"],
                timeout=600)
        if p.returncode:
            report("clean copy of HEAD for the Phase 1 re-run", False, (p.stderr or p.stdout).strip()[-300:])
            return
        (wt / "models").symlink_to(ROOT / "models")     # git-ignored input, documented in plan.md 0.4
        env = load_env()
        env["HF_HUB_OFFLINE"] = "1"
        env.pop("FF_MODEL", None)                        # use the checkout's own default path
        for cmd, out in PHASE1_RUNS:
            od = tmp / "out"
            argv = ["flock", "/tmp/flashference-gpu.lock", str(ROOT / ".venv/bin/python"),
                    *cmd.split(), "--out", str(od)]
            p = run(argv, env=env, cwd=wt, timeout=3600)
            name = f"{cmd} re-runs from a clean checkout (HF_HUB_OFFLINE=1)"
            if p.returncode or not (od / out).exists():
                report(name, False, (p.stderr or p.stdout).strip()[-300:])
                continue
            report(name, True)
            old_path = res / out
            if not old_path.exists():
                report(f"{out}: deterministic part identical on a re-run", None, f"{old_path.name} not committed")
                continue
            try:
                old = json.loads(old_path.read_text())["deterministic"]
                fresh = json.loads((od / out).read_text())["deterministic"]
                d = first_difference(old, fresh)
                report(f"{out}: deterministic part identical on a re-run", d is None, d or "")
            except (ValueError, KeyError) as e:
                report(f"{out}: deterministic part identical on a re-run", False, f"unreadable: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@phase("0")
def check_phase0():
    env = load_env()
    p = run(["nvcc", "--version"], env=env)
    if p.returncode == 0:
        report("nvcc --version works", True, p.stdout.strip().splitlines()[-1])
    else:
        report("nvcc --version works, or STATE.md says toolkit: hybrid", state_field("toolkit") == "hybrid")
    p = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], env=env)
    report("nvidia-smi works", p.returncode == 0, p.stdout.strip())
    p = run(["flock", "/tmp/flashference-gpu.lock", str(ROOT / ".venv/bin/python"), "-c",
             "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"],
            env=env, timeout=300)
    report("torch.cuda.is_available()", p.returncode == 0 and " True " in p.stdout, p.stdout.strip() or p.stderr[-200:])
    tracked = set(git("ls-files", "results/0").splitlines())
    benches = sorted(f for f in tracked if re.match(r"results/0/llama-bench-\d+\.json$", f))
    good = []
    for f in benches:
        try:
            rows = json.loads((ROOT / f).read_text())
            r = rows[0]
            if len(rows) == 1 and r["n_gen"] == 128 and r["n_prompt"] == 0 and r["n_gpu_layers"] == 99 \
               and r["model_filename"].endswith("Qwen_Qwen3.5-0.8B-bf16.gguf") \
               and LLAMA_COMMIT.startswith(r["build_commit"][:7]) and len(r["samples_ts"]) == 5:
                good.append(f)
        except (ValueError, KeyError, IndexError, TypeError):
            pass
    report("3 llama-bench tg128 JSON files (-r 5, b11884) committed", len(good) >= 3 and len(good) == len(benches),
           f"{len(good)} valid of {len(benches)}")
    shaf = "results/0/model-sha256.txt"
    if shaf in tracked:
        rec = (ROOT / shaf).read_text().split()[0].lower()
        act = sha256_file(ROOT / "models/Qwen_Qwen3.5-0.8B-bf16.gguf")
        report("model sha256 recorded and matches the file", rec == act,
               f"{act[:16]}…; plan.md value {'matches' if act == PLAN_SHA256.lower() else 'DIFFERS'}")
    else:
        report("model sha256 recorded", False, f"{shaf} not committed")
    srv = ROOT / "llama.cpp/build/bin/llama-server"
    p = run([str(srv), "--version"], env=env) if srv.exists() else None
    txt = (p.stdout + p.stderr) if p else ""
    report(f"llama-server --version reports {LLAMA_COMMIT}", LLAMA_COMMIT in txt,
           next((l for l in txt.splitlines() if "version" in l), "no binary"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--phase")
    g.add_argument("--all", action="store_true")
    g.add_argument("--predictions", action="store_true")
    g.add_argument("--tables", action="store_true")
    g.add_argument("--update-tables", action="store_true")
    a = ap.parse_args()
    if a.update_tables:
        check_tables(update=True); return 0
    if a.tables:
        check_tables()
    elif a.predictions:
        check_predictions()
    else:
        global_checks()
        phases = list(PHASES) if a.all else [a.phase]
        for n in phases:
            if n not in PHASES:
                report(f"phase {n} checks", False, "no checks registered for this phase"); continue
            print(f"--- phase {n}")
            for fn in PHASES[n]:
                fn()
    print(f"{'FAIL' if FAILED else 'PASS'}: {len(FAILED)} failing check(s)")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
