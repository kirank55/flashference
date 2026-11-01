"""Self-test for check.py --predictions: broken histories must fail, for the intended reason, and
clean ones must pass.

Clones this repo at HEAD into a temporary directory, puts the working tree's check.py in it (so an
edit is tested before it is committed), builds one small history per case on top of it (never
touching the real repo), and runs that check.py on each. Commit dates are set
explicitly (from a few seconds in the future onwards), so no case depends on sleeps or on this
machine's stepping wall clock, and the dates still increase along each branch.

  python autopilot/test_check_predictions.py        (about 30 s; exit 1 if any case misbehaves)
"""
import json, os, re, shutil, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ff-checktest-"))
    C = tmp / "clone"
    try:
        subprocess.run(["git", "clone", "-q", str(SRC), str(C)], check=True)

        def sh(*a, check=True, env=None):
            p = subprocess.run(a, cwd=C, capture_output=True, text=True, env=env)
            if check and p.returncode:
                raise SystemExit(f"{a}: {p.stderr}")
            return p
        sh("git", "config", "user.email", "selftest@localhost"); sh("git", "config", "user.name", "selftest")
        wt = (SRC / "autopilot" / "check.py").read_bytes()       # test the check.py being committed
        if (C / "autopilot" / "check.py").read_bytes() != wt:
            (C / "autopilot" / "check.py").write_bytes(wt)
            sh("git", "commit", "-q", "-am", "self-test: the working tree's check.py")
        base = sh("git", "rev-parse", "HEAD").stdout.strip()
        T = max(int(time.time()), int(sh("git", "log", "-1", "--format=%ct").stdout)) + 10
        nb = C / "NOTEBOOK.md"
        last = max(re.findall(r"^(P-\d{3}) \|", nb.read_text(), re.M))
        new = f"P-{int(last[2:]) + 1:03d}"

        def fresh(name):
            sh("git", "checkout", "-q", "-B", name, base)

        def commit(at, msg, *paths):
            d = f"@{int(at)} +0000"
            env = {**os.environ, "GIT_AUTHOR_DATE": d, "GIT_COMMITTER_DATE": d}
            sh("git", "add", *(paths or ["NOTEBOOK.md"])); sh("git", "commit", "-q", "-m", msg, env=env)

        def add_pred(evid, tail="got __ | error __ | learned __"):
            t = nb.read_text(); i = t.index("\n", t.index(f"{last} |")) + 1
            nb.write_text(t[:i] + f"{new} | 2026-01-01 | 0.0 | self-test prediction | evidence: {evid} | {tail}\n" + t[i:])

        def set_result(tail):
            nb.write_text(re.sub(rf"^({new} \|.*\| )got .*$", rf"\g<1>{tail}", nb.read_text(), flags=re.M))

        def evidence(at, name, text):
            (C / "results" / name).write_text(text); commit(at, "evidence", f"results/{name}")

        def check():
            p = sh("python3", "autopilot/check.py", "--predictions", check=False)
            return p.returncode, p.stdout

        def predicted(evid, at=T, tail="got __ | error __ | learned __"):
            add_pred(evid, tail); commit(at, "predict")

        FILLED = "got 1 | error 0 | learned x"
        cases = []                                       # (name, expected substring or None for pass, result)
        fresh("baseline"); cases.append(("the real history", None, check()))
        fresh("deleted")
        nb.write_text(re.sub(rf"^{last} \|.*\n", "", nb.read_text(), flags=re.M)); commit(T, "delete")
        cases.append(("a prediction line deleted", "deleted: ", check()))
        fresh("relabel")
        nb.write_text(re.sub(r"^(P-001 \|.*\| )got [^|]*\|", r"\1got VOID: relabelled |", nb.read_text(), flags=re.M))
        commit(T, "relabel"); cases.append(("a filled-in result relabelled VOID", "got changed in", check()))
        fresh("void-provable"); predicted("results/st.json")
        evidence(T + 60, "st.json", json.dumps({"started_utc": iso(T + 30)}))
        set_result("got VOID: a miss | error n/a | learned x"); commit(T + 61, "void")
        cases.append(("a provably ordered run marked VOID", "a miss can't be voided", check()))
        fresh("born-void"); predicted("results/st.json", tail="got VOID: x | error n/a | learned x")
        evidence(T + 60, "st.json", json.dumps({"started_utc": iso(T + 30)}))
        cases.append(("a prediction committed already VOID", "first committed with a result already filled in", check()))
        fresh("early-json"); predicted("results/st.json")
        evidence(T + 60, "st.json", json.dumps({"some_key": "2020-01-01T00:00:00Z"})); set_result(FILLED); commit(T + 61, "fill")
        cases.append(("an early ISO time under an unknown JSON key", "before the prediction commit", check()))
        fresh("early-epoch"); predicted("results/st.json")
        evidence(T + 60, "st.json", json.dumps({"started_utc": iso(T + 30), "start_realtime_ns": (T - 3600) * 10**9}))
        set_result(FILLED); commit(T + 61, "fill")
        cases.append(("a late ISO start but an early epoch-ns time", "before the prediction commit", check()))
        fresh("early-text"); predicted("results/st.txt")
        evidence(T + 60, "st.txt", "run started 2020-01-01 00:00:00\n"); set_result(FILLED); commit(T + 61, "fill")
        cases.append(("an early time in text evidence", "before the prediction commit", check()))
        fresh("no-time"); predicted("results/st.txt")
        evidence(T + 60, "st.txt", "no time here\n"); set_result(FILLED); commit(T + 61, "fill")
        cases.append(("evidence with no timestamp", "records no timestamp", check()))
        fresh("in-step-window"); predicted("results/st.json")
        evidence(T + 60, "st.json", json.dumps({"started_utc": iso(T + 3.1)})); set_result(FILLED); commit(T + 61, "fill")
        cases.append(("a run 3.1 s after the prediction second, not VOID", "order unprovable", check()))
        fresh("void-in-window"); predicted("results/st.json")
        evidence(T + 60, "st.json", json.dumps({"started_utc": iso(T + 3.1)}))
        set_result("got VOID: started 3.1 s after the prediction | error n/a | learned x"); commit(T + 61, "void")
        cases.append(("the same run marked VOID after its evidence", None, check()))
        fresh("void-early"); predicted("results/st.json")
        set_result("got VOID: premature | error n/a | learned x"); commit(T + 1, "void")
        evidence(T + 60, "st.json", json.dumps({"started_utc": iso(T + 3.1)}))
        cases.append(("a VOID mark committed before its evidence", "VOID mark first appears before", check()))
        fresh("edited"); predicted("results/st.json")
        evidence(T + 60, "st.json", json.dumps({"started_utc": iso(T + 30)})); set_result(FILLED); commit(T + 61, "fill")
        evidence(T + 62, "st.json", json.dumps({"started_utc": iso(T + 30), "x": 1}))
        cases.append(("evidence edited after its first commit", "changed after its first commit", check()))
        fresh("rewritten"); predicted("results/st.json")
        evidence(T + 60, "st.json", json.dumps({"started_utc": iso(T + 30)})); set_result(FILLED); commit(T + 61, "fill")
        set_result("got 2 | error 0 | learned x"); commit(T + 62, "rewrite")
        cases.append(("a filled-in result rewritten later", "got changed in", check()))
        fresh("rewritten-uncommitted"); predicted("results/st.json")
        evidence(T + 60, "st.json", json.dumps({"started_utc": iso(T + 30)})); set_result(FILLED); commit(T + 61, "fill")
        set_result("got 1 | error 9 | learned x")
        cases.append(("a filled-in error rewritten in the working tree", "error changed in the working tree", check()))
        sh("git", "checkout", "-q", "--", "NOTEBOOK.md")
        fresh("clean"); predicted("results/st.json")
        evidence(T + 60, "st.json", json.dumps({"started_utc": iso(T + 30)})); set_result(FILLED); commit(T + 61, "fill")
        cases.append(("a clean prediction and run", None, check()))
        bad = 0
        for name, want, (code, out) in cases:
            fails = [l for l in out.splitlines() if l.startswith("FAIL")]
            ok = code == 0 and not fails if want is None else code != 0 and any(want in l for l in fails)
            bad += not ok
            why = next((l for l in fails if want and want in l), fails[0] if fails else "no FAIL line")
            print(f"{'ok  ' if ok else 'BAD '} {name}: exit {code}, want {'pass' if want is None else repr(want)}"
                  + ("" if want is None and ok else f" | {why[:150]}"))
        return 1 if bad else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
