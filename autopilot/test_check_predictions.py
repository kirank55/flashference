"""Self-test for check.py --predictions: broken histories must fail, clean ones must pass.

Clones this repo at HEAD into a temporary directory, builds one small history per case on top
of it (never touching the real repo), and runs that clone's own check.py on each.

  python autopilot/test_check_predictions.py        (about 15 s; exit 1 if any case misbehaves)
"""
import json, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ff-checktest-"))
    C = tmp / "clone"
    try:
        subprocess.run(["git", "clone", "-q", str(SRC), str(C)], check=True)

        def sh(*a, check=True):
            p = subprocess.run(a, cwd=C, capture_output=True, text=True)
            if check and p.returncode:
                raise SystemExit(f"{a}: {p.stderr}")
            return p
        sh("git", "config", "user.email", "selftest@localhost"); sh("git", "config", "user.name", "selftest")
        base = sh("git", "rev-parse", "HEAD").stdout.strip()
        nb = C / "NOTEBOOK.md"
        last = max(re.findall(r"^(P-\d{3}) \|", nb.read_text(), re.M))
        new = f"P-{int(last[2:]) + 1:03d}"

        def fresh(name):
            sh("git", "checkout", "-q", "-B", name, base)

        def commit(msg, *paths):
            sh("git", "add", *(paths or ["NOTEBOOK.md"])); sh("git", "commit", "-q", "-m", msg)

        def add_pred(evid, tail="got __ | error __ | learned __"):
            t = nb.read_text(); i = t.index("\n", t.index(f"{last} |")) + 1
            nb.write_text(t[:i] + f"{new} | 2026-01-01 | 0.0 | self-test prediction | evidence: {evid} | {tail}\n" + t[i:])

        def fill():
            nb.write_text(re.sub(rf"^({new} \|.*\| )got __ \| error __ \| learned __$", r"\1got 1 | error 0 | learned x",
                                 nb.read_text(), flags=re.M))

        def evidence(name, text):
            (C / "results" / name).write_text(text); commit("evidence", f"results/{name}")

        def check():
            return sh("python3", "autopilot/check.py", "--predictions", check=False).returncode

        now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cases = []
        fresh("baseline"); cases.append(("the real history", 0, check()))
        fresh("deleted")
        nb.write_text(re.sub(rf"^{last} \|.*\n", "", nb.read_text(), flags=re.M)); commit("delete")
        cases.append(("a prediction line deleted", 1, check()))
        fresh("relabel")
        nb.write_text(re.sub(r"^(P-001 \|.*\| )got [^|]*\|", r"\1got VOID: relabelled |", nb.read_text(), flags=re.M))
        commit("relabel"); cases.append(("a result relabelled VOID", 1, check()))
        fresh("born-void"); add_pred("results/st.json", "got VOID: x | error n/a | learned x"); commit("predict")
        time.sleep(1.1); evidence("st.json", json.dumps({"started_utc": now()}))
        cases.append(("a prediction committed already VOID", 1, check()))
        fresh("early-json"); add_pred("results/st.json"); commit("predict"); time.sleep(1.1)
        evidence("st.json", json.dumps({"some_key": "2020-01-01T00:00:00Z"})); fill(); commit("fill")
        cases.append(("an early time under an unknown JSON key", 1, check()))
        fresh("early-text"); add_pred("results/st.txt"); commit("predict"); time.sleep(1.1)
        evidence("st.txt", "run started 2020-01-01 00:00:00\n"); fill(); commit("fill")
        cases.append(("an early time in text evidence", 1, check()))
        fresh("no-time"); add_pred("results/st.txt"); commit("predict"); time.sleep(1.1)
        evidence("st.txt", "no time here\n"); fill(); commit("fill")
        cases.append(("evidence with no timestamp", 1, check()))
        fresh("edited"); add_pred("results/st.json"); commit("predict"); time.sleep(2.2)
        evidence("st.json", json.dumps({"started_utc": now()})); fill(); commit("fill")
        evidence("st.json", json.dumps({"started_utc": now(), "x": 1}))
        cases.append(("evidence edited after its first commit", 1, check()))
        fresh("clean"); add_pred("results/st.json"); commit("predict"); time.sleep(2.2)
        evidence("st.json", json.dumps({"started_utc": now()})); fill(); commit("fill")
        cases.append(("a clean prediction and run", 0, check()))
        bad = 0
        for name, want, got in cases:
            ok = (got != 0) == (want != 0)
            bad += not ok
            print(f"{'ok  ' if ok else 'BAD '} {name}: check.py exit {got} (want {'fail' if want else 'pass'})")
        return 1 if bad else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
