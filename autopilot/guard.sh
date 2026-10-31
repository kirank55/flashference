#!/usr/bin/env bash
# guard.sh: Claude Code PreToolUse hook on Bash (autoplan.md bootstrap B5).
# Reads the tool call's JSON on stdin. Exit 2 blocks the command (stderr says why); exit 0 allows.
# It prevents accidents; it is not a sandbox. Rules are matched on whole command words
# (shell-tokenized), so `LLAMA_MTP_ADAPT=1` or the word "adapt" never match `apt`.
FF_GUARD_INPUT="$(cat)" FF_GUARD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" \
  exec python3 -I - <<'PY'
import json, os, re, shlex, sys

ROOT = os.path.realpath(os.environ["FF_GUARD_ROOT"])
HOME = os.path.realpath(os.path.expanduser("~"))
OPT = os.path.join(HOME, "opt")

try:
    data = json.loads(os.environ.get("FF_GUARD_INPUT") or "{}")
except ValueError:
    sys.exit(0)                      # not a tool call we understand: let it through
cmd = (data.get("tool_input") or {}).get("command") or ""
cwd0 = data.get("cwd") or os.getcwd()

def block(why):
    print(f"guard.sh blocked this command ({why}). See autoplan.md §1.", file=sys.stderr)
    sys.exit(2)

SEPS = set(";&|()\n")
SHELL_WORDS = {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "for",
               "!", "{", "}", "case", "esac", "select", "function"}

def strip_heredocs(s):
    """Drop heredoc bodies, so text written to a file is never read as commands."""
    out, pending = [], []
    for line in s.split("\n"):
        if pending:
            if line.strip() == pending[0]:
                pending.pop(0)
            continue
        out.append(line)
        pending += [m.group(2) for m in
                    re.finditer(r"(?<!<)<<(?!<)-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", line)]
    return "\n".join(out)

def tokenize(s):
    s = strip_heredocs(s)
    lex = shlex.shlex(s, posix=True, punctuation_chars=";&|()<>\n")
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    lex.commenters = ""
    try:
        return list(lex)
    except ValueError:                # unbalanced quotes: fall back to a rough split
        return s.replace("\n", " ; ").split()

def kind(t):
    if t and set(t) <= set(";&|()<>\n"):
        return "redir" if ("<" in t or ">" in t) else "sep"
    return "word"

def segments(tokens):
    """Yield (words, redirect_targets) for each simple command."""
    words, targets, i = [], [], 0
    while i < len(tokens):
        t = tokens[i]; k = kind(t)
        if k == "sep":
            if words or targets: yield words, targets
            words, targets = [], []
        elif k == "redir":
            if words and words[-1].isdigit():
                words.pop()              # the fd number of 2>file
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if kind(nxt) == "word":
                i += 1
                if ">" in t and not (t.endswith("&") and (nxt.isdigit() or nxt == "-")):
                    targets.append(nxt)
        elif "`" in t:
            pass                         # backtick substitution: not analysed
        else:
            words.append(t)
        i += 1
    if words or targets: yield words, targets

def resolve(path, cwd):
    if "$" in path or "`" in path:
        return None                   # can't know what this expands to
    if any(c in path for c in "*?["):
        parent = os.path.dirname(path.rstrip("/")) or "."
        if any(c in parent for c in "*?["):
            return None
        r = resolve(parent, cwd)
        # a glob may only empty a directory strictly inside the repo or ~/opt
        return None if r in (ROOT, OPT, HOME, "/") else r
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.join(cwd, p)
    return os.path.realpath(p)

def inside(p, base):
    return p == base or p.startswith(base + os.sep)

WRAPPERS = {"env", "nohup", "setsid", "time", "nice", "stdbuf", "timeout", "command",
            "builtin", "exec", "ionice", "chrt", "taskset"}
READONLY = {"ls", "cat", "head", "tail", "stat", "file", "du", "df", "grep", "rg", "wc", "less",
            "md5sum", "sha256sum", "readlink", "realpath", "test", "[", "diff", "cmp", "find"}

def strip_prefix(words):
    """Drop env assignments and wrapper commands; return the real command words."""
    w = list(words)
    while w:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w[0]) or w[0] in SHELL_WORDS:
            w.pop(0); continue
        if w[0] in WRAPPERS:
            name = w.pop(0)
            if name == "timeout":        # timeout [opts] DURATION cmd
                while w and w[0].startswith("-"): w.pop(0)
                if w: w.pop(0)
            while w and w[0].startswith("-"): w.pop(0)
            continue
        if w[0] == "flock":              # flock [opts] FILE cmd...
            w.pop(0)
            while w and w[0].startswith("-"): w.pop(0)
            if w: w.pop(0)
            continue
        if w[0] == "xargs":
            w.pop(0)
            while w and w[0].startswith("-"): w.pop(0)
            w.append("__STDIN__")        # arguments come from stdin: unknown
            continue
        break
    return w

def check(s, cwd, depth=0):
    if depth > 4:
        return cwd
    for words, targets in segments(tokenize(s)):
        if any(t.lower().startswith("/mnt/c") for t in targets):
            block("write under /mnt/c")
        w = strip_prefix(words)
        if not w:
            continue
        prog = os.path.basename(w[0])
        args = w[1:]
        if prog in ("bash", "sh", "zsh", "dash") and "-c" in args:
            i = args.index("-c")
            if i + 1 < len(args):
                check(args[i + 1], cwd, depth + 1)
            continue
        if prog == "eval":
            check(" ".join(args), cwd, depth + 1); continue
        if prog == "cd":
            if args and args[0] != "-":
                r = resolve(args[0], cwd)
                if r: cwd = r
            continue
        if prog in ("sudo", "su", "doas", "pkexec"):
            block("sudo")
        if prog in ("apt", "apt-get", "aptitude", "apt-add-repository", "add-apt-repository") or \
           (prog == "dpkg" and any(a in ("-i", "--install", "-r", "--remove", "-P", "--purge") for a in args)):
            block("apt/dpkg")
        if prog in ("wsl.exe", "wsl"):
            block("wsl.exe")
        if any(a.lower().startswith("/mnt/c") for a in args) and prog not in READONLY:
            block("write under /mnt/c")
        if prog == "find" and any(a.lower().startswith("/mnt/c") for a in args) and \
           any(a in ("-delete", "-exec", "-execdir", "-ok") for a in args):
            block("write under /mnt/c")
        if prog == "rm":
            flags = "".join(a[1:] for a in args if a.startswith("-") and not a.startswith("--"))
            longs = {a for a in args if a.startswith("--")}
            rec = "r" in flags or "R" in flags or "--recursive" in longs
            force = "f" in flags or "--force" in longs
            if rec and force:
                paths = [a for a in args if not a.startswith("-")]
                for p in paths:
                    r = None if p == "__STDIN__" else resolve(p, cwd)
                    if r is None:
                        block(f"rm -rf with a path the guard can't resolve: {p}")
                    if r in (ROOT, OPT, HOME, "/") or not (inside(r, ROOT) or inside(r, OPT)):
                        block(f"rm -rf outside the repo and ~/opt: {p}")
        if prog == "git":
            check_git(args, cwd)
    return cwd

GIT_OPTS_WITH_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}

def check_git(args, cwd):
    a = list(args)
    gdir = cwd
    while a and a[0].startswith("-"):
        opt = a.pop(0)
        if opt in GIT_OPTS_WITH_ARG and a:
            val = a.pop(0)
            if opt == "-C":
                gdir = resolve(val, gdir) or gdir
    if not a:
        return
    sub, rest = a[0], a[1:]
    if sub == "push":
        if any(x in ("--force", "-f", "--tags", "--force-with-lease", "--mirror", "--all", "--delete", "-d")
               or x.startswith("--force") or (x.startswith("-") and not x.startswith("--") and "f" in x)
               for x in rest):
            block("git push with --force/--tags/--delete")
        if not inside(gdir, ROOT) or inside(gdir, os.path.join(ROOT, "llama.cpp")) or \
           inside(gdir, os.path.join(ROOT, "ref")):
            block("git push from outside the op repo")
        if [x for x in rest if x not in ("-q", "--quiet", "-v", "--verbose", "--porcelain")] != ["origin", "op"]:
            block("only `git push origin op` is allowed")
    elif sub == "tag":
        listing = (not rest) or any(x in ("-l", "--list", "-v", "--verify") or
                                    x.startswith(("--list", "--contains", "--points-at", "--merged",
                                                  "--no-merged", "--sort", "-n", "--format", "--no-contains"))
                                    for x in rest)
        if not listing or any(x in ("-d", "--delete", "-f", "--force", "-a", "-s", "-m") for x in rest):
            block("creating, moving or deleting a tag")
    elif sub == "update-ref" and any("refs/tags" in x for x in rest):
        block("moving a tag with update-ref")
    elif sub == "reset" and "--hard" in rest:
        block("git reset --hard")
    elif sub == "clean":
        block("git clean")
    elif sub == "branch":
        if "-D" in rest or (any(x in ("-d", "--delete") for x in rest) and any(x in ("-f", "--force") for x in rest)):
            block("git branch -D")
    elif sub in ("checkout", "switch"):
        if any(x in ("main", "v2", "origin/main", "origin/v2", "refs/heads/main", "refs/heads/v2")
               for x in rest):
            block("checking out main or v2")

check(cmd, cwd0)
sys.exit(0)
PY
