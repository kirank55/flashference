# wait for the E1 queue (python pid 3972), then run E4 -> E2 -> E3 with 3 passes per sweep config (1 warm-up + 3 timed)
$ErrorActionPreference = "Continue"
cd C:\Users\kiran\code\p\infer\slmspeed
while (Get-Process -Id 26784 -ErrorAction SilentlyContinue) { Start-Sleep 10 }
$env:PYTHONUNBUFFERED = "1"
python scripts\e_bench.py --only e1-code-trim8k-n2 2>&1
python scripts\e_bench.py --module e_cfg2 --reps 3 --only e4-lang-fix4k-dyn4k e4-code-fix4k-dyn4k e4-lang-fix2k-dyn6k e4-free-fix4k-dyn4k e2-free-base e2-free-adapt4 e2-free-n3 e2-free-n3-p0.5 e2-free-n4-p0.5 e2-free-adapt3 e3-rep-mtp e3-rep-ngs-mtp-n4m8 e3-rep-ngmod-mtp e3-rep-ngs-only e3-free-ngs-mtp-n4m8 e3-free-mtp e4-code-fix2k-dyn6k e2-free-n3-p0.3 e2-free-n4 e3-free-ngmod-mtp e3-free-ngs-only 2>&1


