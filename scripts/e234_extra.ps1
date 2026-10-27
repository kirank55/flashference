cd C:\Users\kiran\code\p\infer\slmspeed
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONUTF8 = "1"; $env:E_MAXWAIT = "600"
python scripts\e_bench.py --module e_cfg2 --reps 3 --only e3-rep-plain e3-free-ngmod-mtp
python scripts\e_bench.py --reps 3 --only e1-free-plain --tag=-e234
git add results\2026-10-27-bf16-E.raw.json; git commit -qm "raw data: extra controls"
Write-Output "QUEUE2 DONE"
