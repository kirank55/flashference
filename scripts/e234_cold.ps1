Set-Location C:\Users\kiran\code\p\infer\slmspeed
$env:PYTHONUTF8='1'
foreach ($t in '-c1','-c2','-c3') {
  python -W ignore scripts\e_bench.py --module e_cfg2 --reps 1 --tag=$t --only e3-free-ngmod-mtp-cold e3-rep-ngmod-mtp-cold
}
git add results/2026-10-27-bf16-E.raw.json scripts
git commit -m "raw data: cold ngram-mod" | Out-Null
"QUEUE3 DONE"
