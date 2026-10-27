# E1 rerun -> E4 -> E2 -> E3 -> E5 ceiling; 3 timed passes for sweeps; commits raw data after each block; stops if the GPU stays busy >10 min
cd C:\Users\kiran\code\p\infer\slmspeed
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONUTF8 = "1"; $env:E_MAXWAIT = "600"
function Block($name, $cmd) {
  Write-Output "=== BLOCK $name $(Get-Date -Format HH:mm:ss)"
  Invoke-Expression $cmd
  $rc = $LASTEXITCODE
  git add results\2026-10-27-bf16-E.raw.json 2>&1 | Out-Null
  git commit -qm "raw data: $name" 2>&1 | Out-Null
  if ($rc -ne 0) { Write-Output "BLOCK $name exit $rc -> stopping queue (GPU busy?)"; exit 1 }
}
Block "E1 code trim8k n2" "python scripts\e_bench.py --only e1-code-trim8k-n2"
Block "E4" "python scripts\e_bench.py --module e_cfg2 --reps 3 --only e4-lang-fix4k-dyn4k e4-code-fix4k-dyn4k e4-free-fix4k-dyn4k e4-lang-fix2k-dyn6k"
Block "E2" "python scripts\e_bench.py --module e_cfg2 --reps 3 --only e2-free-base e2-free-adapt4 e2-free-n3 e2-free-n3-p0.5 e2-free-n4-p0.5 e2-free-adapt3"
Block "E3" "python scripts\e_bench.py --module e_cfg2 --reps 3 --only e3-rep-mtp e3-rep-ngs-mtp-n4m8 e3-rep-ngmod-mtp e3-rep-ngs-only e3-free-mtp e3-free-ngs-mtp-n4m8"
Block "E5 ceiling" "python scripts\e_bench.py --module e_cfg2 --reps 3 --only e5-plain-q8head e5-trim8k-q8head"
Write-Output "QUEUE DONE $(Get-Date -Format HH:mm:ss)"
