Set-Location (Split-Path $PSScriptRoot -Parent)
python scripts\bf16_bench.py --session B3 --only control mtp-n1-pdef mtp-n2-pdef mtp-n3-pdef mtp-n1-p0.5 mtp-n2-p0.5 mtp-n3-p0.5 mtp-n1-p0.75 mtp-n2-p0.75 mtp-n3-p0.75
python scripts\bf16_bench.py --session B3 --only control --tag=-mid
python scripts\bf16_bench.py --session B3 --only q4mtp-control q4mtp-n1-pdef q4mtp-n2-pdef q4mtp-n3-pdef q4draft-n2
python scripts\bf16_bench.py --session B3 --only control --tag=-end
