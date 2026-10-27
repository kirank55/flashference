Set-Location (Split-Path $PSScriptRoot -Parent)
python scripts\bf16_bench.py --session B2 --only control backend-sampling bs-topk1 req-nonsampling-topk1
python scripts\bf16_bench.py --session B2 --only np-auto bs-np-auto bs-nograph
python scripts\bf16_bench.py --session B2 --only control --tag=-end
