Set-Location (Split-Path $PSScriptRoot -Parent)
python scripts\bf16_bench.py --session B2 --only control --tag=-r2a
python scripts\bf16_bench.py --session B2 --only backend-sampling --tag=-r2
python scripts\bf16_bench.py --session B2 --only control --tag=-r2b
