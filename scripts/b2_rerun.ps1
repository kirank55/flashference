Set-Location C:\Users\kiran\code\p\infer\slmspeed
python scripts\bf16_bench.py --session B2 --only control --tag=-r2a
python scripts\bf16_bench.py --session B2 --only backend-sampling --tag=-r2
python scripts\bf16_bench.py --session B2 --only control --tag=-r2b
