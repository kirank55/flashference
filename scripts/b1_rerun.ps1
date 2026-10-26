Set-Location C:\Users\kiran\code\p\infer\slmspeed
python scripts\bf16_bench.py --session B1 --only control --tag=-r2a
python scripts\bf16_bench.py --session B1 --only f16-weights gen512 threads8 bench-default --tag=-r2
python scripts\bf16_bench.py --session B1 --only control --tag=-r2b
