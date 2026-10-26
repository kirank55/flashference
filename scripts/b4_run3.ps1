Set-Location C:\Users\kiran\code\p\infer\slmspeed
python scripts\bf16_bench.py --session B4 --only control --tag=-c3a
python scripts\bf16_bench.py --session B4 --only trim8k-n3-bs trim8k-n4-bs
python scripts\bf16_bench.py --session B4 --only control --tag=-c3b
