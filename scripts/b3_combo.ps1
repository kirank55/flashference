Set-Location C:\Users\kiran\code\p\infer\slmspeed
python scripts\bf16_bench.py --session B3 --only control --tag=-c1
python scripts\bf16_bench.py --session B3 --only mtp-n1-bs
python scripts\bf16_bench.py --session B3 --only mtp-n1-pdef --tag=-c2
python scripts\bf16_bench.py --session B3 --only control --tag=-c3
