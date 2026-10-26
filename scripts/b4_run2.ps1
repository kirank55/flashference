Set-Location C:\Users\kiran\code\p\infer\slmspeed
python scripts\bf16_bench.py --session B4 --only control --tag=-mid
python scripts\bf16_bench.py --session B4 --only trim8k-n2-bs trim16k-n2-bs trim32k-n2-bs trim4k-n1-bs trim2k-n1-bs trim4k-n2-bs trim2k-n2-bs
python scripts\bf16_bench.py --session B4 --only control --tag=-mid2
python scripts\bf16_bench.py --session B4 --only mtp-n2-bs own-n2-bs trim16k-n1-nobs own-control
python scripts\bf16_bench.py --session B4 --only control --tag=-end
