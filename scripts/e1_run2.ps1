Set-Location C:\Users\kiran\code\p\infer\slmspeed
$env:PYTHONUTF8='1'
python scripts\e_bench.py --only e1-code-trim8k-n2 e1-code-trim32k-n2 e1-code-trim8k-n1 e1-lang-plain e1-lang-untrim-n2 e1-lang-trim8k-n2 e1-lang-trim32k-n2 e1-lang-trim8k-n1 e1-t07-trim8k-n2-greedy e1-t07-trim8k-n2-probabilistic e1-t07-trim8k-n1-greedy
python scripts\e_bench.py --only e1-free-plain --tag=-end
