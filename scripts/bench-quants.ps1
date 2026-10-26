# Baseline: decode and prompt speed across quant levels, full GPU offload, flash attention on
Set-Location (Split-Path $PSScriptRoot -Parent)
$models = 'Q4_K_M','Q5_K_M','Q6_K','Q8_0','bf16' | ForEach-Object { "-m", "models\Qwen_Qwen3.5-0.8B-$_.gguf" }
& .\llama-cpp\llama-bench.exe @models -ngl 99 -fa 1 -p 512 -n 128 -r 3 -o md 2>$null
