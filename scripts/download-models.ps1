# Downloads Qwen3.5-0.8B GGUF quants into ./models
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force models | Out-Null
foreach ($q in 'Q4_K_M','Q5_K_M','Q6_K','Q8_0','bf16') {
  curl.exe -L -o "models\Qwen_Qwen3.5-0.8B-$q.gguf" "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/Qwen_Qwen3.5-0.8B-$q.gguf"
}
