# Downloads the llama.cpp Windows CUDA 12.4 build into ./llama-cpp
param([string]$Build = "b11884")
Set-Location (Split-Path $PSScriptRoot -Parent)
$base = "https://github.com/ggml-org/llama.cpp/releases/download/$Build"
curl.exe -L -o llama.zip "$base/llama-$Build-bin-win-cuda-12.4-x64.zip"
curl.exe -L -o cudart.zip "$base/cudart-llama-bin-win-cuda-12.4-x64.zip"
Expand-Archive -Force llama.zip llama-cpp
Expand-Archive -Force cudart.zip llama-cpp
Remove-Item llama.zip, cudart.zip
