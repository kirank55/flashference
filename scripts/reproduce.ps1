<#
.SYNOPSIS
  One-command reproduction of the slmspeed headline results (fast code + fast copy-heavy generation, Qwen3.5-0.8B bf16, llama.cpp).
.DESCRIPTION
  1. builds (or locates) llama.cpp b11884 (commit 63412d873) + patches/*.patch
  2. downloads (or locates) the bf16 GGUF
  3. runs plain vs code config vs copy-heavy config on prompts\ (1 warm-up pass + N timed passes, fresh server per config)
  4. prints tok/s, acceptance and the ratio vs plain on the same prompt set (scripts\reproduce_bench.py)
  The token-rank file that trims the draft head is shipped in results\; the trim itself happens at model load (env vars), no new GGUF is written.
.PARAMETER Smoke           1 timed pass, 64 tokens, code+copy only (a few minutes at most)
.PARAMETER Passes          timed passes per config (default 3)
.PARAMETER LlamaDir        folder with a PATCHED llama-server (skips the build). Also: env SLMSPEED_LLAMA
.PARAMETER Model           bf16 GGUF path (default models\Qwen_Qwen3.5-0.8B-bf16.gguf, downloaded if missing)
.PARAMETER Build           force a (re)build of the patched llama.cpp into work\llama-patched
.PARAMETER BuildOnly       build/locate and stop
.PARAMETER IncludeProse    also run ordinary-prose sets (shows where the n-gram config is slower than MTP alone)
.PARAMETER RequireIdle     abort instead of warning when the GPU is busy
.PARAMETER CudaArch        CMAKE_CUDA_ARCHITECTURES for a native CUDA build (default: native)
.EXAMPLE
  .\scripts\reproduce.ps1 -Smoke
  .\scripts\reproduce.ps1                 # full: 3 timed passes x 256 tokens
#>
[CmdletBinding()]
param(
  [switch]$Smoke, [int]$Passes = 3, [int]$NPredict = 256,
  [string]$LlamaDir, [string]$Model, [switch]$Build, [switch]$BuildOnly,
  [switch]$IncludeProse, [switch]$RequireIdle, [string]$CudaArch = "native"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root
$LlamaTag = "b11884"; $LlamaCommit = "63412d873"
$Work = Join-Path $Root "work"; $Src = Join-Path $Work "llama.cpp"; $Out = Join-Path $Work "llama-patched"; $Bin = Join-Path $Out "bin"
$ModelDefault = Join-Path $Root "models\Qwen_Qwen3.5-0.8B-bf16.gguf"
$ModelUrl = "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/Qwen_Qwen3.5-0.8B-bf16.gguf"

function Native([string]$exe, [string[]]$a) {   # run a native command, stderr is not an error in PS 5.1; check the exit code instead
  $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
  & $exe @a 2>&1 | ForEach-Object { "$_" } | Out-Host
  $rc = $LASTEXITCODE; $ErrorActionPreference = $old
  if ($rc -ne 0) { throw "$exe $($a -join ' ') failed (exit $rc)" }
}
function Find-Vcvars {
  if ($env:VCVARS -and (Test-Path $env:VCVARS)) { return $env:VCVARS }
  $vw = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vw) {
    $p = & $vw -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($p) { $v = Join-Path $p "VC\Auxiliary\Build\vcvars64.bat"; if (Test-Path $v) { return $v } }
  }
  throw "MSVC (Visual Studio Build Tools, C++ workload) not found; install it or set env VCVARS to vcvars64.bat, or pass -LlamaDir with a patched llama-server."
}
function Get-PatchedSource {
  New-Item -ItemType Directory -Force $Work | Out-Null
  if (-not (Test-Path (Join-Path $Src ".git"))) {
    Native git @("-c", "core.autocrlf=false", "clone", "--depth", "1", "--branch", $LlamaTag, "https://github.com/ggml-org/llama.cpp.git", $Src)
  }
  Push-Location $Src
  try {
    $head = (& git rev-parse HEAD).Trim()
    if (-not $head.StartsWith($LlamaCommit)) { throw "llama.cpp checkout is $head, expected $LlamaCommit ($LlamaTag)" }
    Native git @("checkout", "--", ".")            # start from the pristine tag, then apply both patches in order
    Native git @("apply", (Join-Path $Root "patches\b4-trim-draft-head.patch"))
    Native git @("apply", (Join-Path $Root "patches\e2-e4-adaptive-n-dynamic-vocab.patch"))
  } finally { Pop-Location }
}
function Build-Llama {
  Get-PatchedSource
  New-Item -ItemType Directory -Force $Bin | Out-Null
  $vc = Find-Vcvars
  $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
  $bd = Join-Path $Src "build-slmspeed"
  if ($nvcc) {
    Write-Host "== CUDA toolkit found: native CUDA build (arch $CudaArch)"
    $cfg = "cmake -S . -B build-slmspeed -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CudaArch -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl"
    $copyRelease = $false
  } else {
    Write-Host "== No CUDA toolkit: hybrid build (patched CPU-side code + the official b11884 CUDA 12.4 DLLs; the patch touches no ggml-cuda code)"
    $cfg = "cmake -S . -B build-slmspeed -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl"
    $copyRelease = $true
    $base = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag"
    foreach ($z in "llama-$LlamaTag-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip") {
      $zp = Join-Path $Work $z
      if (-not (Test-Path $zp)) { Native curl.exe @("-L", "-o", $zp, "$base/$z") }
      Expand-Archive -Force $zp $Bin
    }
  }
  $bat = Join-Path $Work "build-slmspeed.cmd"
  Set-Content -Encoding ASCII $bat @("@echo off", "call `"$vc`"", "cd /d `"$Src`"", $cfg, "if errorlevel 1 exit /b 1", "cmake --build build-slmspeed --target llama-server llama-tokenize llama-quantize llama-bench -j 4")
  Native cmd @("/c", $bat)
  $built = Join-Path $bd "bin"
  if (-not (Test-Path (Join-Path $built "llama-server.exe"))) { throw "build produced no llama-server.exe in $built" }
  Copy-Item (Join-Path $built "*") $Bin -Force            # overwrites the release exes/dlls with the patched ones (ggml-cuda.dll / cuBLAS stay from the release in hybrid mode)
  Write-Host "== patched llama.cpp ready: $Bin"
}

# ---- 1. locate / build llama.cpp
$srv = $null
if ($LlamaDir) { $srv = $LlamaDir } elseif ($env:SLMSPEED_LLAMA) { $srv = $env:SLMSPEED_LLAMA }
elseif (-not $Build) {
  foreach ($c in @($Bin, (Join-Path $Root "llama-cpp-e"))) { if (Test-Path (Join-Path $c "llama-server.exe")) { $srv = $c; break } }   # llama-cpp-e = the original dev build of the same patched source
}
if ($Build -or -not $srv) { Build-Llama; $srv = $Bin }
if (-not (Test-Path (Join-Path $srv "llama-server.exe"))) { throw "llama-server.exe not found in $srv" }
Write-Host "== llama.cpp: $srv"
if ($BuildOnly) { return }

# ---- 2. model
if (-not $Model) { $Model = $ModelDefault }
if (-not (Test-Path $Model)) {
  Write-Host "== downloading the bf16 GGUF (~1.5 GB) to $Model"
  New-Item -ItemType Directory -Force (Split-Path $Model) | Out-Null
  Native curl.exe @("-L", "-C", "-", "-o", $Model, $ModelUrl)
}
Write-Host "== model: $Model"

# ---- 3. benchmark
$py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }
$env:PYTHONUTF8 = "1"
$a = @("scripts\reproduce_bench.py", "--llama-bin", $srv, "--model", $Model, "--passes", $Passes, "--n-predict", $NPredict)
if ($Smoke) { $a += "--smoke" }
if ($IncludeProse) { $a += "--include-prose" }
if ($RequireIdle) { $a += "--require-idle" }
& $py @a
if ($LASTEXITCODE -ne 0) { throw "benchmark failed (exit $LASTEXITCODE)" }
