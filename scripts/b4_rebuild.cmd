@echo off
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%"
cd /d "%~dp0..\llama-src"
cmake --build build-b4 --target llama-server llama-tokenize llama-quantize llama-bench -j 4
