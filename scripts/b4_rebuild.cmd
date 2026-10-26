@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d C:\Users\kiran\code\p\infer\slmspeed\llama-src
cmake --build build-b4 --target llama-server llama-tokenize llama-quantize llama-bench -j 4
