@echo off
rem B4: configure + build llama.cpp b11884 (CPU side only; CUDA backend DLL reused from the release in llama-cpp\)
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d C:\Users\kiran\code\p\infer\slmspeed\llama-src
cmake -S . -B build-b4 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
if errorlevel 1 exit /b 1
cmake --build build-b4 --target llama-server llama-tokenize llama-quantize llama-bench -j 4
