#!/usr/bin/env bash
# Prepara il pod ambiente Python, llama.cpp, pesi del modello di base.
# Uso: bash prepara_pod.sh   (idempotente; log in /workspace/log/prepara_pod.log)
set -euo pipefail
W=/workspace
mkdir -p "$W/log" "$W/modelli" "$W/gguf"
exec > >(tee -a "$W/log/prepara_pod.log") 2>&1
echo "== inizio $(date -u +%FT%TZ)"

# 1. Ambiente Python: venv che riusa il torch dell'immagine
if [ ! -d "$W/venv" ]; then python -m venv --system-site-packages "$W/venv"; fi
source "$W/venv/bin/activate"
pip install -q --upgrade pip
pip install -q "transformers @ git+https://github.com/huggingface/transformers.git@main" \
  peft accelerate datasets huggingface_hub sentencepiece protobuf openai jsonschema

# 2. llama.cpp dal ramo corrente, con CUDA se il compilatore c'e'
if [ ! -d "$W/llama.cpp" ]; then git clone --depth 1 --branch v1.0.0 https://github.com/ggml-org/llama.cpp "$W/llama.cpp"; fi
cd "$W/llama.cpp"
CUDA=OFF; command -v nvcc >/dev/null 2>&1 && CUDA=ON
cmake -B build -DGGML_CUDA=$CUDA -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j 48 --target llama-server llama-cli llama-quantize llama-bench 2>&1 | tail -3
# I requisiti del convertitore fissano torch (CPU) e una transformers piu' vecchia:
# vanno in un ambiente separato, altrimenti sovrascrivono quelli di addestramento.
deactivate
if [ ! -d "$W/venv-conv" ]; then python -m venv "$W/venv-conv"; fi
"$W/venv-conv/bin/pip" install -q --upgrade pip
"$W/venv-conv/bin/pip" install -q -r requirements/requirements-convert_hf_to_gguf.txt
source "$W/venv/bin/activate"

# 3. Pesi del modello di base
hf download Qwen/Qwen3.5-0.8B --local-dir "$W/modelli/Qwen3.5-0.8B" >/dev/null

# 4. Versioni fissate
{
  echo "data: $(date -u +%FT%TZ)"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
  echo "python: $(python --version 2>&1)"
  echo "llama.cpp: $(git -C "$W/llama.cpp" rev-parse HEAD) cuda=$CUDA"
  echo "modello: $(cat "$W/modelli/Qwen3.5-0.8B/.cache/huggingface/download/config.json.metadata" 2>/dev/null | head -1)"
  pip freeze | grep -i -E "^(torch|transformers|peft|accelerate|datasets|tokenizers|huggingface.hub|safetensors|gguf|numpy)[ =@]"
} > "$W/log/versioni.txt"
cat "$W/log/versioni.txt"
echo "== fine $(date -u +%FT%TZ) PRONTO"
