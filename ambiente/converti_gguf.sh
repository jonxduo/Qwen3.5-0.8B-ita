#!/usr/bin/env bash
# Converte un modello Hugging Face in GGUF (solo parte linguistica) e lo quantizza a 8 e 4 bit.
# Uso: bash converti_gguf.sh <cartella_modello> <nome_uscita>
set -euo pipefail
W=/workspace; SRC=$1; NOME=$2; L=$W/llama.cpp
"$W/venv-conv/bin/python" "$L/convert_hf_to_gguf.py" "$SRC" --outfile "$W/gguf/$NOME-f16.gguf" --outtype f16 2>&1 | tail -25
"$L/build/bin/llama-quantize" "$W/gguf/$NOME-f16.gguf" "$W/gguf/$NOME-Q8_0.gguf" Q8_0 2>&1 | tail -3
"$L/build/bin/llama-quantize" "$W/gguf/$NOME-f16.gguf" "$W/gguf/$NOME-Q4_K_M.gguf" Q4_K_M 2>&1 | tail -3
ls -la "$W/gguf/" && sha256sum "$W/gguf/$NOME"-*.gguf
