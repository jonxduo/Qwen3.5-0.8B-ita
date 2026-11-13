#!/usr/bin/env bash
# Dopo le tre fasi: valutazione di ogni fase, conversione in GGUF, prova di regressione sul compito A1.
set -uo pipefail
cd /workspace && source venv/bin/activate
while [ ! -f log/tre_fasi.fatto ]; do
  tmux has-session -t fasi 2>/dev/null || { echo "le fasi si sono fermate senza finire" > log/finale.errore; exit 1; }
  sleep 30
done
for f in cpt sft orpo; do
  python eval/valuta_prosa.py modelli/ita-$f risultati/prosa_$f.json > log/valuta_$f.log 2>&1
done
for f in sft orpo; do
  rm -f modelli/ita-$f/model.safetensors.index.json
  bash ambiente/converti_gguf.sh modelli/ita-$f ita-$f > log/converti_$f.log 2>&1
  llama.cpp/build/bin/llama-quantize gguf/ita-$f-f16.gguf gguf/ita-$f-Q5_K_M.gguf Q5_K_M >> log/converti_$f.log 2>&1
done
python eval/prova_schema_json.py esempi/scadenze.sistema.txt esempi/scadenze.schema.json eval/testi_scadenze.jsonl - risultati/a1 \
  gguf/qwen35-08b-base-Q8_0.gguf gguf/ita-sft-Q8_0.gguf gguf/ita-orpo-Q8_0.gguf gguf/ita-orpo-Q4_K_M.gguf > log/a1_regressione.log 2>&1
sha256sum gguf/ita-*.gguf > risultati/sha256.txt
echo FINALE_FATTO > log/finale.fatto
