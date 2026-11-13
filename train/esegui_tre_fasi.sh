#!/usr/bin/env bash
# Le tre fasi in sequenza. Log in /workspace/log/{cpt,sft,orpo}.log
set -euo pipefail
cd /workspace && source venv/bin/activate
python train/addestra.py cpt  modelli/wsp     modelli/ita-cpt  dati/cpt.jsonl  --max_token 30e6 > log/cpt.log 2>&1
python train/addestra.py sft  modelli/ita-cpt modelli/ita-sft  dati/sft.jsonl                  > log/sft.log 2>&1
python train/addestra.py orpo modelli/ita-sft modelli/ita-orpo dati/orpo.jsonl                 > log/orpo.log 2>&1
echo TUTTO_FATTO > log/tre_fasi.fatto
