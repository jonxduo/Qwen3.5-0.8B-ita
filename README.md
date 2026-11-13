# Qwen3.5-0.8B-ita

Esperimento di adattamento all'italiano di Qwen 3.5 0.8B, in tre fasi (prosa, istruzioni, preferenze), con il codice e tutte le misure prima e dopo.

**Stato: sperimentale (v0.1, 13 novembre 2026).** Il modello ha imparato a modellare meglio l'italiano scritto, ma leggendo le risposte la qualità della prosa non è chiaramente migliore di quella del modello di partenza, e in alcuni casi è peggiore. Non è stato valutato su documenti reali. I dettagli sono sotto, senza abbellimenti: lo scopo del repository è rendere ripetibile l'esperimento, non proporre un modello pronto all'uso.

## Che cos'è

- Modello di partenza: [`RikiyaT/Qwen3.5-0.8B-weight-space-programming`](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming), che è [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) con due soli tensori modificati (`layers.15.mlp.down_proj` e `layers.19.mlp.down_proj`; verificato confrontando i pesi). Quei due tensori sono rimasti congelati in tutte le fasi, e lo script lo controlla a fine fase.
- Architettura invariata: modello linguistico da 752 milioni di parametri (strati Gated DeltaNet e di attenzione) più encoder visivo, non toccato. Tokenizer invariato (248.320 token).
- Addestrato in circa 2 ore e 20 minuti su una GPU NVIDIA A40.

## File del rilascio

I pesi sono negli allegati del [rilascio v0.1](../../releases), non nel repository.

| File | Contenuto | Dimensione |
|---|---|---|
| `Qwen3.5-0.8B-ita-Q8_0.gguf` | Modello finale (dopo la fase 3), 8 bit. Scelta consigliata | 812 MB |
| `Qwen3.5-0.8B-ita-Q5_K_M.gguf` | Modello finale, 5 bit | 581 MB |
| `Qwen3.5-0.8B-ita-Q4_K_M.gguf` | Modello finale, 4 bit | 535 MB |
| `Qwen3.5-0.8B-ita-fase2-Q8_0.gguf` | Modello dopo la fase 2 (senza la fase di preferenze), 8 bit | 812 MB |
| `Qwen3.5-0.8B-ita-safetensors.tar` | Modello finale in formato Hugging Face (bf16, con encoder visivo), per proseguire l'addestramento | 1,7 GB |

I GGUF contengono solo il modello linguistico. Impronte SHA-256 in `risultati/sha256.txt`.

## Uso

Serve **llama.cpp v1.0.0 o successivo**: le versioni precedenti non caricano i GGUF di questa architettura (manca un tensore). Convertitore e runtime devono avere la stessa versione.

```bash
llama-server -m Qwen3.5-0.8B-ita-Q8_0.gguf -c 4096 --jinja
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Riassumi in due frasi: ..."}],
  "temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.12,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Il modello è stato addestrato e valutato con il ragionamento disattivo (`enable_thinking: false`). Per ottenere JSON usare sempre l'uscita vincolata da schema (`response_format` con `json_schema`): senza vincolo il modello addestrato rispetta lo schema meno spesso di quello di partenza (vedi sotto).

## Risultati

Tutte le misure sono fatte con `eval/valuta_prosa.py` sugli stessi dati, tenuti da parte dall'addestramento.

| Misura | Qwen ufficiale | Partenza | Dopo fase 1 | Dopo fase 2 | Finale (fase 3) |
|---|---|---|---|---|---|
| Perplessità, 12 testi d'ufficio scritti a mano | 16,0 | 16,2 | 11,3 | 11,1 | 11,1 |
| Perplessità, Wikipedia italiana | 15,1 | 15,4 | 11,7 | 12,0 | 12,2 |
| Perplessità, web italiano | 25,3 | 25,7 | 19,4 | 20,1 | 20,3 |
| Loss sulle risposte a istruzioni (300 esempi) | 1,83 | 1,84 | 1,75 | 1,56 | 1,62 |
| Risposta preferita più probabile della scartata (200 coppie) | 42% | 42% | 40,5% | 41,5% | 57% |

Più bassa è meglio per perplessità e loss; più alta è meglio per l'ultima riga.

**Che cosa è migliorato.** La perplessità scende di circa il 30%, anche su testi d'ufficio che non somigliano ai dati di addestramento. Nelle 20 richieste di prova il modello addestrato resta più aderente al testo fornito (in un riassunto riporta le cifre corrette dove il modello di partenza le sbagliava), in un caso su due riconosce che l'informazione richiesta non è nel contesto, pur aggiungendo nella versione finale un dettaglio inventato (il modello di partenza inventava la risposta in entrambi), e ha smesso di incorniciare le risposte con intestazioni e grassetti.

**Che cosa non è migliorato o è peggiorato.**

- La prosa generata contiene parole inesistenti ("avvisazione", "effettuiare", "Chiusezza") ed errori di accordo. Anche il modello di partenza commette errori di lingua; non li ho contati, quindi non posso dire quale dei due ne faccia di più.
- Le email restano spesso incoerenti nel contenuto, prima e dopo.
- Dopo la fase 2 compaiono registro colloquiale e, in una risposta su venti, emoji: vengono dai dati di conversazione informale.
- Dopo la fase 3 le risposte sono più prolisse e tendono a commentare sé stesse ("Ecco una versione riscritta..."): è lo stile delle risposte preferite nel dataset usato.
- Estrazione di scadenze in JSON su 10 testi, a 8 bit. Con schema vincolato: uscita valida 10 su 10 e citazioni letterali 15 su 17 (modello ufficiale: 10 su 10 e 11 su 12). Senza schema: rispetta lo schema in 3 casi su 10, contro 8 su 10 del modello ufficiale.

Venti richieste e dieci testi non hanno valore statistico: servono a leggere le risposte, che sono tutte in `risultati/generazioni/` e `risultati/estrazione/`. Gli indici automatici (Gulpease, varietà lessicale) sono nei file ma da soli dicono poco: per esempio la quota di "ripetizioni" sale solo perché alcune risposte sono di una parola.

## Come è stato addestrato

| Fase | Dati | Metodo |
|---|---|---|
| 1. Prosa (continual pre-training) | 30 milioni di token in blocchi da 2048 | Addestramento completo del modello linguistico, embedding congelati, tasso 1e-5 con riscaldamento e decadimento a coseno, 81 minuti |
| 2. Istruzioni (SFT) | 27.700 esempi, 14,6 milioni di token | LoRA rango 32, alfa 64, su tutti i moduli lineari tranne i due protetti, tasso 1e-4, loss solo sull'ultima risposta, 35 minuti |
| 3. Preferenze (ORPO) | 8.000 coppie | LoRA rango 32, tasso 3e-5, lambda 0,1, 25 minuti |

A ogni fase il LoRA viene fuso nei pesi. Parametri esatti in `risultati/addestramento_*.json`, andamento in `risultati/log/`, versioni delle librerie in `ambiente/versioni.txt`.

### Dati e licenze

| Fase | Fonte | Quantità | Licenza |
|---|---|---|---|
| 1 | [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (`ita_Latn`) | 62% dei caratteri | ODC-By 1.0 |
| 1 | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.it`) | 37% | CC BY-SA 3.0, GFDL |
| 1 | [`alberto98fx/Letteratura-Italiana`](https://huggingface.co/datasets/alberto98fx/Letteratura-Italiana) | 1% | Non dichiarata; testi di Project Gutenberg |
| 2 | [`DeepMount00/italian_conversations`](https://huggingface.co/datasets/DeepMount00/italian_conversations) | 9.000 | "CC", variante non specificata |
| 2 | [`crux82/squad_it`](https://huggingface.co/datasets/crux82/squad_it) | 7.000, di cui uno su sette con contesto scambiato e risposta "Non è presente nel testo." | Non dichiarata; SQuAD originale CC BY-SA 4.0 |
| 2 | [`efederici/capybara-claude-15k-ita`](https://huggingface.co/datasets/efederici/capybara-claude-15k-ita) | 6.000 | MIT |
| 2 | [`DanielSc4/alpaca-cleaned-italian`](https://huggingface.co/datasets/DanielSc4/alpaca-cleaned-italian) | 6.000 | CC BY 4.0 |
| 2 | [`HuggingFace-Italian-Community/ITA-Base`](https://huggingface.co/datasets/HuggingFace-Italian-Community/ITA-Base) | 16 | Apache 2.0 |
| 3 | [`efederici/evol-dpo-ita`](https://huggingface.co/datasets/efederici/evol-dpo-ita) | 8.000 | Apache 2.0 |

Tre fonti non dichiarano una licenza precisa: chi intende riusare il modello in un prodotto dovrebbe verificarle o riaddestrare senza. Nessun dato privato è stato usato.

### Differenze rispetto al progetto iniziale

L'esperimento seguiva una guida in tre fasi che indicava alcune fonti poi risultate inutilizzabili:

- cinque dataset indicati non esistono su Hugging Face (tra cui entrambi quelli previsti per la fase 3): sono stati sostituiti con quelli della tabella;
- MLSUM, previsto per la sintesi, è riservato alla ricerca non commerciale ed è stato escluso: **il modello non ha visto esempi di riassunto**;
- la raccolta di letteratura è in gran parte poesia antica, e ne è entrato l'1% invece di una quota importante;
- gli insiemi sintetici su misura (risposte ancorate a un contesto con citazioni, email formali, coppie di stile) non sono stati generati.

L'ultimo punto è con ogni probabilità la ragione principale dei limiti osservati: i dati di istruzione disponibili sono in buona parte tradotti a macchina o colloquiali, cioè proprio lo stile che si voleva correggere.

## Ripetere l'esperimento

Su una macchina con una GPU da almeno 40 GB:

```bash
bash ambiente/prepara_pod.sh                      # ambienti Python, llama.cpp v1.0.0, modello ufficiale
hf download RikiyaT/Qwen3.5-0.8B-weight-space-programming --local-dir /workspace/modelli/wsp
python train/prepara_dati.py /workspace/dati 220  # scarica e filtra le fonti
bash train/esegui_tre_fasi.sh                     # le tre fasi in sequenza
bash train/finalizza.sh                           # valutazione, GGUF, prova di estrazione
```

Gli script assumono la cartella `/workspace` con le sottocartelle `train`, `eval`, `ambiente`, `esempi` copiate da questo repository. Note utili per chi lavora su questa architettura:

- il vocabolario da 248.000 token rende i logit enormi (8 GB per 8.000 token): la loss è calcolata a blocchi e solo sulle posizioni etichettate (`train/comune.py`);
- i requisiti del convertitore GGUF sovrascrivono PyTorch e `transformers`: servono due ambienti Python separati;
- dopo l'addestramento il tokenizer va copiato dal modello di base, non risalvato: la `transformers` corrente scrive una classe che il convertitore non riconosce;
- conviene installare `flash-linear-attention`, altrimenti gli strati DeltaNet usano un'implementazione di riferimento più lenta.

## Limiti

- Modello da meno di un miliardo di parametri: inventa fatti, sbaglia calcoli e ragionamenti a più passi. Va usato solo su testo fornito nel contesto e con verifica umana.
- Nessuna valutazione su documenti reali, né su benchmark standard.
- Le altre lingue non sono state misurate dopo l'addestramento e possono essere peggiorate.
- La parte visiva non è stata né addestrata né provata.

## Licenza

Pesi e codice sono distribuiti con licenza Apache 2.0, come il modello di origine (vedi `LICENSE` e `NOTICE`).
