"""Prepara i dati delle tre fasi (CPT, SFT, ORPO) da fonti pubbliche con licenza nota.
Uso: python prepara_dati.py <cartella_uscita> <milioni_di_caratteri_cpt>
Scrive: cpt.jsonl, sft.jsonl, orpo.jsonl, piu' i file tenuti da parte per la valutazione (heldout_*.jsonl) e fonti.json.
"""
import json, random, re, sys
from pathlib import Path
from datasets import load_dataset

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
MC = float(sys.argv[2]) * 1e6
random.seed(13)
fonti = {}

def scrivi(nome, righe):
    with open(out / nome, "w", encoding="utf-8") as f:
        for r in righe:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{nome}: {len(righe)} righe", flush=True)

def pulito(t):
    if len(t) < 800:
        return False
    lettere = sum(c.isalpha() or c.isspace() for c in t) / len(t)
    return lettere >= 0.80 and t.count("http") <= 2 and "<div" not in t and "{" not in t[:2000] and not re.search(r"[぀-ヿ一-鿿Ѐ-ӿ]", t)

# ---------- CPT ----------
cpt, held_wiki, held_web = [], [], []
# Wikipedia italiano (CC BY-SA 3.0 / GFDL): 30%
n = 0; tot = 0
for es in load_dataset("wikimedia/wikipedia", "20231101.it", split="train", streaming=True).shuffle(seed=13, buffer_size=20000):
    t = es["text"]
    if len(t) < 1500 or not pulito(t):
        continue
    t = t[:8000].rsplit("\n", 1)[0] if len(t) > 8000 else t
    n += 1
    if n <= 200:
        held_wiki.append({"fonte": "wikipedia", "text": t[:4000]}); continue
    cpt.append({"fonte": "wikipedia", "text": t}); tot += len(t)
    if tot >= 0.30 * MC:
        break
fonti["wikimedia/wikipedia 20231101.it"] = {"licenza": "CC BY-SA 3.0 / GFDL", "caratteri": tot, "fase": "cpt"}
print("wikipedia", tot, flush=True)
# FineWeb-2 italiano (ODC-By): 50%
n = 0; tot = 0
for es in load_dataset("HuggingFaceFW/fineweb-2", "ita_Latn", split="train", streaming=True).shuffle(seed=13, buffer_size=20000):
    t = es["text"]
    if float(es.get("language_score") or 0) < 0.98 or len(t) > 15000 or not pulito(t):
        continue
    n += 1
    if n <= 200:
        held_web.append({"fonte": "fineweb-2", "text": t[:4000]}); continue
    cpt.append({"fonte": "fineweb-2", "text": t}); tot += len(t)
    if tot >= 0.50 * MC:
        break
fonti["HuggingFaceFW/fineweb-2 ita_Latn"] = {"licenza": "ODC-By 1.0", "caratteri": tot, "fase": "cpt"}
print("fineweb-2", tot, flush=True)
# Letteratura italiana (testi di Project Gutenberg, pubblico dominio; il dataset non dichiara licenza): 20%
righe = load_dataset("alberto98fx/Letteratura-Italiana", split="train")["text"]
blocchi, corr, dentro = [], [], True
for r in righe:
    b = r.lower()
    if "*** end of" in b or "end of the project gutenberg" in b or "end of project gutenberg" in b:
        dentro = False
    if "*** start of" in b or "*end*the small print" in b:
        dentro = True; corr = []; continue
    if not dentro or any(k in b for k in ("gutenberg", "ebook", "e-book", "http", "www.", "distributed proofread", "transcriber", "produced by", "liber liber")):
        continue
    corr.append(r)
    if sum(len(x) for x in corr) >= 6000:
        t = "\n".join(corr).strip(); corr = []
        if len(t) > 3000 and sum(c.isalpha() or c.isspace() for c in t) / len(t) > 0.85:
            blocchi.append(t)
random.shuffle(blocchi); tot = 0
for t in blocchi:
    cpt.append({"fonte": "letteratura", "text": t}); tot += len(t)
    if tot >= 0.20 * MC:
        break
fonti["alberto98fx/Letteratura-Italiana"] = {"licenza": "non dichiarata; testi di Project Gutenberg in pubblico dominio", "caratteri": tot, "fase": "cpt", "blocchi_disponibili": len(blocchi)}
print("letteratura", tot, "blocchi disponibili", len(blocchi), flush=True)
random.shuffle(cpt)
scrivi("cpt.jsonl", cpt); scrivi("heldout_wikipedia.jsonl", held_wiki); scrivi("heldout_fineweb.jsonl", held_web)

# ---------- SFT ----------
SIS = ["Sei un assistente che risponde in italiano in modo chiaro e naturale.", "Rispondi in italiano corretto, con frasi dirette e senza giri di parole.",
       "Sei un assistente professionale. Scrivi in italiano chiaro e preciso."]
def conv(msgs, fonte):
    msgs = [m for m in msgs if m["content"] and m["content"].strip()]
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant" or sum(len(m["content"]) for m in msgs) > 6000:
        return None
    if msgs[0]["role"] != "system" and random.random() < 0.3:
        msgs = [{"role": "system", "content": random.choice(SIS)}] + msgs
    return {"fonte": fonte, "messages": msgs}
def taglia(msgs):
    # tiene la conversazione fino a un turno dell'assistente scelto a caso (l'ultimo e' quello che si impara)
    idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    return msgs[: random.choice(idx) + 1] if idx else msgs
sft = []
R = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}
k = 0
for es in load_dataset("HuggingFace-Italian-Community/ITA-Base", split="train"):
    if es.get("tools"):
        continue
    c = conv([{"role": R.get(m["from"], m["from"]), "content": m["value"]} for m in es["conversations"]], "ita-base")
    if c: sft.append(c); k += 1
fonti["HuggingFace-Italian-Community/ITA-Base"] = {"licenza": "Apache 2.0", "esempi": k, "fase": "sft"}
def prendi(nome, cfg, quanti, fn, fonte, lic):
    ds = load_dataset(nome, cfg, split="train").shuffle(seed=13); k = 0
    for es in ds:
        c = fn(es)
        if c:
            sft.append(c); k += 1
        if k >= quanti:
            break
    fonti[nome] = {"licenza": lic, "esempi": k, "fase": "sft"}; print(nome, k, flush=True)
prendi("DeepMount00/italian_conversations", None, 9000, lambda es: conv(taglia([{"role": m["role"], "content": m["content"]} for m in es["conversation"]]), "italian_conversations"), "italian_conversations", "CC (variante non specificata nella scheda)")
prendi("efederici/capybara-claude-15k-ita", None, 6000, lambda es: conv(taglia([{"role": R.get(m["role"], m["role"]), "content": m["content"]} for m in es["conversations"]]), "capybara-ita"), "capybara-ita", "MIT")
prendi("DanielSc4/alpaca-cleaned-italian", "it", 6000, lambda es: conv([{"role": "user", "content": es["instruction"] + ("\n\n" + es["input"] if es["input"] else "")}, {"role": "assistant", "content": es["output"]}], "alpaca-it"), "alpaca-it", "CC BY 4.0")
# Domande ancorate a un contesto (SQuAD-it), con casi senza risposta ottenuti scambiando il contesto
sq = load_dataset("crux82/squad_it", split="train").shuffle(seed=13).select(range(9000))
ISTR = "Rispondi in modo breve usando solo il contesto. Se la risposta non è nel contesto, scrivi: Non è presente nel testo."
k = 0
for i, es in enumerate(sq):
    if not es["answers"]["text"]:
        continue
    neg = i % 7 == 0
    ctx = sq[(i + 4321) % len(sq)]["context"] if neg else es["context"]
    if neg and es["answers"]["text"][0] in ctx:
        continue
    ris = "Non è presente nel testo." if neg else es["answers"]["text"][0]
    ris = ris if neg or ris.endswith(".") else ris[0].upper() + ris[1:] + "."
    sft.append({"fonte": "squad-it", "messages": [{"role": "system", "content": ISTR}, {"role": "user", "content": f"Contesto:\n{ctx}\n\nDomanda: {es['question']}"}, {"role": "assistant", "content": ris}]}); k += 1
    if k >= 7000:
        break
fonti["crux82/squad_it"] = {"licenza": "non dichiarata nel dataset; SQuAD originale CC BY-SA 4.0", "esempi": k, "fase": "sft"}
random.shuffle(sft)
scrivi("heldout_sft.jsonl", sft[:300]); scrivi("sft.jsonl", sft[300:])

# ---------- ORPO ----------
orpo = []
for es in load_dataset("efederici/evol-dpo-ita", split="train").shuffle(seed=13):
    q, c, r = es["question"], es["chosen"], es["rejected"]
    if q and c and r and c.strip() != r.strip() and len(c) < 2500 and len(r) < 2500 and len(q) < 1500:
        orpo.append({"prompt": q, "chosen": c, "rejected": r})
    if len(orpo) >= 8200:
        break
fonti["efederici/evol-dpo-ita"] = {"licenza": "Apache 2.0", "esempi": len(orpo) - 200, "fase": "orpo"}
scrivi("heldout_orpo.jsonl", orpo[:200]); scrivi("orpo.jsonl", orpo[200:])
json.dump(fonti, open(out / "fonti.json", "w"), ensure_ascii=False, indent=1)
print("FATTO", json.dumps(fonti, ensure_ascii=False))
