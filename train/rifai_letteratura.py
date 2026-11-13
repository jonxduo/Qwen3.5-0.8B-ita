"""Ricostruisce la quota di letteratura del CPT (il primo filtro scartava quasi tutto) e riscrive cpt.jsonl.
Uso: python rifai_letteratura.py <cartella_dati> <milioni_di_caratteri_letteratura>"""
import json, random, re, sys
from datasets import load_dataset
d, MC = sys.argv[1], float(sys.argv[2]) * 1e6
random.seed(13)
righe = load_dataset("alberto98fx/Letteratura-Italiana", split="train")["text"]
INIZIO = re.compile(r"\*\*\*\s*start of|start of (the|this) project gutenberg|\*end\*the small print", re.I)
FINE = re.compile(r"\*\*\*\s*end of|end of (the |this )?project gutenberg", re.I)
TESTA = re.compile(r"^\s*(the )?project gutenberg('s)? (ebook|etext)", re.I)
SCARTA = ("gutenberg", "ebook", "e-book", "etext", "http", "www.", "distributed proofread", "transcriber", "produced by", "liber liber", "copyright", "license")
blocchi, corr, dentro, salto, libri = [], [], True, 0, 0
for r in righe:
    if TESTA.search(r):
        libri += 1; corr = []; dentro = True; salto = 40; continue   # nuova opera: si saltano le righe di intestazione
    if FINE.search(r):
        dentro = False; corr = []; continue
    if INIZIO.search(r):
        dentro = True; corr = []; salto = 0; continue
    if salto > 0:
        salto -= 1; continue
    b = r.lower()
    if not dentro or any(k in b for k in SCARTA):
        continue
    corr.append(r)
    if sum(len(x) for x in corr) >= 6000:
        t = "\n".join(corr).strip(); corr = []
        if len(t) > 3000 and sum(c.isalpha() or c.isspace() for c in t) / len(t) > 0.85:
            blocchi.append(t)
print("opere riconosciute:", libri, "blocchi:", len(blocchi), "caratteri: %.1fM" % (sum(map(len, blocchi)) / 1e6))
random.shuffle(blocchi); nuovi = []; tot = 0
for t in blocchi:
    nuovi.append({"fonte": "letteratura", "text": t}); tot += len(t)
    if tot >= MC: break
altri = [json.loads(x) for x in open(f"{d}/cpt.jsonl", encoding="utf-8")]
altri = [x for x in altri if x["fonte"] != "letteratura"]
tutti = altri + nuovi; random.shuffle(tutti)
with open(f"{d}/cpt.jsonl", "w", encoding="utf-8") as f:
    for x in tutti: f.write(json.dumps(x, ensure_ascii=False) + "\n")
fonti = json.load(open(f"{d}/fonti.json")); fonti["alberto98fx/Letteratura-Italiana"].update(caratteri=tot, blocchi_disponibili=len(blocchi)); json.dump(fonti, open(f"{d}/fonti.json", "w"), ensure_ascii=False, indent=1)
import collections; c = collections.Counter()
for x in tutti: c[x["fonte"]] += len(x["text"])
s = sum(c.values()); print({k: f"{v / 1e6:.0f}M ({100 * v / s:.0f}%)" for k, v in c.items()})
print("esempio:", random.choice(nuovi)["text"][:500].replace("\n", " | "))
