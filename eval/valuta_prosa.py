"""Valutazione prima/dopo secondo la sezione 6 della guida: perplessita' su testi tenuti da parte, loss sulle risposte,
preferenze, e generazione su 20 richieste fisse con indici di leggibilita' e di difetto.
Uso: python valuta_prosa.py <cartella_modello> <uscita.json> [cartella_dati]
"""
import json, math, re, sys
import torch
from transformers import AutoTokenizer
sys.path.insert(0, "/workspace/train")
from comune import carica, nascosti, loss_a_blocchi

cart, uscita = sys.argv[1], sys.argv[2]; dati = sys.argv[3] if len(sys.argv) > 3 else "/workspace/dati"
tok = AutoTokenizer.from_pretrained(cart); m = carica(cart).eval()
EOT = tok.convert_tokens_to_ids("<|endoftext|>")
R = {"modello": cart}

def ppl(percorso):
    tot = n = 0
    for r in open(percorso, encoding="utf-8"):
        ids = tok(json.loads(r)["text"], add_special_tokens=False)["input_ids"][:1024]
        if len(ids) < 8: continue
        x = torch.tensor([ids], device="cuda"); y = torch.cat([x[:, 1:], torch.full((1, 1), -100, device="cuda")], 1)
        with torch.no_grad():
            s, k, _ = loss_a_blocchi(m, nascosti(m, x), y, con_grad=False)
        tot += s; n += k
    return round(math.exp(tot / n), 2)
R["perplessita"] = {"ufficio (scritti a mano, mai visti)": ppl("/workspace/eval/testi_ufficio.jsonl"),
                    "wikipedia tenuta da parte": ppl(f"{dati}/heldout_wikipedia.jsonl"), "web tenuto da parte": ppl(f"{dati}/heldout_fineweb.jsonl")}
print(R["perplessita"], flush=True)

def logp_risposta(msgs, risposta, limite=512):
    pre = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
    ris = tok(risposta + "<|im_end|>\n", add_special_tokens=False)["input_ids"][:limite]
    ids = torch.tensor([pre + ris], device="cuda"); y = torch.full_like(ids, -100); y[0, len(pre) - 1: -1] = ids[0, len(pre):]
    with torch.no_grad():
        s, k, _ = loss_a_blocchi(m, nascosti(m, ids), y, con_grad=False)
    return -s, k
tot = n = 0
for r in open(f"{dati}/heldout_sft.jsonl", encoding="utf-8"):
    d = json.loads(r)["messages"]
    if sum(len(x["content"]) for x in d) > 5000: continue
    s, k = logp_risposta(d[:-1], d[-1]["content"]); tot -= s; n += k
R["loss_risposte_istruzioni"] = round(tot / n, 4)
vince = k2 = 0
for r in open(f"{dati}/heldout_orpo.jsonl", encoding="utf-8"):
    d = json.loads(r); u = [{"role": "user", "content": d["prompt"]}]
    sc, kc = logp_risposta(u, d["chosen"], 384); sr, kr = logp_risposta(u, d["rejected"], 384)
    vince += (sc / kc) > (sr / kr); k2 += 1
R["preferita_piu_probabile"] = round(vince / k2, 3)
print({k: R[k] for k in ("loss_risposte_istruzioni", "preferita_piu_probabile")}, flush=True)

# ---- generazione: parametri della guida per prosa ed email ----
def indici(t):
    parole = re.findall(r"[A-Za-zÀ-ÿ']+", t); frasi = [f for f in re.split(r"(?<=[.!?])\s+", t.strip()) if f]
    if not parole: return None
    lettere = sum(len(p) for p in parole)
    p100 = [p.lower() for p in parole[:100]]
    quad = [tuple(parole[i:i + 4]) for i in range(len(parole) - 3)]
    return {"parole": len(parole), "gulpease": 89 + (300 * len(frasi) - 10 * lettere) / len(parole), "ttr100": len(set(p100)) / len(p100),
            "gerundio_iniziale": sum(bool(re.match(r"^[\W]*[A-Za-zÀ-ÿ]+(ando|endo)\b", f)) for f in frasi) / max(1, len(frasi)),
            "quadrigrammi_ripetuti": 1 - len(set(quad)) / max(1, len(quad)),
            "non_latino": len(re.findall(r"[぀-ヿ一-鿿Ѐ-ӿ؀-ۿ가-힯]", t)) / max(1, len(t)),
            "emoji": len(re.findall(r"[\U0001F300-\U0001FAFF☀-➿]", t))}
gen = []
for r in open("/workspace/eval/prompt_prosa.jsonl", encoding="utf-8"):
    d = json.loads(r)
    ids = tok(tok.apply_chat_template([{"role": "user", "content": d["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt").to("cuda")
    torch.manual_seed(1234)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        g = m.generate(**ids, max_new_tokens=320, do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.12)
    nuovi = g[0, ids["input_ids"].shape[1]:]
    t = tok.decode(nuovi, skip_special_tokens=True).strip()
    gen.append({**d, "risposta": t, "troncata": bool(len(nuovi) >= 320), "indici": indici(t)})
R["generazioni"] = gen
ok = [g["indici"] for g in gen if g["indici"]]
R["medie"] = {k: round(sum(i[k] for i in ok) / len(ok), 4) for k in ok[0]}
R["medie"]["troncate"] = sum(g["troncata"] for g in gen); R["medie"]["vuote"] = len(gen) - len(ok)
R["medie"]["con_caratteri_non_latini"] = sum(1 for i in ok if i["non_latino"] > 0); R["medie"]["con_emoji"] = sum(1 for i in ok if i["emoji"] > 0)
print(R["medie"], flush=True)
json.dump(R, open(uscita, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
