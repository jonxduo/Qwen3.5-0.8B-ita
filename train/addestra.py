"""Addestramento in tre fasi secondo la guida: cpt (prosa italiana), sft (istruzioni), orpo (preferenze stilistiche).
Uso: python addestra.py <fase> <cartella_base> <cartella_uscita> <dati.jsonl> [--lr X] [--max_token N] [--token_passo N] [--accumula N] [--lora_r N]
Il modello di uscita e' sempre fuso e salvato in bf16. I due tensori del checkpoint di partenza restano intatti.
"""
import argparse, json, math, random, sys, time
import torch
from transformers import AutoTokenizer
sys.path.insert(0, "/workspace/train")
from comune import carica, nascosti, loss_a_blocchi, salva, PROTETTI

ap = argparse.ArgumentParser()
ap.add_argument("fase", choices=["cpt", "sft", "orpo"]); ap.add_argument("base"); ap.add_argument("uscita"); ap.add_argument("dati")
ap.add_argument("--lr", type=float); ap.add_argument("--max_token", type=float, default=1e12); ap.add_argument("--token_passo", type=int, default=6144)
ap.add_argument("--accumula", type=int, default=2); ap.add_argument("--lora_r", type=int, default=32); ap.add_argument("--lam", type=float, default=0.1)
a = ap.parse_args()
random.seed(13); torch.manual_seed(13)
tok = AutoTokenizer.from_pretrained(a.base)
EOT = tok.convert_tokens_to_ids("<|endoftext|>")
m = carica(a.base)
prima = {n: p.detach().clone() for n, p in m.named_parameters() if any(k in n for k in PROTETTI)}
assert len(prima) == 2, list(prima)
righe = [json.loads(r) for r in open(a.dati, encoding="utf-8")]

# ---------- dati ----------
lotti = []
if a.fase == "cpt":
    L = 2048; flusso = []; n_tok = 0
    for i in range(0, len(righe), 512):
        for ids in tok([r["text"] for r in righe[i:i + 512]], add_special_tokens=False)["input_ids"]:
            flusso.extend(ids); flusso.append(EOT)
        if len(flusso) >= a.max_token:
            break
    flusso = flusso[: int(min(len(flusso), a.max_token)) // L * L]
    blocchi = [flusso[i:i + L] for i in range(0, len(flusso), L)]; random.shuffle(blocchi)
    B = max(1, a.token_passo // L)
    for i in range(0, len(blocchi) - B + 1, B):
        x = torch.tensor(blocchi[i:i + B]); y = torch.cat([x[:, 1:], torch.full((B, 1), -100)], 1)
        lotti.append((x, y))
else:
    def codifica(msgs, risposta):
        pre = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p = tok(pre, add_special_tokens=False)["input_ids"]; r = tok(risposta + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        return p, r
    def impacchetta(seqs, limite):
        # seqs: lista di (prefisso, risposta); lotti a lunghezza simile, riempiti a destra
        out = []; seqs = sorted(seqs, key=lambda s: len(s[0]) + len(s[1])); corr = []
        for s in seqs:
            lun = len(s[0]) + len(s[1])
            if corr and (-(-lun // 128) * 128) * (len(corr) + 1) > limite:
                out.append(corr); corr = []
            corr.append(s)
        if corr: out.append(corr)
        return out
    def tensori(lotto):
        Lm = -(-max(len(p) + len(r) for p, r in lotto) // 128) * 128  # lunghezze arrotondate: meno forme diverse, piu' velocita'
        x = torch.full((len(lotto), Lm), EOT); y = torch.full((len(lotto), Lm), -100)
        for i, (p, r) in enumerate(lotto):
            ids = torch.tensor(p + r); x[i, : len(ids)] = ids
            y[i, len(p) - 1: len(ids) - 1] = ids[len(p):]
        return x, y
    if a.fase == "sft":
        es = []
        for r in righe:
            p, ris = codifica(r["messages"][:-1], r["messages"][-1]["content"])
            if len(p) + len(ris) <= 1536 and len(ris) >= 2:
                es.append((p, ris))
        random.shuffle(es)
        for i in range(0, len(es), 2048):  # secchi mescolati, ordinati al loro interno per lunghezza
            lotti.extend(tensori(l) for l in impacchetta(es[i:i + 2048], a.token_passo))
        random.shuffle(lotti)
    else:
        es = []
        for r in righe:
            pc, c = codifica([{"role": "user", "content": r["prompt"]}], r["chosen"]); _, rj = codifica([{"role": "user", "content": r["prompt"]}], r["rejected"])
            c, rj = c[:384], rj[:384]
            if len(pc) <= 512:
                es.append((pc, c, rj))
        random.shuffle(es)
        for i in range(0, len(es) - 3, 4):
            g = es[i:i + 4]
            lotti.append((tensori([(p, c) for p, c, _ in g]), tensori([(p, r) for p, _, r in g])))
n_token = sum((l[0].numel() if a.fase != "orpo" else l[0][0].numel() + l[1][0].numel()) for l in lotti)
print(f"fase {a.fase}: {len(righe)} righe -> {len(lotti)} lotti, {n_token / 1e6:.1f}M token (compreso il riempimento)", flush=True)

# ---------- parametri addestrabili ----------
if a.fase == "cpt":
    for n, p in m.named_parameters():
        p.requires_grad_("language_model" in n and "embed_tokens" not in n and not any(k in n for k in PROTETTI))
    lr = a.lr or 1e-5
else:
    from peft import LoraConfig, get_peft_model
    bersagli = r".*language_model\.layers\.(?!(?:15|19)\.mlp\.down_proj)\d+\..*(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)"
    pm = get_peft_model(m, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, target_modules=bersagli))
    nomi = [n for n, p in pm.named_parameters() if p.requires_grad]
    assert not any(k in n for n in nomi for k in PROTETTI), "LoRA applicato a un tensore protetto"
    print(f"LoRA r={a.lora_r}: {len(nomi) // 2} moduli", flush=True)
    lr = a.lr or (1e-4 if a.fase == "sft" else 3e-5)
par = [p for p in m.parameters() if p.requires_grad]
print(f"parametri addestrabili {sum(p.numel() for p in par) / 1e6:.1f}M, lr {lr}", flush=True)
ott = torch.optim.AdamW(par, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
agg = math.ceil(len(lotti) / a.accumula); risc = max(1, int(0.04 * agg))
sched = torch.optim.lr_scheduler.LambdaLR(ott, lambda s: (s + 1) / risc if s < risc else 0.5 * (1 + math.cos(math.pi * (s - risc) / max(1, agg - risc))))

def log1mexp(x):
    return torch.log1p(-torch.exp(x.clamp(max=-1e-4)))

# ---------- ciclo ----------
m.train(); t0 = time.time(); visti = 0; acc_l = acc_n = 0.0; acc_or = 0.0; acc_k = 0
for i, lotto in enumerate(lotti):
    if a.fase != "orpo":
        x, y = (t.to("cuda") for t in lotto)
        h = nascosti(m, x); s, n, _ = loss_a_blocchi(m, h, y, blocco=1024)
        acc_l += s; acc_n += n; visti += x.numel()
    else:
        lp = []
        for x, y in lotto:
            x, y = x.to("cuda"), y.to("cuda"); h = nascosti(m, x); sel = y != -100
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m.lm_head(h[sel])
            tl = -torch.nn.functional.cross_entropy(lg.float(), y[sel], reduction="none")
            riga = sel.nonzero()[:, 0]
            lp.append(torch.zeros(x.shape[0], device="cuda").index_add(0, riga, tl) / sel.sum(1).clamp(min=1)); visti += x.numel()
        lpc, lpr = lp
        rapporto = (lpc - lpr) - (log1mexp(lpc) - log1mexp(lpr))
        l_or = -torch.nn.functional.logsigmoid(rapporto).mean(); l_nll = -lpc.mean()
        (l_nll + a.lam * l_or).backward()
        acc_l += l_nll.item(); acc_n += 1; acc_or += l_or.item(); acc_k += (lpc > lpr).float().mean().item()
    if (i + 1) % a.accumula == 0 or i == len(lotti) - 1:
        torch.nn.utils.clip_grad_norm_(par, 1.0); ott.step(); sched.step(); ott.zero_grad(set_to_none=True)
    if (i + 1) % 100 == 0 or i == len(lotti) - 1:
        dt = time.time() - t0; fatto = (i + 1) / len(lotti)
        extra = f" orpo {acc_or / acc_n:.3f} preferita_piu_probabile {acc_k / acc_n:.2f}" if a.fase == "orpo" else ""
        print(f"lotto {i + 1}/{len(lotti)} loss {acc_l / acc_n:.4f}{extra} lr {sched.get_last_lr()[0]:.2e} {visti / dt:.0f} tok/s  trascorsi {dt / 60:.0f} min, restanti {dt / fatto * (1 - fatto) / 60:.0f} min  picco {torch.cuda.max_memory_allocated() / 1e9:.0f} GB", flush=True)
        acc_l = acc_n = 0.0; acc_or = 0.0; acc_k = 0
    if a.fase == "cpt" and i + 1 == len(lotti) // 2:
        salva(m, a.base, a.uscita + "-meta"); m.train()

m.eval()
if a.fase != "cpt":
    m = pm.merge_and_unload()
dopo = dict(m.named_parameters())
for n, p in prima.items():
    assert torch.equal(p, dopo[n].detach()), f"tensore protetto modificato: {n}"
print("tensori protetti intatti:", list(prima), flush=True)
salva(m, a.base, a.uscita)
json.dump({"fase": a.fase, "base": a.base, "dati": a.dati, "lotti": len(lotti), "token": n_token, "lr": lr, "token_passo": a.token_passo, "accumula": a.accumula,
           "lora_r": None if a.fase == "cpt" else a.lora_r, "minuti": round((time.time() - t0) / 60, 1)}, open(a.uscita + "/addestramento.json", "w"), indent=1)
print("FINE", a.fase, "in", round((time.time() - t0) / 60, 1), "min ->", a.uscita, flush=True)
