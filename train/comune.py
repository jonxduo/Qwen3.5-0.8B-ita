"""Funzioni comuni: caricamento, loss a blocchi (il vocabolario da 248k rende i logit enormi), salvataggio."""
import shutil
from pathlib import Path
import torch
from transformers import AutoModelForMultimodalLM

# Tensori modificati dal checkpoint weight-space-programming: non vanno toccati.
PROTETTI = ("layers.15.mlp.down_proj", "layers.19.mlp.down_proj")

def carica(cartella):
    m = AutoModelForMultimodalLM.from_pretrained(cartella, dtype=torch.float32, device_map="cuda")
    for p in m.parameters():
        p.requires_grad_(False)
    return m

def nascosti(m, ids):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m.model(input_ids=ids).last_hidden_state

def loss_a_blocchi(m, h, etichette, blocco=2048, con_grad=True):
    """Cross-entropy sulle sole posizioni etichettate, calcolata a blocchi.
    h: stati nascosti [B, L, D]; etichette: [B, L] gia' allineate (etichetta in posizione t = token t+1), -100 = ignora.
    Con con_grad=True fa backward blocco per blocco e propaga il gradiente a h. Restituisce (somma_loss, n_token, logp_per_riga)."""
    sel = etichette != -100
    righe = sel.nonzero()[:, 0]
    hs, ys = h[sel], etichette[sel]
    n = ys.numel()
    logp_riga = torch.zeros(h.shape[0], device=h.device)
    if not con_grad:
        tot = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, n, blocco):
                lp = -torch.nn.functional.cross_entropy(m.lm_head(hs[i:i + blocco]).float(), ys[i:i + blocco], reduction="none")
                tot -= lp.sum().item(); logp_riga.index_add_(0, righe[i:i + blocco], lp)
        return tot, n, logp_riga
    foglia = hs.detach().requires_grad_(True)
    tot = 0.0
    for i in range(0, n, blocco):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m.lm_head(foglia[i:i + blocco])
        l = torch.nn.functional.cross_entropy(lg.float(), ys[i:i + blocco], reduction="sum") / n
        l.backward(); tot += l.item() * n
    hs.backward(foglia.grad)
    return tot, n, logp_riga

def salva(m, base, uscita):
    Path(uscita).mkdir(parents=True, exist_ok=True)
    m.to(torch.bfloat16).save_pretrained(uscita, safe_serialization=True)
    m.to(torch.float32)
    # Il tokenizer non cambia: si copiano i file del modello di base.
    for f in Path(base).iterdir():
        if f.is_file() and f.suffix in {".json", ".jinja", ".txt"} and f.name not in {"config.json", "generation_config.json", "model.safetensors.index.json"}:
            shutil.copy(f, uscita)
    for f in ("LICENSE",):
        if (Path(base) / f).exists():
            shutil.copy(Path(base) / f, uscita)
