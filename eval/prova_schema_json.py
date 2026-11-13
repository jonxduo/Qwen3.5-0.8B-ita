"""llama-server con decodifica vincolata dallo schema JSON.
Per ogni GGUF: avvia il server (solo CPU), esegue i casi con e senza schema, valida l'uscita,
controlla che le citazioni siano letterali e confronta con l'uscita in precisione piena.
Uso: python p1_server_schema.py <sistema.txt> <schema.json> <casi.jsonl> <rif_hf.jsonl|-> <uscita_prefisso> <gguf>...
"""
import json, re, subprocess, sys, time, urllib.request
import jsonschema

sistema_path, schema_path, casi_path, rif_path, prefisso, *ggufs = sys.argv[1:]
SERVER = "/workspace/llama.cpp/build/bin/llama-server"
PORTA = 8090
THREADS = "8"  # vicino a un PC d'ufficio, non ai 96 core del pod
sistema = open(sistema_path, encoding="utf-8").read().strip()
schema = json.load(open(schema_path, encoding="utf-8"))
schema.pop("$comment", None)
casi = [json.loads(r) for r in open(casi_path, encoding="utf-8")]
rif = {} if rif_path == "-" else {json.loads(r)["id"]: json.loads(r)["risposta"] for r in open(rif_path, encoding="utf-8")}

def norm(s):
    return re.sub(r"\s+", " ", s).strip()

def chiama(testo, con_schema):
    corpo = {"messages": [{"role": "system", "content": sistema}, {"role": "user", "content": testo}],
             "temperature": 0, "max_tokens": 600, "chat_template_kwargs": {"enable_thinking": False}}
    if con_schema:
        corpo["response_format"] = {"type": "json_schema", "json_schema": {"name": "uscita", "schema": schema, "strict": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORTA}/v1/chat/completions", json.dumps(corpo).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))

def carica(s):
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return None

for gguf in ggufs:
    nome = gguf.split("/")[-1].removesuffix(".gguf")
    log = open(f"/workspace/log/server_{nome}.log", "w")
    proc = subprocess.Popen([SERVER, "-m", gguf, "--port", str(PORTA), "-c", "4096", "-t", THREADS, "--jinja", "-np", "1"],
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORTA}/health", timeout=2).status == 200:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        else:
            raise SystemExit(f"{nome}: il server non parte, vedi {log.name}")
        for con_schema in (False, True):
            st = dict(casi=0, json_ok=0, schema_ok=0, cit=0, cit_ok=0, uguale_hf=0, think=0, tok=0, ms=0.0, ms_prompt=0.0, tok_prompt=0)
            with open(f"{prefisso}_{nome}_{'schema' if con_schema else 'libero'}.jsonl", "w", encoding="utf-8") as out:
                for c in casi:
                    r = chiama(c["testo"], con_schema)
                    msg = r["choices"][0]["message"]
                    testo = msg.get("content") or ""
                    st["casi"] += 1
                    st["think"] += int("<think>" in testo or bool(msg.get("reasoning_content")))
                    t = r.get("timings", {})
                    st["tok"] += t.get("predicted_n", 0); st["ms"] += t.get("predicted_ms", 0.0)
                    st["tok_prompt"] += t.get("prompt_n", 0); st["ms_prompt"] += t.get("prompt_ms", 0.0)
                    obj = carica(testo)
                    if obj is not None:
                        st["json_ok"] += 1
                        try:
                            jsonschema.validate(obj, schema); st["schema_ok"] += 1
                        except jsonschema.ValidationError:
                            pass
                        if isinstance(obj, dict):
                            for s in obj.get("scadenze", []) or []:
                                cit = s.get("citazione") if isinstance(s, dict) else None
                                if isinstance(cit, str):
                                    st["cit"] += 1; st["cit_ok"] += int(norm(cit) in norm(c["testo"]))
                        if c["id"] in rif and carica(rif[c["id"]]) == obj:
                            st["uguale_hf"] += 1
                    out.write(json.dumps({"id": c["id"], "risposta": testo}, ensure_ascii=False) + "\n")
            vel = st["tok"] / (st["ms"] / 1000) if st["ms"] else 0
            velp = st["tok_prompt"] / (st["ms_prompt"] / 1000) if st["ms_prompt"] else 0
            print(f"{nome:28s} {'schema' if con_schema else 'libero':7s} json {st['json_ok']}/{st['casi']}  schema {st['schema_ok']}/{st['casi']}  "
                  f"citazioni letterali {st['cit_ok']}/{st['cit']}  uguale a HF {st['uguale_hf']}/{st['casi']}  ragionamento {st['think']}  "
                  f"gen {vel:.0f} tok/s  prompt {velp:.0f} tok/s  ({THREADS} thread)")
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True).stdout.strip()
        print(f"{nome:28s} memoria residente del server: {int(rss or 0)/1024:.0f} MB (contesto 4096)")
    finally:
        proc.terminate(); proc.wait(timeout=30); log.close()
