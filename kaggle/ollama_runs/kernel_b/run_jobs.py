"""Run SketchRAG experiment jobs on a Kaggle GPU with a local Ollama server.

The bundle dataset holds the repo code (src/, scripts/), the frozen test examples and the
embedding cache, so the kernel does no data preparation of its own. JOBS is edited per push.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

MODEL = "qwen2.5:7b-instruct"
NUM_CTX = 32768
WORKERS = 4
# Every job is one scripts/run_system.py invocation (resumable); the run name sets results/runs/<run>.
TASKS = "narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum passage_retrieval_en passkey kv_retrieval"
SYSTEMS = ("bm25_rag dense_rag sara_lite sketchrag_sketch sketchrag_gate sketchrag_raw sketchrag_hybrid "
           "sketchrag_agent amber_lite full_context")
# Split over two kernels (2 GPU slots): A = main comparison + EM-LLM-matched runs,
# B = ablations and budget sweeps at 10 questions per task.
PART = "B"
if PART == "A":
    EM_LB = "narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique passage_retrieval_en"
    EM_SYS = "dense_rag bm25_rag sketchrag_gate sketchrag_raw full_context"
    JOBS = [  # in priority order; later ones are skipped if the time budget runs out
        f"--run main --tasks {TASKS} --n 30 --systems {SYSTEMS}",
        f"--run main --tasks longbook_qa_eng --n 20 --systems {SYSTEMS}",
        # same base LLM as the EM-LLM reproduction (Phi-3.5-mini) on the same question subset
        f"--run emllm_match --tasks {EM_LB} --n 20 --systems {EM_SYS} --model phi3.5 --num-ctx 32768",
        f"--run emllm_match --tasks passkey kv_retrieval --n 10 --systems {EM_SYS} --model phi3.5 --num-ctx 32768",
    ]
else:
    JOBS = []
    # ablations of the frozen SketchRAG-gate config (one override each); baseline rows come from the same ids in main
    for ov in ["ret.w_lex=0.5", "ret.contiguity=0", "ret.cooccur=false", "ret.replace_digest=false",
               "mem.pointer_mode=reservoir", "ret.confidence=noise"]:
        JOBS.append(f"--run ablate --tasks {TASKS} --n 10 --systems sketchrag_gate --set {ov}")
    # memory budget sweeps (RQ3/RQ5); the frozen 64-cell / 512-width point is run too so each curve is self-contained
    for b in [8, 16, 32, 64, 128, 256]:
        JOBS.append(f"--run sweep_cells --tasks {TASKS} --n 10 --systems sketchrag_gate sketchrag_sketch --set mem.max_cells={b}")
    for w in [64, 128, 512, 2048]:
        JOBS.append(f"--run sweep_cms --tasks {TASKS} --n 10 --systems sketchrag_gate --set mem.cms_width={w}")
TIME_BUDGET_S = 11.3 * 3600  # stop launching jobs before Kaggle's 12 h limit
T0 = time.time()
W = "/kaggle/working"
REPO = "/tmp/sketchrag"  # outside /kaggle/working so the output holds only results


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


# ---- repo bundle (Kaggle may or may not have extracted the zip) ----
os.makedirs(REPO, exist_ok=True)
# the zip is uploaded as .zip.bin so Kaggle stores it as-is instead of auto-extracting it
zipfile.ZipFile(glob.glob("/kaggle/input/**/sketchrag_bundle.zip.bin", recursive=True)[0]).extractall(REPO)
os.makedirs(f"{REPO}/cache", exist_ok=True)
shutil.copy(glob.glob("/kaggle/input/**/emb.sqlite", recursive=True)[0], f"{REPO}/cache/emb.sqlite")
os.makedirs(f"{REPO}/results/runs", exist_ok=True)
for prev in glob.glob("/kaggle/input/**/results/runs/*", recursive=True):  # resume from an earlier output
    shutil.copytree(prev, f"{REPO}/results/runs/{os.path.basename(prev)}", dirs_exist_ok=True)
# Part B runs n=10, which the bundle doesn't freeze. The seed-0 subsample is nested, so the n=10 questions
# are a subset of the frozen n=30 files; their ids were computed locally with loaders.load_longbench/load_infbench.
N10_IDS = {"narrativeqa": ["narrativeqa/949abce7e7abc634fc08ff28f996a235ddd6041358bee155", "narrativeqa/e726a44d5d159de71612e7733d453f36d881d6f19f484fe4", "narrativeqa/e1514d530becf70246cb43e50b44a1c615cbbded416ccb15", "narrativeqa/fb2d8ee43b24c62eaa479e62d1ac1fabdbb461cc1a60099b", "narrativeqa/8317d375904e2fcba00401b32177f783d5f8d348a1ef503f", "narrativeqa/cb33b635d4a123fa9708b209bbfa323f19af169d0b71bb2f", "narrativeqa/33a9901ae9f88659b67c4f203e1a37b1c346d54c42b5b507", "narrativeqa/379ccfa193f4938199579c116c1faa1fa5a90c370c02c280", "narrativeqa/e76cb4d216f09e3bab6d769ae770429fe10498e6a2ca9aad", "narrativeqa/fdf07620fd1004f6da364b2f495439949740b9ea217ba467"], "qasper": ["qasper/04af9dc96013a1bc7faecbc589f7ea5c207e92c7d9a3495e", "qasper/2589b46ee58a7600e17fa89a0d4fffd9a0faf1df49b3c035", "qasper/f3aba3579b9e3373ce708f10b33510d6a198c3bae58c5ad7", "qasper/466bd29bcab1cdfdef327777808236bd2677e2a54414a32a", "qasper/0e83a6f7ee840931e1851402cc87bd34f52fe8bfa4dc1cab", "qasper/aaaf5a6e22afc0f84ee63ab9e8dfe9484462ef105aa44ec6", "qasper/9c060535b14310dff1d863489d6ecd52f7ecffda0dcc9fcb", "qasper/85a7083bae278e83df0fea37003937a9294638f488a18a83", "qasper/78224546bfb859e5412a0d5ef1b030ea0b3844c9987c79ff", "qasper/3b3fcd0ee773501a21c5eb6159746fa29bdc3498329525e5"], "multifieldqa_en": ["multifieldqa_en/04b588ff2dea15f4a9c4fdbaabc55aaad1ba3446114d6741", "multifieldqa_en/34f9b2c3d79ced580687f9653a6908215f79b1405a918cd0", "multifieldqa_en/d1f31d2998513a4552d0419e16f8eba896e9ea1c8403605b", "multifieldqa_en/2d48bbc2bd6052c5e67824752a9296cbbc26351616cd6d7e", "multifieldqa_en/462011b6f9beb976aaf7b38082bcf7d70a91c3c74d2c6e95", "multifieldqa_en/46b15f1200c46251053ec3dfa806dbdf515eb34053a5e0d1", "multifieldqa_en/5a18017502d827b72fc74d91f13f6f14193b4964c3420e47", "multifieldqa_en/101225d103936ad083c5da47c3dce73ce569b62cbaa093bd", "multifieldqa_en/73b9b4130b4a4931bed5ea7e24331ce61b59271eac6c6c01", "multifieldqa_en/f9b9a553107eed5b994a9aaf07e8ccac487feaa8ff108ec6"], "hotpotqa": ["hotpotqa/3a75aba6bc2496016339474aef28ffdc9dbebdc00ccc166b", "hotpotqa/ff048b34a9405c880ccf710acdf1a53244e6039cb76c793a", "hotpotqa/4aee23759e842807b156492356b920a11236c7ea3da20732", "hotpotqa/3df75e9b85781f83af8d3011cedb0c5fc6725f3489e48e54", "hotpotqa/a2bb6c8de2a57c51b35e9d5c005cf46e1c0457f4086ef77d", "hotpotqa/887be3f3f2080532b3c0f1dc4d26be4f8a06030bf2e68a1a", "hotpotqa/e9b828ee7169d1978f3bcb6de5bbb0f54f9ee3bfc0ffdb8b", "hotpotqa/d3249d0550c7105bdd4ed60c9ca9d955db08d362574c2086", "hotpotqa/c1cfe7334235bf4a9ef11293d8be2d4055a70ac89334be2c", "hotpotqa/c5d6aa8cb0ad17a88eb66d578e86f783e3ac1f3684df9425"], "2wikimqa": ["2wikimqa/8dedfb4f00b873379cb4457d81113516b8db6c4017a1c912", "2wikimqa/0a64d8873482d91efc595a508218c6ce881c13c95028039e", "2wikimqa/259e029da053a54f862f2455f3b92fb5f1b810647f1c1825", "2wikimqa/70923ff731dcad518ecdcdc8b53e165e8515a96ecf8f23c7", "2wikimqa/ececb22cb4bba60194702771c0d4de89658a4798d7efa1a1", "2wikimqa/544828db02efd1313bfc0a1bdeb8c3df5985c299254b0222", "2wikimqa/34896de8d6a2c70e09b6dc3d113475a58899961d2aa356b3", "2wikimqa/49035ecce6e78a355307dd6a1ddb11a87cab76090d655d2f", "2wikimqa/78436f55ae9c254da5a54558ddd196c94850ba7a2b6fa637", "2wikimqa/c1d354f1d1855b21eb3c3cd204d5030080ef9e3c7e4b0195"], "musique": ["musique/865a174ebe32ad81f573feb122198c1f16e6f2fd6c3fdda5", "musique/9f271f822294cc528b1f20904bb7694884af4084e086d20f", "musique/0da908a928c375084638f1e0ab10c329679eaacae7e4ffe0", "musique/f9a50652b4e5d7000c42ae27b2477a54d2a88cfceec0ec3b", "musique/5188d3c87c6fead7f796c62c3f1bd13ea8576bab724ea06d", "musique/2b182c3ce4bb7bb2c28fc1f86d7a76591488e7d2aef1bb35", "musique/37fb18d72ff590a69ba9588553a36989161f9b48c72dfe29", "musique/fe8275738f337cc423752bea9b254da45cecdf2559f38352", "musique/a3854b7b279e499bd9263a9c581a5efb025fea9efefa00e6", "musique/a6e0032c62ece331c56364ced53e1c4076c303e4ae287074"], "gov_report": ["gov_report/c325a40777333d0e9c79489ab40a509f45e261d2054ea3d7", "gov_report/38b8e4cacdb00469ee30fe44f4b285ce26e6d34d65883186", "gov_report/45c9d84fac2d35cc13b5a0b6716a3d1a9a5ccff921c4dd6a", "gov_report/8ea6dbec3f301d7d0fc538441e0a776832f777a928c44666", "gov_report/94f563e56143c3d06aefcf8e166786722d595c1f4d83cd2a", "gov_report/f67887ca668cbec0cc6bec7c9f079d3c7fe75962957e0f2d", "gov_report/782123f3af90344e3508996b17d47cf959a4521d6b1886de", "gov_report/642cbca15cea50e376cef6d3e81f3dc4e046cc649f6119f4", "gov_report/679f8b9b9a028f34abb79be22c978d1d4b377ce6ca1f12f9", "gov_report/86a657d0330ec4cd8a81c74c1ad8e0c1ab6cab7c1d321292"], "qmsum": ["qmsum/6a093742535636fe9687002f2db47be1d5af18de447655fe", "qmsum/8579ec587c40c2c562a4cf69c8eddcb9133b1bdacf585d15", "qmsum/ccd19ca5bb434c2d2489bf6f0377d7b7ecbbba427e369cee", "qmsum/5770240f11b9eae7d4a9a8a8de1e70bd557229796d9324f0", "qmsum/c89e36776d5ceeb609d0033cb4d2fb227500190a519e297b", "qmsum/8ef12c4d77bfb1f68893c483a13032694595f971f23f723e", "qmsum/baa9a253fa531bf3e7060a1137a38c9441c66e7611440d16", "qmsum/a0d685f51d4520fac9c8c7139e76c889302e3f6f98cb8d81", "qmsum/628c3567292cb09e980d4d2e3042ebfcc44f369a2a16d2fd", "qmsum/c0534dea030f2e295fce90c3249f37a019aa216a8716a032"], "passage_retrieval_en": ["passage_retrieval_en/bd9d3468c50000f39e476ba33358347ef358d49f2424458c", "passage_retrieval_en/8ef79d61d06b7273cde31281fc54ebdeb77f35f160123b92", "passage_retrieval_en/962c3a326d5d38b018350a3ebf2fe641028d35e77bf31505", "passage_retrieval_en/f9b33a4108ee7edd32b71191142bd9e039c49023b67d23ca", "passage_retrieval_en/9fc26f96b43015e87a6731d87252dfd2a05822394e24447b", "passage_retrieval_en/159714eaac7cd6b32bee0acf029b3521ae89350872592aa3", "passage_retrieval_en/4990fa084cf8619c119026d36dfe6df5929260f0c0ee4998", "passage_retrieval_en/607bb749cf081c01e18730aae6248370390dc014dd2f5b3b", "passage_retrieval_en/3a8663f0b412dc84f24381a904b283e56f8c17138572ba3c", "passage_retrieval_en/12770e16fbd9c64011ecbfe6117e8b96ccaefb5176c0a57e"], "passkey": ["passkey/17", "passkey/34", "passkey/197", "passkey/220", "passkey/236", "passkey/250", "passkey/316", "passkey/335", "passkey/396", "passkey/447"], "kv_retrieval": ["kv_retrieval/26", "kv_retrieval/56", "kv_retrieval/108", "kv_retrieval/130", "kv_retrieval/201", "kv_retrieval/366", "kv_retrieval/370", "kv_retrieval/419", "kv_retrieval/431", "kv_retrieval/459"]}
for t, keep in N10_IDS.items():
    src, dst = f"{REPO}/data_raw/examples/{t}_n30_s0.jsonl", f"{REPO}/data_raw/examples/{t}_n10_s0.jsonl"
    rows = {json.loads(l)["id"]: l for l in open(src, encoding="utf-8") if l.strip()}
    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(rows[i] for i in sorted(keep, key=list(rows).index))
sh(f"{sys.executable} -m pip install -q rank_bm25 faiss-cpu")

# ---- Ollama ----
sh("apt-get update -qq && apt-get install -y -qq zstd > /dev/null")  # Ollama tarballs are .tar.zst
sh("curl -fsSL https://ollama.com/install.sh | sh 2>&1 | tail -5")
env = dict(os.environ, OLLAMA_NUM_PARALLEL=str(WORKERS), OLLAMA_KEEP_ALIVE="-1", OLLAMA_FLASH_ATTENTION="1",
           OLLAMA_KV_CACHE_TYPE="q8_0")
server = subprocess.Popen(["ollama", "serve"], env=env, stdout=open(f"{W}/ollama.log", "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        break
    except Exception:
        time.sleep(2)
for m in (MODEL, "phi3.5") if PART == "A" else (MODEL,):
    sh(f"ollama pull {m}", stdout=subprocess.DEVNULL)
sh("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv")

run_env = dict(os.environ, OLLAMA_HOST="http://127.0.0.1:11434", SKETCHRAG_RAW=f"{REPO}/data_raw",
               PYTHONUNBUFFERED="1")
for job in JOBS:
    if time.time() - T0 > TIME_BUDGET_S:
        print("time budget reached; skipping:", job, flush=True)
        continue
    extra = "" if "--model" in job else f" --model {MODEL} --num-ctx {NUM_CTX}"
    cmd = (f"cd {REPO} && timeout {int(TIME_BUDGET_S - (time.time() - T0))} {sys.executable} scripts/run_system.py "
           f"{job}{extra} --workers {WORKERS}")
    subprocess.run(cmd, shell=True, env=run_env)

shutil.copytree(f"{REPO}/results", f"{W}/results", dirs_exist_ok=True)
server.terminate()
print("total hours", (time.time() - T0) / 3600, flush=True)
