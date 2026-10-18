"""EM-LLM (Fountas et al., ICLR 2025) reproduction at reduced scale on Kaggle 2xT4.

Official repo, official prediction/eval scripts, Phi-3.5-mini-instruct config (the
smallest model the repo ships a config for). Changes forced by the hardware are
listed in CHANGES and saved next to the results.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

W = "/kaggle/working"
REPO = "/kaggle/temp/EM-LLM-model" if os.path.isdir("/kaggle/temp") else "/tmp/EM-LLM-model"
DATA = os.path.dirname(glob.glob("/kaggle/input/**/lb__qasper.jsonl", recursive=True)[0])
OUT = os.path.join(REPO, "benchmark/results/phi35_mini/subset")
CHANGES = [
    "model weights loaded in float16 instead of torch_dtype=auto (bfloat16): T4 has no bf16 support",
    "min_free_cpu_memory 100 -> 8 (GB) with the repo's disk offload enabled (--allow_disk_offload True): "
    "Kaggle VM has ~29 GB RAM and run v6 was OOM-killed on NarrativeQA without offload",
    "min_block_size 8 -> 16 and max_cached_block 384 -> 200 (n_mem=3072 kept): the per-layer GPU block cache "
    "(max_cached_block x 64 tokens x 32 heads) needed 9.6 GB and ran out of memory next to the weights on a 15 GB T4",
    "one job with the model split over both T4s (the repo's use_hf_acc path) instead of one job per GPU; "
    "weights pre-downloaded to a local folder because that path needs one",
    "data: fixed-seed subset (20 per LongBench task, 10 per InfiniteBench task) matching the SketchRAG test split",
    "KV-retrieval key located by its opening quote instead of a fixed offset 6: the exported inputs have no leading space",
]


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


sh(f"git clone --depth 1 https://github.com/em-llm/EM-LLM-model {REPO}")
# The Kaggle image runs Python 3.13, which cannot build EM-LLM's pinned 2024 dependencies,
# so the repo gets its own Python 3.11 environment.
VENV = "/tmp/emllm-venv"
sh(f"{sys.executable} -m pip install -q uv")
sh(f"{sys.executable} -m uv venv -p 3.11 {VENV}")
PY = f"{VENV}/bin/python"
sh(f"{sys.executable} -m uv pip install -p {PY} 'torch==2.3.1' --index-url https://download.pytorch.org/whl/cu121")
sh(f"{sys.executable} -m uv pip install -p {PY} 'transformers==4.41.0' 'datasets==2.21.0' 'numpy<2' 'pyarrow<17' fschat omegaconf "
   "accelerate 'rouge==1.0.1' 'fuzzywuzzy==0.18.0' 'jieba==0.42.1' evaluate rouge_score pyyaml")
sh(f"cd {REPO} && {sys.executable} -m uv pip install -p {PY} -e . --no-deps")

# --- hardware patches (see CHANGES) ---
pred = os.path.join(REPO, "benchmark/pred.py")
src = open(pred).read()
assert src.count('torch_dtype="auto"') == 2
src = src.replace('torch_dtype="auto"', "torch_dtype=torch.float16")
# our KV export strips InfiniteBench's leading space, so the key's opening quote sits at index 5, not 6
old_kv = """"key": eg["input"][6:44]
            }
            assert eg['input'][6] == '"'
            assert eg['input'][43] == '"'"""
assert old_kv in src
src = src.replace(old_kv, """"key": eg["input"][eg["input"].index('"'):eg["input"].index('"') + 38]
            }""")
open(pred, "w").write(src)
cfg = os.path.join(REPO, "config/phi35_mini.yaml")
c = open(cfg).read().replace("min_free_cpu_memory: 100", "min_free_cpu_memory: 8")
c = c.replace("min_block_size: 8", "min_block_size: 16").replace("max_cached_block: 384", "max_cached_block: 200")
assert "min_block_size: 16" in c and "max_cached_block: 200" in c
# the repo's multi-GPU loader (accelerate.load_checkpoint_and_dispatch) needs a local checkpoint folder
LOCAL_MODEL = "/tmp/phi35-mini-instruct"
sh(f"{PY} -c \"from huggingface_hub import snapshot_download; "
   f"snapshot_download('microsoft/Phi-3.5-mini-instruct', local_dir='{LOCAL_MODEL}')\"")
c = c.replace("path: microsoft/Phi-3.5-mini-instruct", f"path: {LOCAL_MODEL}")
assert LOCAL_MODEL in c
open(cfg, "w").write(c)

# --- data in the layout pred.py expects ---
# v6 finished 2wikimqa, hotpotqa, multifieldqa_en, musique; v7 qasper, passage_retrieval_en; this run does the rest
ONLY = {"narrativeqa"}  # v8: KV (and PassKey, same length) OOM-killed the 29 GB VM at chunk 100/343
lb_tasks = []
for f in sorted(glob.glob(f"{DATA}/**/lb__*.jsonl", recursive=True)):
    t = os.path.basename(f)[len("lb__"):-6]
    if t not in ONLY:
        continue
    dst = os.path.join(REPO, "benchmark/data/longbench", t)
    sh(f"{PY} -c \"from datasets import Dataset; import json; "
       f"Dataset.from_list([json.loads(l) for l in open('{f}')]).save_to_disk('{dst}')\"")
    lb_tasks.append(t)
ib_dir = os.path.join(REPO, "benchmark/data/infinite-bench")
os.makedirs(ib_dir, exist_ok=True)
ib_tasks = []
for f in sorted(glob.glob(f"{DATA}/**/ib__*.jsonl", recursive=True)):
    t = os.path.basename(f)[len("ib__"):-6]
    if t not in ONLY:
        continue
    shutil.copy(f, os.path.join(ib_dir, t + ".jsonl"))
    ib_tasks.append(t)
print("tasks:", lb_tasks, ib_tasks, flush=True)

n_gpu = int(subprocess.run("nvidia-smi --list-gpus | wc -l", shell=True, capture_output=True, text=True).stdout)
gpu_name = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True,
                          text=True).stdout.strip()
order = ["qasper", "passage_retrieval_en", "kv_retrieval", "passkey", "narrativeqa"]  # NarrativeQA last: heaviest
datasets = ",".join(sorted(lb_tasks + ib_tasks, key=order.index))
# one job, model spread across all visible GPUs (pred.py sets use_hf_acc when >1 GPU is visible)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(n_gpu)))
log = open(f"{W}/worker_0.log", "w")
proc = subprocess.Popen([PY, "benchmark/pred.py", "--config_path", "config/phi35_mini.yaml", "--output_dir_path", OUT,
                         "--datasets", datasets, "--world_size", "1", "--rank", "0", "--allow_disk_offload", "True"],
                        cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
codes = [proc.wait()]
print("worker exit codes:", codes, flush=True)
print(open(f"{W}/worker_0.log").read()[-3000:], flush=True)

shutil.rmtree(os.path.join(OUT, "offload_data"), ignore_errors=True)
try:
    sh(f"cd {REPO} && {PY} benchmark/eval.py --dir_path {OUT}")
except subprocess.CalledProcessError as e:
    print("eval.py failed:", e, flush=True)
# copy results without the per-run log folders (their names contain ':' which breaks Windows downloads)
shutil.copytree(OUT, f"{W}/emllm_results", dirs_exist_ok=True, ignore=shutil.ignore_patterns("logs"))
json.dump({"changes": CHANGES, "worker_exit_codes": codes, "n_gpu": n_gpu,
           "gpu": gpu_name}, open(f"{W}/run_info.json", "w"), indent=1)
