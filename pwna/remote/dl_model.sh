#!/bin/bash
# Download a model with fetch_parallel.py (sizes and sha256 checked): from ModelScope (files listed by its
# API), or, if that fails, from Hugging Face through the AutoDL academic proxy (files listed by the HF API).
# Then verify that transformers can read the config and the tokenizer.
# Usage: bash pwna/remote/dl_model.sh <repo id> <local dir>
export PATH=/root/miniconda3/bin:$PATH
REPO=$1
DIR=$2
cd /root/autodl-tmp/pwna_work
EXT="('.json', '.safetensors', '.txt', '.model', '.jinja')"
check() {
  python -c "from transformers import AutoConfig, AutoTokenizer; c = AutoConfig.from_pretrained('$DIR'); AutoTokenizer.from_pretrained('$DIR'); print('ok', c.model_type, c.num_hidden_layers, 'layers', c.num_attention_heads, 'heads', c.num_key_value_heads, 'kv heads')" \
    && ls $DIR/*.safetensors > /dev/null 2>&1
}
if check 2> /dev/null; then echo "$REPO already present in $DIR"; exit 0; fi
FREE_GB=$(df -BG --output=avail /root/autodl-tmp | tail -n 1 | tr -dc '0-9')
echo "free space on the data disk: ${FREE_GB} GB"
if [ "${FREE_GB:-0}" -lt 10 ] && [ -d /root/autodl-tmp/models/Qwen2.5-3B-Instruct ]; then
  echo "less than 10 GB free: removing the superseded Qwen2.5-3B-Instruct (public, re-downloadable)"
  rm -rf /root/autodl-tmp/models/Qwen2.5-3B-Instruct
  df -h /root/autodl-tmp | tail -n 1
fi
FILES=$(curl -s --max-time 60 "https://www.modelscope.cn/api/v1/models/$REPO/repo/files?Recursive=true&Revision=master" | python -c "
import json, sys
try:
    fs = (json.load(sys.stdin).get('Data') or {}).get('Files') or []
except Exception:
    fs = []
print(' '.join(f['Path'] for f in fs if f.get('Type') == 'blob' and f['Path'].endswith($EXT)))")
OK=0
if [ -n "$FILES" ]; then
  echo "ModelScope files: $FILES"
  FETCH_SOURCE=modelscope FETCH_META=modelscope python pwna/remote/fetch_parallel.py $REPO $DIR $FILES && OK=1
fi
if [ $OK -eq 0 ]; then
  echo "ModelScope failed; trying Hugging Face through the academic proxy"
  source /etc/network_turbo 2> /dev/null
  FILES=$(curl -s --max-time 60 "https://huggingface.co/api/models/$REPO" | python -c "
import json, sys
try:
    sib = json.load(sys.stdin).get('siblings', [])
except Exception:
    sib = []
print(' '.join(s['rfilename'] for s in sib if s['rfilename'].endswith($EXT)))")
  if [ -z "$FILES" ]; then echo "no files listed for $REPO"; exit 1; fi
  echo "Hugging Face files: $FILES"
  FETCH_SOURCE=hf python pwna/remote/fetch_parallel.py $REPO $DIR $FILES || exit 1
fi
check
