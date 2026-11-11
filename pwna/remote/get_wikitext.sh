#!/bin/bash
# WikiText-103 (raw) validation + test as one plain-text corpus: validation first (calibration half of
# g1_llm.split_text), test second (evaluation half). Exits non-zero if the corpus cannot be built.
set -e
export PATH=/root/miniconda3/bin:$PATH
mkdir -p /root/autodl-tmp/corpora && cd /root/autodl-tmp/corpora
python -c "import pyarrow" 2>/dev/null || pip install -q pyarrow 2>&1 | tail -n 2
P=datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1
for f in test validation; do
  if [ ! -s wt103_$f.parquet ]; then
    curl -sfL --max-time 300 -o wt103_$f.parquet "https://hf-mirror.com/$P/${f}-00000-of-00001.parquet" \
      || (source /etc/network_turbo 2>/dev/null; curl -sfL --max-time 300 -o wt103_$f.parquet "https://huggingface.co/$P/${f}-00000-of-00001.parquet")
  fi
done
python - <<'EOF'
import pyarrow.parquet as pq
parts = []
for f in ("validation", "test"):          # validation first -> calibration half, test second -> evaluation half
    parts.append("".join(pq.read_table(f"wt103_{f}.parquet").column("text").to_pylist()))
txt = "\n".join(parts)
open("wikitext103_valtest.txt", "w", encoding="utf-8").write(txt)
print("chars", len(txt), "approx tokens", len(txt) // 4)
EOF
test -s wikitext103_valtest.txt
ls -la /root/autodl-tmp/corpora
