export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/pwna_work
R=/root/autodl-tmp/pwna_results/p8
mkdir -p $R /root/autodl-tmp/models /root/autodl-tmp/corpora
sed -i 's/\r$//' pwna/remote/*.sh
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python -c "import torch, transformers; assert torch.cuda.is_available() and transformers.__version__ == '4.57.6'" 2>/dev/null \
  || bash pwna/remote/setup_env.sh
python -c "import sys; sys.path.insert(0, 'pwna'); sys.path.insert(0, 'pwna/remote'); import ota, shard_llm, g1_llm, s_dispersion, fixedpoint_range, combiner_opt, p8_runner; import transformers, torch; print('import ok, transformers', transformers.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" || { echo "IMPORT FAILED"; exit 1; }
ls /root/autodl-tmp/models
df -h /root/autodl-tmp | tail -n 1
free -g | head -n 2
nproc
if pgrep -f p8_runner.py > /dev/null; then echo "p8 runner already running"; exit 0; fi
if [ -f $R/ALL_DONE ]; then echo "p8 already finished"; exit 0; fi
nohup python pwna/remote/p8_runner.py >> $R/p8_runner.log 2>&1 &
sleep 10
pgrep -f p8_runner.py > /dev/null && echo "P8_STARTED_$((3+5))"
tail -n 5 $R/p8_runner.log
