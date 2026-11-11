export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/pwna_work
sed -i 's/\r$//' pwna/remote/*.sh 2>/dev/null
pip config list 2>/dev/null | head -3
pip install -q "transformers==4.57.6" accelerate safetensors requests pyarrow 2>&1 | tail -3
python - <<'EOF'
import transformers, huggingface_hub, accelerate, torch
print("transformers", transformers.__version__, "| hub", huggingface_hub.__version__,
      "| accelerate", accelerate.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
EOF
