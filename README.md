# 私有权重的空中归一化聚合（分布式注意力）

研究对象是"比值"型聚合 $\sum_j d_j \bar o_j / \sum_j d_j$（softmax/LSE 合并、核回归），用空中计算（AirComp）在多址信道上实现，权重 $d_j$ 只在设备本地可知。

- **论文：** `paper/twc/main.pdf`（IEEE TWC 投稿稿，IEEEtran 双栏，13 页）。
- **方案：**
  - 部件 A，饱和感知分裂比值：设备对实际到达的权重做安全聚合，比值仍然是凸组合。
  - 部件 B，噪声匹配电平上报：预先登记 κ=1。
  - 握手：以数字握手（Z₂₅₆ 上的温度计编码）为主，音调握手是低开销选项。
- **应用：**
  - KV 分片的边缘 LLM 解码：模型为 Qwen3-4B、Llama-3.1-8B、Qwen3-8B，最多 128 台设备，上下文最长 32k。
  - 传感器空间场回归。

## 目录

| 路径 | 内容 |
|---|---|
| `paper/twc/` | 论文：`main.tex`、`sections/`（正文各节和附录）、`refs.tex`。<br>`tables/` 由脚本生成；`figures/` 放图的 PDF 和生成脚本 `make_figures.py`。 |
| `pwna/` | 仿真、实验与分析代码（见下表）。 |
| `pwna/remote/` | GPU 服务器上的实验脚本（见"GPU 实验"一节）。 |
| `results/` | 论文用到的全部原始结果和分析输出，见 `results/README.md`。 |

### 代码

| 文件 | 作用 |
|---|---|
| `ota.py` | 空口聚合模块：信道、合并器、各类方案的功率控制、握手 |
| `shard_llm.py` | KV 分片注意力的注入引擎 |
| `g1_llm.py` | LLM 实验主脚本 |
| `field_sim.py` | 空间场仿真 |
| `theory_mc.py` | 理论结果的蒙特卡洛验证，共 69 项 |
| `p8_analysis.py` | 结果分析，输出 `results/p8_analysis/summary.md` |
| `p8_tables.py` | 生成论文表格和 `results/p8_analysis/paper_numbers.md` |
| `combiner_opt.py` | 合并器优化 |
| `fixedpoint_range.py` | 安全聚合的定点环大小 |
| `s_dispersion.py` | 瓶颈尺度 S 的离散度 |
| `sweep_summary.py` | 所需 SNR 的插值 |

## 复现论文（Windows / PowerShell，Python：numpy、scipy、matplotlib）

```powershell
cd <工程根目录>
python pwna\remote\p8_verify.py           # 校验 results\p8：应为 43 jobs OK
python pwna\theory_mc.py                  # 理论验证：应为 69/69 通过（约 30 秒）
python pwna\p8_analysis.py                # → results\p8_analysis\summary.md
python pwna\p8_tables.py                  # → paper\twc\tables\tab_*.tex、results\p8_analysis\paper_numbers.md
python paper\twc\figures\make_figures.py  # → paper\twc\figures\*.pdf
cd paper\twc; latexmk -pdf main.tex       # 编译论文（TeX Live / TinyTeX）
```

## GPU 实验

LLM 部分需要 RTX 5090（32 GB）级别的显卡。工程复制到服务器的 `/root/autodl-tmp/pwna_work` 后：

- `bash pwna/remote/start_p8.sh`：
  - 准备环境（`setup_env.sh`）；
  - 下载模型和语料（`dl_model.sh`、`fetch_parallel.py`、`get_wikitext.sh`）；
  - 启动 `p8_runner.py`，按任务列表调度全部实验。
- `python pwna/remote/p8_runner.py --supp`：负 SNR 补充实验。
- `python pwna/remote/p8_runner.py --list`：列出全部任务及其命令行。
- `p7_runner.py`：对照组任务（`results/p7`），用法相同。
- `p7_verify.py` / `p8_verify.py`：逐个核对结果文件与任务定义是否一致。

## 许可

见 `LICENSE`。
