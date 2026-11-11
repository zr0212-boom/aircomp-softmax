# 结果

论文 `paper/twc/main.pdf` 里的全部数字、表和图，都由这个文件夹里的数据经脚本生成。每个任务有一个结果文件 `*.json` 和对应的运行输出 `*.out`。

## 目录

| 路径 | 内容 |
|---|---|
| `p8/` | 主实验，43 个任务（`pwna/remote/p8_runner.py`）。 |
| `p8_supp/` | 负 SNR 补充实验（−15 到 −2.5 dB），18 个任务（`p8_runner.py --supp`）。信道抽样与 SNR 无关，分析时直接并入对应曲线。 |
| `p7/` | 对照组，46 个任务（`pwna/remote/p7_runner.py`）。与 `p8/` 用相同的协议和种子，逐单元配对。 |
| `p8_analysis/` | 分析输出：<br>`summary.md`：全部分析，包括 κ 的预先登记选择、所需 SNR、配对自助法 95% 置信区间；<br>`paper_numbers.md`：正文数字对照表。 |
| `local/` | 本地仿真：<br>`theory_mc.json`：理论蒙特卡洛，69/69 通过；<br>`field_{nw,poe}_sw{0,10}_p8.json`：空间场仿真。 |
| `combiner/` | 合并器优化（maxmin 与 invsum 两种目标），输入为 `p7/p7_f5_comb_q4.npz`。 |

模型缩写：`q4` = Qwen3-4B-Instruct-2507（主模型），`l8` = Llama-3.1-8B-Instruct，`q8` = Qwen3-8B。

## 任务与论文位置

| 论文位置 | 任务 |
|---|---|
| κ 的预先登记（开发集） | `p8_f1_q4_pydoc`、`p7_f3_pydoc` |
| 主表（Table II）、保真度曲线图 | `p8_f1_{q4,l8,q8}_{8k,16k,32k}` 与配对的 `p7_f1_*`，加上 `p8_supp/p8s_f1_*` |
| 设计消融表（Table III） | `p8_f2_kap`、`p8_f2_db15`、`p8_f2_db6`、`p8_f2_r2`、`p8_f2_r8` |
| 一轮式设计 | `p8_f3_oneround`、`p7_f3_blind_*` |
| 信道损伤表（Table IV） | `p8_f5_*`（逐头信道、静态信道、相干块、相位误差、相位斜坡、CSI 误差、大尺度扩展、组合场景），加上 `p8_supp/p8s_f5_*` |
| NIAH 表（Table V） | `p8_f6_*` |
| 安全数字合并的环大小与时延表（Table VI）、工作区间图 | `p8_f4_*`、`p8_f1_q4_8k_nsweep`、`p7_f1_q4_8k_nsweep`、`p7_f1_q4_8k_zf` |
| 空间场表（Table VII） | `local/field_*_p8.json` |
| 第 V 节的合并器数字 | `combiner/` |
| 理论验证 | `local/theory_mc.json` |

κ 只用原始 0–25 dB 网格上的开发集结果选择（预先登记），补充实验的单元不参与。
