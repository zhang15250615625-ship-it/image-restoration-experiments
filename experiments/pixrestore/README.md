# PixRestore-S 小样本预训练推理复现

本目录实际运行了 [PixRestore: Unified Image Restoration via Pixel Diffusion Transformer](https://arxiv.org/abs/2608.16793) 的公开单步 S 模型。论文首发于 **2026-08-17**，材料按 **预印本** 标注；本地实验执行于 2026-11-09，归入 2026.11.16 组会。这里复现公开模型的推理流程与自定义小样本实验，没有从头训练，也没有复现论文完整数据集、排行榜或训练消融。

官方来源：[源码](https://github.com/csslc/PixRestore)、[模型](https://huggingface.co/VCLab-PolyU/PixRestore)、[DINOv2](https://github.com/facebookresearch/dinov2)。公开包仅包含自写脚本、可公开的样图与实验结果；上游代码和权重通过脚本获取，不随包分发。

## 实际结果

4 张图、3 种固定退化、3 个模型 seed，共 **36 次不同配置预测**；每个配置重复计时 3 次并保存第一次预测，未挑选最优 seed。输入基线每任务 n=4，模型均值每任务 n=12。

| 任务 | 输入 PSNR / SSIM | PixRestore-S PSNR / SSIM | PSNR 变化 |
|---|---:|---:|---:|
| 4× bicubic 超分 | 29.1804 / 0.86329 | 28.2061 / 0.83915 | **−0.9743 dB** |
| Gaussian σ=25/255 去噪 | 20.6136 / 0.26890 | 33.2635 / 0.90161 | +12.6499 dB |
| JPEG quality=10，额外泛化检查 | 28.5409 / 0.80911 | 29.3986 / 0.84941 | +0.8577 dB |

SR 主 seed 的 4 张图中有 3 张 PSNR 下降。astronaut 面部出现伪细节，chelsea 毛发更加锐利但偏离参考图。所有负结果均保留；不能把视觉锐化直接解释为保真度提升，也没有测 LPIPS 来证明感知优势。去噪改善明确，JPEG 的有限正结果只说明这 4 张合成图上的表现。

三组 seed 的“4 图均值”的样本标准差分别为：SR 0.00517 dB、去噪 0.00239 dB、JPEG 0.00820 dB。该数值描述固定样本的采样波动，**不是总体置信区间**。同一 seed 的 3 次重复输出最大绝对差为 0。

单步 **1 NFE**；每次恢复包含 DINO 特征提取和一次 DiT 前向，三类任务的平均计时中位数约 **101 ms/图**。RTX 3060 Laptop GPU 6 GB、batch=1、512×512、BF16 autocast，峰值 PyTorch allocated 约 **327 MiB**。allocated 不等于整卡占用；reserved 原始值也保存在 CSV。参数 FP32 存储，恢复器 31,645,421 + 冻结 DINO 22,056,576 = **53,701,997 参数**。性能数值仅适用于本机与该协议。

## 快速复核已有结果：不需要 GPU

在本目录打开终端，使用 Python 3.10 的独立环境：

```powershell
python -m venv .venv-audit
.\.venv-audit\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv-audit\Scripts\python.exe verify_results.py --results results
.\.venv-audit\Scripts\python.exe make_figures.py --results results
```

Linux/macOS 把解释器路径替换为 `.venv-audit/bin/python`。CPU 审核检查输入哈希、全部浮点数组的 PSNR/SSIM、PNG 量化、CSV/JSON 对应、seed、NFE、计时中位数、汇总均值和路由归一化。已存 `results/verification.json` 为 PASS，48 行全部一致，指标重算误差为 0。此审核重算归档证据，不等同于重新执行模型。

## 从公开权重重新运行

需要 Git、Python 3.10、支持 BF16 的 NVIDIA CUDA GPU。`requirements.txt` 固定实际使用的 PyTorch 2.1.1/CUDA11.8、xformers 等版本；其他硬件/依赖版本可能产生不同指标与耗时。建议新建环境，避免修改已有科研环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_sources.py
.\.venv\Scripts\python.exe download_weights.py
.\.venv\Scripts\python.exe prepare_inputs.py
.\.venv\Scripts\python.exe run_experiment.py --output results_reproduced
.\.venv\Scripts\python.exe verify_results.py --results results_reproduced
.\.venv\Scripts\python.exe make_figures.py --results results_reproduced
```

`run_experiment.py` 不允许覆盖已有输出目录。省略 `--output` 时自动写入新的 `runs/run_日期_时间`。可用 `--seeds 20261116 --repeats 3` 先做 12 个配置的小验证，完整默认值是 3 个 seed 和 3 次计时重复。README 命令中的新建环境流程是可复现安装说明；本次实际推理使用基于既有 SinSR 科研环境的隔离 venv，记录的具体版本见 `results/run_config.json`，未修改原环境。没有另外做一次从零安装环境的验证。

源码固定为 PixRestore `909ca06e614cf72e218d52b4fa41077c603d872b`，DINOv2 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。下载脚本同时核对文件长度和 SHA256；PixRestore 权重的哈希由 Hugging Face 的 LFS 元数据给出，DINO 权重与配置哈希是本次从官方地址下载后的固定快照，哈希出处记录在 `weights/manifest.json`。若上游文件变动，脚本报错而不静默换模型。下载总量约 217 MB，另有两个源码仓库。

## 实验协议与边界

- 输入来自 `skimage.data` 原始数组，中心裁为正方形后双三次缩放到 512。astronaut 本来为 512×512；coffee 裁后 400×400、chelsea 300×300、rocket 427×427，均需上采样。因此它是自定义小样本验证，不能视为原生 512 分辨率 benchmark。
- SR：参考图 512→128 双三次下采样，再 128→512 双三次上采样作为网络输入。基线即该上采样输入。
- 去噪：独立 RGB Gaussian σ=25/255，观测 seed 为 20261001+图像编号；加噪后截断 [0,1]、量化到 PNG。对所有模型 seed 使用同一观测。该量化/截断协议与论文完整基准可能不同。
- JPEG：quality=10、4:2:0 色度子采样，解码后输入。**它不属于论文报告的八项 benchmark 任务，仅作为额外泛化检查。**
- 模型 base seed 为 20261116、20261216、20261316，实际 seed=base+图像编号。同一图像跨任务保持 seed 规则一致，不从多次输出中选最好的一次。
- 公布的 checkpoint 已经过单步微调。这里用官方 `sample_multistep_fm` 的 `n_steps=1`、linear schedule、cfg_scale=1；不是未单步微调 teacher 的 10/4/1 步消融，也不比较不同训练预算。
- 指标：输出 [-1,1]→[0,1]，截断后对 RGB 全图计算，无边界裁剪；SSIM 为 scikit-image 默认 7×7 均匀窗口、样本协方差。先用浮点数据计算，再保存量化 PNG，因此从 PNG 重算可能略有不同。未换成 Y 通道或论文中其他评估配置来追平分数。
- 计时：一次全局预热、每配置 3 次 CUDA 同步测量取中位数；包含 DINO 提取和 DiT 恢复，排除加载模型、读写文件、CPU 图像预处理和指标计算。表中再平均 12 个配置的中位数。
- 路由统计来自只读 forward hook；记录官方 softmax(float)→BF16 cast→clamp 的层权重，不改变模型结果。`[1,2,4,6,8,10]` 是 **零基 DINO block 索引**；热图每任务取 4 个不同输入平均，不能由此认定因果作用或证明论文消融结论。

## 平台适配

官方模型、采样器和检查点均未改写，权重采用官方 `load_weights` 的 `strict=True` 加载。Windows/PyTorch2.1 不支持这里的 `torch.compile`，外部 `runtime.py` 将可选编译装饰器替换为恒等包装，让原函数以 eager 模式执行；其余仍调用官方 `build_model`、`build_flow`、`load_dinov2`、`extract_layers` 和采样函数。使用单卡 batch=1 与官方 BF16 autocast，并将作者配置内的 DINO 绝对路径改为本地解析路径。公开日志/JSON 使用相对路径。

## 文件索引

| 文件 | 用途 |
|---|---|
| `prepare_inputs.py`、`data/manifest.json` | 原图、裁剪、退化、固定观测 seed 与哈希 |
| `setup_sources.py`、`download_weights.py` | 固定源码 commit 与检查点哈希 |
| `run_experiment.py`、`runtime.py` | 真实推理、计时、数值/路由记录与平台适配 |
| `results/metrics.csv`、`summary.json` | 48 行指标与全部/主 seed/seed 波动汇总 |
| `results/arrays/`、`images/` | 36 个浮点输出、PNG 可视化与输入参照 |
| `results/router_weights.json` | 36 组逐输入层权重 |
| `verify_results.py`、`results/verification.json` | 无 GPU 的指标复核及报告 |
| `make_figures.py`、`results/figures/` | 正负结果对比图、图表、无文字三列条带 |

## 图片来源与许可

本次使用的 `skimage.data` 0.24 文档将下列图像注明为 public domain 或 CC0。详细原始说明、来源链接、原图数组哈希和裁剪框保存在 `data/manifest.json`。

| 图片 | 作者/来源 | 文档记载许可 |
|---|---|---|
| [astronaut](https://scikit-image.org/docs/0.24.x/api/skimage.data.html#skimage.data.astronaut) | NASA，Eileen Collins 照片 | Public domain |
| [coffee](https://scikit-image.org/docs/0.24.x/api/skimage.data.html#skimage.data.coffee) | Rachel Michetti / Pikolo Espresso Bar | CC0 |
| [chelsea](https://scikit-image.org/docs/0.24.x/api/skimage.data.html#skimage.data.chelsea) | Stefan van der Walt | CC0 |
| [rocket](https://scikit-image.org/docs/0.24.x/api/skimage.data.html#skimage.data.rocket) | SpaceX，DSCOVR/Falcon 9 | Public domain |

这些许可说明仅适用于样图。论文、模型与上游源码仍遵循各自的许可和使用条款；本项目的复现工作不代表原作者背书。
