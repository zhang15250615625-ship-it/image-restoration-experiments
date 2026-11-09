# 图像恢复复现实验

记录图像恢复方法的代码、实验结果、结果分析和实验感悟。当前实验为 **DDRM 预训练推理复现**：调用官方采样器与公开权重，在4张固定公开图像上完成4倍超分辨率，比较恢复质量、推理步数和随机种子。

## 内容入口

| 内容 | 入口 |
|---|---|
| 代码与运行方法 | [环境、脚本和复现命令](experiments/ddrm/README.md) |
| 实验结果 | [逐图指标](experiments/ddrm/results/metrics.csv) · [配置与汇总](experiments/ddrm/results/summary.json) · [图像对比](experiments/ddrm/results/figures/comparison_all.png) |
| 实验结果分析 | [DDRM结果分析](experiments/ddrm/RESULTS_ANALYSIS.md) |
| 实验感悟 | [DDRM实验感悟](experiments/ddrm/REFLECTIONS.md) |

## DDRM结果

256×256 RGB图像经4倍块平均下采样得到64×64观测，观测噪声为0。测试图为 astronaut、coffee、chelsea、rocket；使用 RTX 3060 Laptop 6GB、FP16、batch=1。以下为主随机种子下四张图的均值。

| 方法 | RGB PSNR / dB ↑ | SSIM ↑ | 推理时间 / 秒每图 |
|---|---:|---:|---:|
| 最近邻 | 25.21 | 0.7726 | <0.01（CPU单次） |
| 双三次 | 26.91 | 0.8162 | <0.01（CPU单次） |
| DDRM 20步 | 27.61 | 0.8416 | 4.92 |
| DDRM 100步 | 28.02 | 0.8551 | 24.62 |

20步相对双三次平均提高 0.70 dB；100步相对20步提高 0.42 dB，耗时约 5.01 倍。GPU计时包含CUDA同步与官方采样器的CPU轨迹拷贝，不含权重加载，每图仅计时一次。

20步共测试3组随机种子，四图平均PSNR为27.6621 ± 0.0517 dB，±为三个运行均值的样本标准差。本实验用于观察这四张图上的行为，未重现论文完整ImageNet基准。

![四张公开图的真实恢复结果](experiments/ddrm/results/figures/comparison_all.png)

## 复现与复核

在 [实验目录](experiments/ddrm/README.md) 按说明安装依赖、下载固定版本源码与权重，然后运行实验。输入、配置、浮点输出和指标均已保留。

安装依赖后，在 `experiments/ddrm` 目录执行 `python verify_results.py`，即可用CPU独立复算存档主实验的PSNR、SSIM和观测一致性，并核对CSV、JSON与PNG；无需模型权重或GPU。

## 来源

论文与算法来自 [DDRM项目](https://ddrm-ml.github.io/)，代码来自 [官方仓库](https://github.com/bahjat-kawar/ddrm)。本项目完成推理封装、评测和分析，模型沿用公开预训练权重。数据和第三方代码的来源见 [许可说明](licenses/THIRD_PARTY_NOTICES.md)。
