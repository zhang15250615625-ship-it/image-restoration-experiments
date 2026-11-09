# 第三方来源与许可说明

- DDRM算法和官方实现来自Bahjat Kawar等作者，NeurIPS 2022。固定代码版本32b6b3ccfda532ba01c9cc5b6e7456c3a06a6ca2。源码由setup_official.ps1获取，MIT文本见ddrm-MIT.txt。
- ImageNet 256×256无条件扩散权重来自[OpenAI guided-diffusion](https://github.com/openai/guided-diffusion)。提供官方下载脚本和SHA256，权重不随仓库分发。
- 四张测试图来自scikit-image公开样例。astronaut为NASA公有领域图片；coffee和chelsea由摄影师以CC0发布；rocket来自SpaceX公有领域发布。准确来源、完整说明和预处理记录见[输入来源](../experiments/ddrm/data/sources.json)。
- 输出图像由本项目运行预训练模型产生，原始输入及其作者信息保留在sources.json中。
- 论文理论和论文报告的原始指标归原作者；本项目实测数据单独存储在实验结果目录。

## PixRestore 2026

- [PixRestore 官方实现](https://github.com/csslc/PixRestore)和模型由 Lingchen Sun 等作者提供，源码固定在 909ca06e614cf72e218d52b4fa41077c603d872b。官方 README 与 Hugging Face 模型卡声明 Apache-2.0；本仓库不重新分发其源码或权重，仅提供固定版本获取脚本。
- [DINOv2](https://github.com/facebookresearch/dinov2)源码固定在 7764ea0f912e53c92e82eb78a2a1631e92725fc8，按上游 Apache-2.0 使用；模型从 Meta 官方地址下载，不随仓库分发。
- PixRestore 测试图的作者、许可、原始尺寸、中心裁剪、512 缩放及输入哈希见 [输入 manifest](../experiments/pixrestore/data/manifest.json)。astronaut/rocket 为公有领域来源，coffee（Rachel Michetti）与 chelsea（Stefan van der Walt）为 CC0。输出是这些图像经退化及预训练恢复后的实验结果。
- 原论文和论文图不随仓库分发。方法归原作者，本项目自写封装负责输入准备、兼容适配、测量、存档和复核，不主张原算法贡献。
