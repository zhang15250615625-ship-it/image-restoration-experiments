# 第三方来源与许可说明

- DDRM算法和官方实现来自Bahjat Kawar等作者，NeurIPS 2022。固定代码版本32b6b3ccfda532ba01c9cc5b6e7456c3a06a6ca2。源码由setup_official.ps1获取，MIT文本见ddrm-MIT.txt。
- ImageNet 256×256无条件扩散权重来自[OpenAI guided-diffusion](https://github.com/openai/guided-diffusion)。提供官方下载脚本和SHA256，权重不随仓库分发。
- 四张测试图来自scikit-image公开样例。astronaut为NASA公有领域图片；coffee和chelsea由摄影师以CC0发布；rocket来自SpaceX公有领域发布。准确来源、完整说明和预处理记录见[输入来源](../experiments/ddrm/data/sources.json)。
- 输出图像由本项目运行预训练模型产生，原始输入及其作者信息保留在sources.json中。
- 论文理论和论文报告的原始指标归原作者；本项目实测数据单独存储在实验结果目录。
