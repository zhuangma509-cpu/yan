## East-Asian Aesthetic Facial Beauty Prediction (FBP)

一个基于双分支特征融合网络（Dual-Branch）的颜值打分项目：
- 分支 A：视觉隐式特征（MobileNetV3/ResNet）
- 分支 B：几何显式特征（三庭五眼先验）
- 融合后输出 1~100 分的标签分布与期望分数

> 说明：本项目用于学术与工程研究，不用于任何歧视性或不当用途。  
> 审美打分天然具有主观性与文化偏差，模型结果不应作为严肃决策依据。

### 1. 项目亮点

- 双分支架构：图像语义 + 几何先验融合，兼顾隐式视觉与显式比例
- LDL（Label Distribution Learning）：不仅预测单点分数，还预测分布
- 复合损失：
  - KL 散度（预测分布 vs 标签分布）
  - 期望分回归 MSE
  - 几何正则（高分时约束三庭五眼偏离）
- 可直接推理：将图片放入固定目录即可输出分数（无需改文件名）
- GPU 训练流程：支持 RTX 系列显卡

### 2. 模型结构概览

输入：
- 人脸图像 `I`
- 68 点关键点 `P`

流程：
1. Branch A (Image Backbone): `I -> CNN backbone -> F_img in R^C`
2. Branch B (Geometric MLP): `P -> GeometricPriorExtractor -> V_geo -> MLP -> F_geo in R^C`
3. Fusion + Head:
   - `concat(F_img, F_geo) -> FC -> logits(K=100)`
   - `softmax(logits) -> y_hat`
   - `score_hat = sum_k y_hat[k] * k`

### 3. 三庭五眼几何先验（显式特征）

在 `utils/geometric_prior.py` 中实现：
- 三庭比例与偏离度：`S_courts`
- 五眼比例与偏离度：`S_eyes`
- 拼接得到 `V_geo`（固定维度特征向量）

### 4. 目录结构（核心）

```text
data/
  ldl_dataset.py
  manifest_dataset.py
  scut_fbp5500.py
models/
  dual_branch_fbp.py
utils/
  geometric_prior.py
  fbp_loss.py
  ldl_labels.py
  training.py
scripts/
  build_scut_fbp5500_manifest.py
  precompute_landmarks_npy.py
train_pipeline.py
infer_fbp_api.py
```

### 5. 数据准备（SCUT-FBP5500 v2）

1. 下载并解压 SCUT-FBP5500 v2  
2. 构建 manifest（`path,score`）  
3. 将原始 1~5 分映射到 1~100（脚本已支持）

示例目录：
- 图片目录：`data/scut_images/SCUT-FBP5500_v2/Images`
- 训练清单：`data/scut_train_fold1.csv`

### 6. 环境安装（推荐 GPU）

推荐新建 conda 环境，避免依赖冲突：

```bash
conda create -n fbp-gpu python=3.10 -y
conda activate fbp-gpu

# CUDA 版 PyTorch (示例 cu121)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 其他依赖
pip install numpy==1.26.4 pillow opencv-python scikit-image face-alignment
```

验证 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 7. 训练流程

#### 7.1 预计算 68 点关键点（强烈建议）

在线检测较慢，建议先离线缓存 `.npy`：

```bash
python scripts/precompute_landmarks_npy.py \
  --manifest data/scut_train_fold1.csv \
  --image-root "d:\yan\data\scut_images\SCUT-FBP5500_v2\Images" \
  --out-dir data/landmarks
```

#### 7.2 训练（自动保存 best/last）

`train_pipeline.py` 已支持：
- `val-ratio` 划分
- 每轮保存 `last.pth`
- 按 `val loss_total` 保存 `best.pth`
- 训练结束自动加载 best

示例：

```bash
python train_pipeline.py \
  --manifest data/scut_train_fold1.csv \
  --image-root "d:\yan\data\scut_images\SCUT-FBP5500_v2\Images" \
  --landmark-mode npy \
  --landmark-dir data/landmarks \
  --epochs 5 \
  --val-ratio 0.1 \
  --checkpoint-dir checkpoints_gpu \
  --num-classes 100
```

### 8. 推理（只放图片，不改名字）

已支持固定目录自动选最新图片：

1. 把图片放到 `d:\yan\input\`（文件名随意）
2. 运行：

```bash
python infer_fbp_api.py
```

或指定图片：

```bash
python infer_fbp_api.py --image "D:\your_image.png"
```

默认会优先加载：
- `checkpoints_gpu/best.pth`
- 若不存在则尝试 `checkpoints/best.pth`

### 9. 输出解释

推理输出包含：
- `Expected score_hat`：1~100 期望分
- `Model output top-k bins`：该图最可能的分数档及概率
- `Human prior (Val)`：在先验分布中的对应概率与累计概率
- `Position within prior`：相对先验中位数的高低位置

注意：模型概率是“给定该图像”的条件概率，不是训练集频率本身。

### 10. 局限性与伦理

- 颜值评估具有主观性，受数据分布、标注人群和文化背景影响显著
- 不建议将本模型用于招聘、教育、金融、司法等高风险场景
- 建议结合公平性、偏差分析与人工复核

### 11. 许可证与数据协议

- 代码许可证：请在本仓库补充（如 MIT / Apache-2.0）
- 数据集（SCUT-FBP5500）有独立使用条款，请遵守原始协议

