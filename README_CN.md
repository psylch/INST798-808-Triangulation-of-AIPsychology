# LLM Personality Triangulation Pipeline

## 📋 项目概述

本项目实现了一个分阶段的LLM人格三角验证pipeline，结合了：
- **Jiang et al. (2024)** 的故事生成 + LIWC分析方法
- **Han et al. (2025)** 的BFI自陈 + 行为任务方法

核心创新：使用Han的严谨BFI数据作为Jiang的多轮对话预热，确保数据一致性。

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置API密钥
cp .env.example .env
# 编辑 .env，填入 OPENROUTER_API_KEY
```

### 2. 小规模测试（推荐先做）

```bash
# 只测试2个模型 × 3个personas
python run_stage1_behaviors.py \
    --models "openai/gpt-4o-mini,anthropic/claude-3.7-sonnet" \
    --personas data/inputs/personas_test.json

python run_stage2_stories.py \
    --models "openai/gpt-4o-mini,anthropic/claude-3.7-sonnet" \
    --personas data/inputs/personas_test.json
```

### 3. 全量运行

```bash
# Stage 1: 收集所有行为数据（约需要数小时）
python run_stage1_behaviors.py

# Stage 2: 生成故事（约需要1-2小时）
python run_stage2_stories.py

# Stage 3: LIWC分析（你后面用LIWC-22工具做）
# Stage 4: 统计分析（你后面用R/Python做）
```

---

## 📊 Pipeline阶段说明

### Stage 1: 行为数据收集 🔴

**脚本**: `run_stage1_behaviors.py`

**收集内容**:
- ✅ BFI-44（Big Five Inventory，44个问题）
- ✅ Risk-Taking（Columbia Card Task，3个场景）
- ✅ Social Bias（IAT，多个刺激对）
- ✅ Honesty（置信度校准，3步测试）
- ✅ Sycophancy（道德困境，2步测试）

**输出**: `data/outputs/behaviors/{model}_{persona_id}.json`

**示例输出**:
```json
{
  "meta": {
    "model": "anthropic/claude-3.7-sonnet",
    "persona_id": "p1",
    "traits": "O+C+E+A+N+"
  },
  "behaviors": {
    "bfi": {"prompt": "...", "response": "(a) 5\n(b) 2\n..."},
    "risk": [...],
    "iat": [...],
    "honesty": [...],
    "sycophancy": [...]
  },
  "errors": {}
}
```

**成本估算**:
- API调用：~60-80次 / (model, persona)
- 总计：~23,000-31,000次（12 models × 32 personas）
- 时间：根据API速率，约2-4小时

**命令行参数**:
```bash
python run_stage1_behaviors.py \
    --models "model1,model2,..."           # 指定模型
    --personas data/inputs/personas.json   # 指定personas文件
    --output-dir data/outputs/behaviors    # 输出目录
    --per-model-concurrency 5              # 每个模型的并发数
    --max-tasks 12                         # 全局最大并发数
```

---

### Stage 2: 故事生成（使用BFI预热）🟡

**脚本**: `run_stage2_stories.py`

**方法**:
1. 读取Stage 1的BFI结果
2. 构造多轮对话：
   ```
   [system] persona system prompt
   [user] BFI 44题
   [assistant] (a) 5, (b) 2, ... ← Stage 1的实际BFI回答
   [user] Please share a personal story in 800 words...
   ```
3. 生成故事

**输出**: `data/outputs/stories/{model}_{persona_id}.txt`

**成本估算**:
- API调用：1次 / (model, persona)
- 总计：384次（12 models × 32 personas）
- 时间：约1-2小时

**命令行参数**:
```bash
python run_stage2_stories.py \
    --models "model1,model2,..."                 # 指定模型
    --personas data/inputs/personas.json         # 指定personas文件
    --behaviors-dir data/outputs/behaviors       # Stage 1输出目录
    --output-dir data/outputs/stories            # 故事输出目录
    --per-model-concurrency 3                    # 并发数（故事较慢）
```

---

### Stage 3: LIWC分析 🟢（你后面做）

使用LIWC-22工具处理stories：

```bash
# 伪代码
liwc-22-cli \
    --input data/outputs/stories/*.txt \
    --output data/outputs/liwc/features.csv
```

输出LIWC特征：
- Affective processes (positive/negative emotion)
- Social processes
- Cognitive processes
- 等等...

---

### Stage 4: 统计分析 🟢（你后面做）

合并数据并分析：

```python
# 伪代码
import pandas as pd

# 加载数据
behaviors = load_behaviors("data/outputs/behaviors/")
liwc = pd.read_csv("data/outputs/liwc/features.csv")

# 合并
df = merge(behaviors, liwc)

# 分析
# RQ1: 自陈traits与语言特征的相关性
# RQ2: 自陈traits与行为的相关性
# RQ3: 语言特征与行为的相关性（三角验证）
```

---

## 🎯 使用场景

### 场景1：测试单个模型

```bash
# 只测试Claude 3.7
python run_stage1_behaviors.py \
    --models "anthropic/claude-3.7-sonnet"

python run_stage2_stories.py \
    --models "anthropic/claude-3.7-sonnet"
```

### 场景2：测试部分personas

创建 `data/inputs/personas_test.json`，只包含3-5个personas：

```json
[
  {"id": "p1", "traits": "O+C+E+A+N+", ...},
  {"id": "p2", "traits": "O+C+E+A+N-", ...},
  {"id": "p17", "traits": "O-C+E+A+N+", ...}
]
```

然后运行：
```bash
python run_stage1_behaviors.py --personas data/inputs/personas_test.json
python run_stage2_stories.py --personas data/inputs/personas_test.json
```

### 场景3：只重跑故事生成

如果Stage 1已完成，只想重新生成故事：

```bash
# 删除旧故事
rm -rf data/outputs/stories/*

# 重新生成（会自动读取Stage 1的BFI结果）
python run_stage2_stories.py
```

### 场景4：分批次运行（控制成本）

```bash
# 第一批：6个小模型
python run_stage1_behaviors.py \
    --models "meta-llama/llama-3.2-3b-instruct,meta-llama/llama-3-8b-instruct,qwen/qwen-2.5-1.5b-instruct,qwen/qwen-2.5-7b-instruct,mistralai/mistral-7b-instruct,allenai/olmo-2-1124-7b-instruct"

# 检查数据质量...

# 第二批：6个大模型
python run_stage1_behaviors.py \
    --models "meta-llama/llama-3.3-70b-instruct,meta-llama/llama-3.1-405b-instruct,qwen/qwen-2.5-72b-instruct,qwen/qwq-32b-preview,anthropic/claude-3.7-sonnet,openai/gpt-4o"
```

---

## 📂 输出目录结构

```
data/outputs/
├── behaviors/              # Stage 1输出
│   ├── anthropic_claude-3.7-sonnet_p1.json
│   ├── anthropic_claude-3.7-sonnet_p2.json
│   └── ...
├── stories/                # Stage 2输出
│   ├── anthropic_claude-3.7-sonnet_p1.txt
│   ├── anthropic_claude-3.7-sonnet_p2.txt
│   └── ...
├── liwc/                   # Stage 3输出（你后面做）
│   └── features.csv
├── analysis/               # Stage 4输出（你后面做）
│   ├── correlations.csv
│   └── regressions.csv
└── logs/                   # 错误日志
    └── errors_*.log
```

---

## 🔍 质量检查

### Stage 1完成后检查

```bash
# 统计收集了多少数据
python scripts/check_stage1.py

# 检查BFI格式是否正确
python scripts/validate_bfi.py

# 查看某个具体的behavior文件
cat data/outputs/behaviors/anthropic_claude-3.7-sonnet_p1.json | jq .
```

### Stage 2完成后检查

```bash
# 统计生成了多少故事
ls data/outputs/stories/*.txt | wc -l

# 检查故事长度分布
python scripts/check_story_lengths.py

# 检查是否明确提到人格特征（应该没有）
python scripts/check_trait_mentions.py
```

---

## ⚠️ 常见问题

### Q1: Stage 1中途中断了怎么办？

**答**: 没关系，重新运行会自动跳过已完成的：

```bash
# 继续运行，只会处理未完成的
python run_stage1_behaviors.py
```

### Q2: 某个模型的API失败了怎么办？

**答**: 检查 `data/outputs/logs/` 中的错误日志，修复问题后重跑：

```bash
# 只重跑失败的模型
python run_stage1_behaviors.py --models "失败的模型名"
```

### Q3: 想要更改story prompt怎么办？

**答**: 修改 `data/inputs/writing_prompt.txt`，然后：

```bash
# 删除旧故事
rm -rf data/outputs/stories/*

# 重新生成
python run_stage2_stories.py
```

### Q4: 如何估算总成本？

**答**:

```
Stage 1: 60-80次 × 384组 = 23k-31k次 API调用
Stage 2: 1次 × 384组 = 384次 API调用
总计: ~23k-32k次

假设平均每次调用 $0.001（取决于模型）:
总成本: $23-32

实际成本取决于：
- 使用的具体模型（GPT-4o贵，小模型便宜）
- OpenRouter的定价
- 是否有free tier
```

---

## 📚 相关文档

- `docs/method_design.md` - 详细的方法设计说明
- `docs/proposal.md` - 研究提案
- `.env.example` - 环境变量配置示例

---

## 🤝 贡献

如果发现问题或有改进建议，请创建issue或PR。

---

## 📄 许可

本项目代码遵循MIT许可。
