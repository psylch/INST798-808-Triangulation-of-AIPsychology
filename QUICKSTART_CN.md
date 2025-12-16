# 🚀 快速开始指南

## 第一步：环境配置

```bash
# 1. 配置API密钥
cp .env.example .env
# 编辑 .env，填入你的 OPENROUTER_API_KEY

# 2. 创建测试用的小规模personas（3个）
python scripts/create_test_personas.py

# 3. 验证环境
python -c "import openai; print('✓ OpenAI installed')"
```

---

## 第二步：小规模测试（强烈推荐！）

### 测试2个模型 × 3个personas = 6组数据

```bash
# Stage 1: 收集行为数据（约5-10分钟，6组数据）
python run_stage1_behaviors.py \
    --models "openai/gpt-4o-mini,anthropic/claude-3.7-sonnet" \
    --personas data/inputs/personas_test.json

# 检查进度
python scripts/check_progress.py

# Stage 2: 生成故事（约3-5分钟，6个故事）
python run_stage2_stories.py \
    --models "openai/gpt-4o-mini,anthropic/claude-3.7-sonnet" \
    --personas data/inputs/personas_test.json

# 再次检查进度
python scripts/check_progress.py
```

### 验证输出

```bash
# 查看生成的behavior数据
ls -lh data/outputs/behaviors/
cat data/outputs/behaviors/openai_gpt-4o-mini_p1.json | jq . | head -50

# 查看生成的故事
ls -lh data/outputs/stories/
cat data/outputs/stories/openai_gpt-4o-mini_p1.txt | head -20

# 检查故事长度（应该接近800词）
wc -w data/outputs/stories/*.txt
```

---

## 第三步：检查数据质量

### 检查BFI格式

```bash
# 检查BFI回答是否符合格式 (a) 1, (b) 2, ...
cat data/outputs/behaviors/openai_gpt-4o-mini_p1.json | \
    jq -r '.behaviors.bfi.response' | head -10
```

期望输出：
```
(a) 5
(b) 2
(c) 4
...
```

### 检查故事质量

```bash
# 1. 检查故事是否明确提到人格特征（应该没有）
grep -i "extroverted\|agreeable\|neurotic" data/outputs/stories/*.txt

# 2. 检查故事长度
for f in data/outputs/stories/*.txt; do
    echo "$f: $(wc -w < $f) words"
done
```

期望：
- ✅ 没有明确提到人格特征词
- ✅ 故事长度在700-900词之间

---

## 第四步：全量运行（确认测试无误后）

### 方案A：一次性全量运行

```bash
# Stage 1: 12 models × 32 personas = 384组
# 预计时间：2-4小时
# 预计成本：$20-30（取决于模型定价）
python run_stage1_behaviors.py

# Stage 2: 384个故事
# 预计时间：1-2小时
# 预计成本：$2-5
python run_stage2_stories.py
```

### 方案B：分批运行（推荐，更安全）

```bash
# 第一批：小模型（6个）
python run_stage1_behaviors.py \
    --models "meta-llama/llama-3.2-3b-instruct,meta-llama/llama-3-8b-instruct,qwen/qwen-2.5-1.5b-instruct,qwen/qwen-2.5-7b-instruct,mistralai/mistral-7b-instruct,allenai/olmo-2-1124-7b-instruct"

# 检查质量
python scripts/check_progress.py

# 如果没问题，继续第二批：大模型（6个）
python run_stage1_behaviors.py \
    --models "meta-llama/llama-3.3-70b-instruct,meta-llama/llama-3.1-405b-instruct,qwen/qwen-2.5-72b-instruct,qwen/qwq-32b-preview,anthropic/claude-3.7-sonnet,openai/gpt-4o"

# 全部完成后，生成故事
python run_stage2_stories.py
```

---

## 第五步：检查完成情况

```bash
# 全面检查
python scripts/check_progress.py

# 期望输出：
# Stage 1: 384/384 (100%)
# Stage 2: 384/384 (100%)
```

---

## 常见问题解决

### Q: 某些模型API调用失败

```bash
# 查看错误日志
cat data/outputs/logs/*.log

# 只重跑失败的模型
python run_stage1_behaviors.py --models "失败的模型名"
```

### Q: 想要暂停/继续

没问题！脚本会自动跳过已完成的数据：

```bash
# 随时可以中断（Ctrl+C）
# 重新运行会继续未完成的部分
python run_stage1_behaviors.py
```

### Q: 修改了story prompt，想重新生成

```bash
# 1. 修改 data/inputs/writing_prompt.txt
# 2. 删除旧故事
rm -rf data/outputs/stories/*
# 3. 重新生成
python run_stage2_stories.py
```

---

## 下一步：数据分析

收集完数据后：

1. **Stage 3**: 使用LIWC-22提取语言特征
   ```bash
   # 你需要LIWC-22工具
   liwc-22-cli --input data/outputs/stories/ --output data/outputs/liwc/
   ```

2. **Stage 4**: 统计分析
   - 合并behaviors + liwc数据
   - 相关性分析
   - 回归模型

3. **写论文** 📝

---

## 成本估算

### 小规模测试（2 models × 3 personas）
- API调用：~400次
- 成本：~$0.5-1（使用mini模型）
- 时间：~10-15分钟

### 全量运行（12 models × 32 personas）
- Stage 1 API调用：~23,000-31,000次
- Stage 2 API调用：~384次
- 总成本：~$20-30（取决于模型定价）
- 总时间：~3-6小时

**建议**：
- 先用小规模测试验证流程
- 使用OpenRouter的free tier（如果有）
- 分批运行，控制成本

---

## 项目结构

```
Final Project/
├── run_stage1_behaviors.py    ← Stage 1主脚本
├── run_stage2_stories.py      ← Stage 2主脚本
├── scripts/
│   ├── check_progress.py      ← 检查进度
│   └── create_test_personas.py ← 创建测试数据
├── src/                        ← 核心逻辑
│   ├── api.py
│   ├── behavior_gen.py
│   └── story_gen.py
├── data/
│   ├── inputs/
│   │   ├── personas.json      ← 32个人格
│   │   ├── personas_test.json ← 测试用（3个）
│   │   ├── writing_prompt.txt
│   │   ├── bfi_prompt.txt
│   │   └── ...
│   └── outputs/
│       ├── behaviors/         ← Stage 1输出
│       ├── stories/           ← Stage 2输出
│       └── logs/              ← 错误日志
├── docs/
│   ├── method_design.md       ← 详细方法说明
│   └── proposal.md            ← 研究提案
├── README_PIPELINE.md         ← 完整pipeline文档
└── QUICKSTART.md             ← 本文档
```

---

## 获取帮助

- 详细文档: `README_PIPELINE.md`
- 方法设计: `docs/method_design.md`
- 检查进度: `python scripts/check_progress.py`

祝实验顺利！🎉
