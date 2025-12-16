# LLM Personality Triangulation Pipeline

## 📋 Project Overview

This project implements a staged LLM personality triangulation pipeline that combines:
- **Jiang et al. (2024)** story generation + LIWC analysis method
- **Han et al. (2025)** BFI self-report + behavioral task method

Core innovation: Using Han's rigorous BFI data as a warm-up for Jiang's multi-turn dialogue to ensure data consistency.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and fill in OPENROUTER_API_KEY
```

### 2. Small-scale Testing (Recommended First)

```bash
# Test only 2 models × 3 personas
python scripts/run_stage1_behaviors.py \
    --models "openai/gpt-4o-mini,anthropic/claude-3.7-sonnet" \
    --personas data/inputs/personas_test.json

python scripts/run_stage2_stories.py \
    --models "openai/gpt-4o-mini,anthropic/claude-3.7-sonnet" \
    --personas data/inputs/personas_test.json
```

### 3. Full Run

```bash
# Stage 1: Collect all behavioral data (takes several hours)
python scripts/run_stage1_behaviors.py

# Stage 2: Generate stories (takes 1-2 hours)
python scripts/run_stage2_stories.py

# Stage 3: LIWC analysis (you'll do this later with LIWC-22 tool)
# Stage 4: Statistical analysis (you'll do this later with R/Python)
```

---

## 📊 Pipeline Stage Details

### Stage 1: Behavioral Data Collection 🔴

**Script**: `scripts/run_stage1_behaviors.py`

**Data Collected**:
- ✅ BFI-44 (Big Five Inventory, 44 questions)
- ✅ Risk-Taking (Columbia Card Task, 3 scenarios)
- ✅ Social Bias (IAT, multiple stimulus pairs)
- ✅ Honesty (confidence calibration, 3-step test)
- ✅ Sycophancy (moral dilemma, 2-step test)

**Output**: `data/outputs/behaviors/{model}_{persona_id}.json`

**Example Output**:
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

**Cost Estimate**:
- API calls: ~60-80 calls / (model, persona)
- Total: ~23,000-31,000 calls (12 models × 32 personas)
- Time: Depending on API rate, approximately 2-4 hours

**Command Line Arguments**:
```bash
python scripts/run_stage1_behaviors.py \
    --models "model1,model2,..."           # Specify models
    --personas data/inputs/personas.json   # Specify personas file
    --output-dir data/outputs/behaviors    # Output directory
    --per-model-concurrency 5              # Concurrency per model
    --max-tasks 12                         # Global max concurrency
```

---

### Stage 2: Story Generation (Using BFI Warm-up) 🟡

**Script**: `scripts/run_stage2_stories.py`

**Method**:
1. Read BFI results from Stage 1
2. Construct multi-turn dialogue:
   ```
   [system] persona system prompt
   [user] BFI 44 questions
   [assistant] (a) 5, (b) 2, ... ← Actual BFI responses from Stage 1
   [user] Please share a personal story in 800 words...
   ```
3. Generate story

**Output**: `data/outputs/stories/{model}_{persona_id}.txt`

**Cost Estimate**:
- API calls: 1 call / (model, persona)
- Total: 384 calls (12 models × 32 personas)
- Time: Approximately 1-2 hours

**Command Line Arguments**:
```bash
python scripts/run_stage2_stories.py \
    --models "model1,model2,..."                 # Specify models
    --personas data/inputs/personas.json         # Specify personas file
    --behaviors-dir data/outputs/behaviors       # Stage 1 output directory
    --output-dir data/outputs/stories            # Story output directory
    --per-model-concurrency 3                    # Concurrency (stories are slower)
```

---

### Stage 3: LIWC Analysis 🟢 (You'll Do This Later)

Use LIWC-22 tool to process stories:

```bash
# Pseudocode
liwc-22-cli \
    --input data/outputs/stories/*.txt \
    --output data/outputs/liwc/features.csv
```

Output LIWC features:
- Affective processes (positive/negative emotion)
- Social processes
- Cognitive processes
- etc...

---

### Stage 4: Statistical Analysis 🟢 (You'll Do This Later)

Merge data and analyze:

```python
# Pseudocode
import pandas as pd

# Load data
behaviors = load_behaviors("data/outputs/behaviors/")
liwc = pd.read_csv("data/outputs/liwc/features.csv")

# Merge
df = merge(behaviors, liwc)

# Analyze
# RQ1: Correlation between self-reported traits and linguistic features
# RQ2: Correlation between self-reported traits and behaviors
# RQ3: Correlation between linguistic features and behaviors (triangulation)
```

---

## 🎯 Use Cases

### Use Case 1: Test Single Model

```bash
# Test only Claude 3.7
python scripts/run_stage1_behaviors.py \
    --models "anthropic/claude-3.7-sonnet"

python scripts/run_stage2_stories.py \
    --models "anthropic/claude-3.7-sonnet"
```

### Use Case 2: Test Subset of Personas

Create `data/inputs/personas_test.json` with only 3-5 personas:

```json
[
  {"id": "p1", "traits": "O+C+E+A+N+", ...},
  {"id": "p2", "traits": "O+C+E+A+N-", ...},
  {"id": "p17", "traits": "O-C+E+A+N+", ...}
]
```

Then run:
```bash
python scripts/run_stage1_behaviors.py --personas data/inputs/personas_test.json
python scripts/run_stage2_stories.py --personas data/inputs/personas_test.json
```

### Use Case 3: Re-run Story Generation Only

If Stage 1 is complete and you only want to regenerate stories:

```bash
# Delete old stories
rm -rf data/outputs/stories/*

# Regenerate (will automatically read BFI results from Stage 1)
python scripts/run_stage2_stories.py
```

### Use Case 4: Run in Batches (Cost Control)

```bash
# First batch: 6 small models
python scripts/run_stage1_behaviors.py \
    --models "meta-llama/llama-3.2-3b-instruct,meta-llama/llama-3-8b-instruct,qwen/qwen-2.5-1.5b-instruct,qwen/qwen-2.5-7b-instruct,mistralai/mistral-7b-instruct,allenai/olmo-2-1124-7b-instruct"

# Check data quality...

# Second batch: 6 large models
python scripts/run_stage1_behaviors.py \
    --models "meta-llama/llama-3.3-70b-instruct,meta-llama/llama-3.1-405b-instruct,qwen/qwen-2.5-72b-instruct,qwen/qwq-32b-preview,anthropic/claude-3.7-sonnet,openai/gpt-4o"
```

---

## 📂 Output Directory Structure

```
data/outputs/
├── behaviors/              # Stage 1 output
│   ├── anthropic_claude-3.7-sonnet_p1.json
│   ├── anthropic_claude-3.7-sonnet_p2.json
│   └── ...
├── stories/                # Stage 2 output
│   ├── anthropic_claude-3.7-sonnet_p1.txt
│   ├── anthropic_claude-3.7-sonnet_p2.txt
│   └── ...
├── liwc/                   # Stage 3 output (you'll do this later)
│   └── features.csv
├── analysis/               # Stage 4 output (you'll do this later)
│   ├── correlations.csv
│   └── regressions.csv
└── logs/                   # Error logs
    └── errors_*.log
```

---

## 🔍 Quality Checks

### After Stage 1 Completion

```bash
# Count how much data was collected
python scripts/check_stage1.py

# Check if BFI format is correct
python scripts/validate_bfi.py

# View a specific behavior file
cat data/outputs/behaviors/anthropic_claude-3.7-sonnet_p1.json | jq .
```

### After Stage 2 Completion

```bash
# Count how many stories were generated
ls data/outputs/stories/*.txt | wc -l

# Check story length distribution
python scripts/check_story_lengths.py

# Check if personality traits are explicitly mentioned (should not be)
python scripts/check_trait_mentions.py
```

---

## ⚠️ FAQ

### Q1: What if Stage 1 is interrupted?

**Answer**: No problem, re-running will automatically skip completed tasks:

```bash
# Continue running, will only process incomplete tasks
python scripts/run_stage1_behaviors.py
```

### Q2: What if a model's API fails?

**Answer**: Check error logs in `data/outputs/logs/`, fix the issue, and re-run:

```bash
# Re-run only the failed model
python scripts/run_stage1_behaviors.py --models "failed_model_name"
```

### Q3: How to change the story prompt?

**Answer**: Modify `data/inputs/writing_prompt.txt`, then:

```bash
# Delete old stories
rm -rf data/outputs/stories/*

# Regenerate
python scripts/run_stage2_stories.py
```

### Q4: How to estimate total cost?

**Answer**:

```
Stage 1: 60-80 calls × 384 groups = 23k-31k API calls
Stage 2: 1 call × 384 groups = 384 API calls
Total: ~23k-32k calls

Assuming average cost per call is $0.001 (depends on model):
Total cost: $23-32

Actual cost depends on:
- Specific models used (GPT-4o is expensive, small models are cheap)
- OpenRouter pricing
- Whether there's a free tier
```

---

## 📚 Related Documentation

- `docs/method_design.md` - Detailed method design documentation
- `docs/proposal.md` - Research proposal
- `.env.example` - Environment variable configuration example

---

## 🤝 Contributing

If you find issues or have suggestions for improvement, please create an issue or PR.

---

## 📄 License

This project code is licensed under the MIT License.
