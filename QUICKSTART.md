# 🚀 Quick Start Guide

## Step 1: Environment Setup

```bash
# 1. Configure API key
cp .env.example .env
# Edit .env and fill in your OPENROUTER_API_KEY

# 2. Create test personas (3 personas)
python scripts/create_test_personas.py

# 3. Verify environment
python -c "import openai; print('✓ OpenAI installed')"
```

---

## Step 2: Small-scale Testing (Highly Recommended!)

### Test 2 models × 3 personas = 6 data groups

```bash
# Stage 1: Collect behavioral data (~5-10 minutes, 6 groups)
python run_stage1_behaviors.py \
    --models "openai/gpt-4o-mini,anthropic/claude-3.7-sonnet" \
    --personas data/inputs/personas_test.json

# Check progress
python scripts/check_progress.py

# Stage 2: Generate stories (~3-5 minutes, 6 stories)
python run_stage2_stories.py \
    --models "openai/gpt-4o-mini,anthropic/claude-3.7-sonnet" \
    --personas data/inputs/personas_test.json

# Check progress again
python scripts/check_progress.py
```

### Verify Output

```bash
# View generated behavior data
ls -lh data/outputs/behaviors/
cat data/outputs/behaviors/openai_gpt-4o-mini_p1.json | jq . | head -50

# View generated stories
ls -lh data/outputs/stories/
cat data/outputs/stories/openai_gpt-4o-mini_p1.txt | head -20

# Check story length (should be close to 800 words)
wc -w data/outputs/stories/*.txt
```

---

## Step 3: Check Data Quality

### Check BFI Format

```bash
# Check if BFI responses follow format (a) 1, (b) 2, ...
cat data/outputs/behaviors/openai_gpt-4o-mini_p1.json | \
    jq -r '.behaviors.bfi.response' | head -10
```

Expected output:
```
(a) 5
(b) 2
(c) 4
...
```

### Check Story Quality

```bash
# 1. Check if stories explicitly mention personality traits (should not)
grep -i "extroverted\|agreeable\|neurotic" data/outputs/stories/*.txt

# 2. Check story length
for f in data/outputs/stories/*.txt; do
    echo "$f: $(wc -w < $f) words"
done
```

Expected:
- ✅ No explicit mentions of personality trait words
- ✅ Story length between 700-900 words

---

## Step 4: Full Run (After Testing is Confirmed)

### Option A: Full Run at Once

```bash
# Stage 1: 12 models × 32 personas = 384 groups
# Estimated time: 2-4 hours
# Estimated cost: $20-30 (depending on model pricing)
python run_stage1_behaviors.py

# Stage 2: 384 stories
# Estimated time: 1-2 hours
# Estimated cost: $2-5
python run_stage2_stories.py
```

### Option B: Batch Run (Recommended, Safer)

```bash
# First batch: Small models (6 models)
python run_stage1_behaviors.py \
    --models "meta-llama/llama-3.2-3b-instruct,meta-llama/llama-3-8b-instruct,qwen/qwen-2.5-1.5b-instruct,qwen/qwen-2.5-7b-instruct,mistralai/mistral-7b-instruct,allenai/olmo-2-1124-7b-instruct"

# Check quality
python scripts/check_progress.py

# If no issues, continue with second batch: Large models (6 models)
python run_stage1_behaviors.py \
    --models "meta-llama/llama-3.3-70b-instruct,meta-llama/llama-3.1-405b-instruct,qwen/qwen-2.5-72b-instruct,qwen/qwq-32b-preview,anthropic/claude-3.7-sonnet,openai/gpt-4o"

# After all complete, generate stories
python run_stage2_stories.py
```

---

## Step 5: Check Completion Status

```bash
# Comprehensive check
python scripts/check_progress.py

# Expected output:
# Stage 1: 384/384 (100%)
# Stage 2: 384/384 (100%)
```

---

## Troubleshooting

### Q: Some model API calls failed

```bash
# View error logs
cat data/outputs/logs/*.log

# Re-run only the failed model
python run_stage1_behaviors.py --models "failed_model_name"
```

### Q: Want to pause/resume

No problem! The script will automatically skip completed data:

```bash
# You can interrupt at any time (Ctrl+C)
# Re-running will continue from where it left off
python run_stage1_behaviors.py
```

### Q: Modified story prompt and want to regenerate

```bash
# 1. Modify data/inputs/writing_prompt.txt
# 2. Delete old stories
rm -rf data/outputs/stories/*
# 3. Regenerate
python run_stage2_stories.py
```

---

## Next Steps: Data Analysis

After data collection:

1. **Stage 3**: Extract linguistic features using LIWC-22
   ```bash
   # You'll need the LIWC-22 tool
   liwc-22-cli --input data/outputs/stories/ --output data/outputs/liwc/
   ```

2. **Stage 4**: Statistical analysis
   - Merge behaviors + liwc data
   - Correlation analysis
   - Regression models

3. **Write paper** 📝

---

## Cost Estimation

### Small-scale Testing (2 models × 3 personas)
- API calls: ~400 calls
- Cost: ~$0.5-1 (using mini models)
- Time: ~10-15 minutes

### Full Run (12 models × 32 personas)
- Stage 1 API calls: ~23,000-31,000 calls
- Stage 2 API calls: ~384 calls
- Total cost: ~$20-30 (depending on model pricing)
- Total time: ~3-6 hours

**Recommendations**:
- Test the workflow with small-scale testing first
- Use OpenRouter's free tier (if available)
- Run in batches to control costs

---

## Project Structure

```
Final Project/
├── run_stage1_behaviors.py    ← Stage 1 main script
├── run_stage2_stories.py      ← Stage 2 main script
├── scripts/
│   ├── check_progress.py      ← Check progress
│   └── create_test_personas.py ← Create test data
├── src/                        ← Core logic
│   ├── api.py
│   ├── behavior_gen.py
│   └── story_gen.py
├── data/
│   ├── inputs/
│   │   ├── personas.json      ← 32 personas
│   │   ├── personas_test.json ← Test data (3 personas)
│   │   ├── writing_prompt.txt
│   │   ├── bfi_prompt.txt
│   │   └── ...
│   └── outputs/
│       ├── behaviors/         ← Stage 1 output
│       ├── stories/           ← Stage 2 output
│       └── logs/              ← Error logs
├── docs/
│   ├── method_design.md       ← Detailed method documentation
│   └── proposal.md            ← Research proposal
├── README_PIPELINE.md         ← Complete pipeline documentation
└── QUICKSTART.md             ← This document
```

---

## Getting Help

- Detailed documentation: `README_PIPELINE.md`
- Method design: `docs/method_design.md`
- Check progress: `python scripts/check_progress.py`

Good luck with your experiment! 🎉
