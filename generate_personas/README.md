# Persona Generation for Market Research

This repository generates large batches of product-specific consumer personas (JSON) from (1) posterior segment weights and (2) a compact market context, using LLM prompts designed for two modes: Multiple-Choice (MC) and Swap (SWAP).
The pipeline favors reproducible procedure/format/order: we seed randomness once per session, save prompts and batch logs, and keep a stable output schema.

## Project Overview

This project generates large batches of **product-specific consumer personas** by combining:

1) **Posterior segment weights** (`weights/{keyword}_posterior.json`) used to *sample* segment combinations (gender, age band, household size, income).  
2) **Compact market context** (`data/contexts/{keyword}.json`) injected into prompts to ground personas in realistic market signals.

Two prompt modes are supported:

- **Multiple Choice (MC)** — baseline demographic personas following a compact schema.  
- **Swap (SWAP)** — richer personas that may include prior brand/product usage and purchase criteria.


## Batch prompting with pre/post-processing

- **Sampling (reproducible):** segment rows are sampled from posterior ratios.  
- **Batch prompting:** rows are grouped into batches (e.g., 10) and sent in a single LLM call per batch to reduce cost/latency and keep context stable.  
- **Per-batch logging:** every batch saves a raw JSON file under `output/{keyword}/personas_log/`; optional prompt snapshots go to `debug_prompts/`.  
- **UUID handling & recovery:** if a batch returns fewer items or missing `uuid`s than expected, the missing entries are collected and **re-generated** into `personas_{keyword}_part_101.json`, `personas_{keyword}_part_102.json`, .. and then merged into the final output.  
- **Final merge:** all valid batch results (+ recovered items) are consolidated into `output/{keyword}/personas_{keyword}_all.json`.


## Project Structure

```
persona_process/
├─ data/
│  └─ contexts/                  # Market context per keyword (e.g., 그릭요거트.json)
├─ output/
│  ├─ personas_{keyword}_all.json   # Final merged personas
│  └─ logs/
│      └─{keyword}/
│           ├─ personas_log/           # Raw per-batch responses (JSON)
│           ├─ debug_prompts/          # (opt) saved system/user prompts per batch
├─ prompts/
│  ├─ prompt_mc.py               # MC prompt & schema
│  └─ prompt_sw.py               # SWAP prompt & schema
├─ weights/
│  └─ {keyword}_posterior.json   # Posterior probabilities for sampling
├─ generate_personas.py          # Main CLI
├─ requirements.txt
└─ .env                          # OpenAI API key (local only)
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Multiple Choice(MC) Mode

The **MC mode** generates personas based primarily on **overall market context** (demographics + market report) for a given product group.  
It is suitable for products where **broad consumer traits** are more relevant than individual prior brand usage.  

**Applicable product groups:** `참치액`, `참치캔`

```powershell
python .\generate_personas.py `
  --mode mc `
  --keywords 참치액 `
  --n_samples 1000 `
  --batch_size 10 `
  --model gpt-4o-mini `
  --temperature 0.2 `
  --log_level DEBUG `
  --log_prompts `
  --prompt_preview_chars 600
```

```powershell
python .\generate_personas.py `
  --mode mc `
  --keywords 참치캔 `
  --n_samples 1000 `
  --batch_size 10 `
  --model gpt-4o-mini `
  --temperature 0.2 `
  --log_level DEBUG `
  --log_prompts `
  --prompt_preview_chars 600
```

### SWAP(SW) Mode

The **SWAP mode** generates personas by combining **overall market context** with **prior product/brand usage patterns**.
This mode enriches personas with substitution/switching behavior and is more suitable for categories with **brand competition**.

**Applicable product groups:** `그릭요거트`, `편의점커피라떼`, `스팸`

```powershell
python .\generate_personas.py `
  --mode swap `
  --keywords 그릭요거트 `
  --n_samples 1000 `
  --batch_size 10 `
  --model gpt-4o-mini `
  --temperature 0.2 `
  --log_level DEBUG `
  --log_prompts `
  --prompt_preview_chars 600
```

```powershell
python .\generate_personas.py `
  --mode swap `
  --keywords 편의점커피라떼 `
  --n_samples 1000 `
  --batch_size 10 `
  --model gpt-4o-mini `
  --temperature 0.2 `
  --log_level DEBUG `
  --log_prompts `
  --prompt_preview_chars 600
```

```powershell
python .\generate_personas.py `
  --mode swap `
  --keywords 스팸 `
  --n_samples 1000 `
  --batch_size 10 `
  --model gpt-4o-mini `
  --temperature 0.2 `
  --log_level DEBUG `
  --log_prompts `
  --prompt_preview_chars 600
```