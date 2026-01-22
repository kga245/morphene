# Morphene Production Pipeline Guide

This guide covers the complete workflow for creating Morphene neural audio instruments.

## Quick Start

### 1. Set Up Project Structure

```bash
# Set up POC (Pyroviolin) first
python -m morphene setup --poc /path/to/project

# Or set up everything
python -m morphene setup --all /path/to/project
```

### 2. Track Source Collection

```bash
# Add musical source for Pyroviolin
python -m morphene sources add \
    --instrument 10 \
    --type musical \
    --name "Violin samples from XYZ library" \
    --origin licensed \
    --license "Royalty-free" \
    --duration 5400  # 1.5 hours in seconds

# Add non-musical source
python -m morphene sources add \
    --instrument 10 \
    --type nonmusical \
    --name "Fire recordings from field session" \
    --origin recorded \
    --duration 5400

# Check status
python -m morphene sources status 10
```

### 3. Preprocess Audio

Use RAVE's preprocessing script:

```bash
python -m rave preprocess \
    --input_path /sources/pyroviolin/musical/processed \
    --input_path /sources/pyroviolin/nonmusical/processed \
    --output_path /training_data/pyroviolin \
    --sampling_rate 48000 \
    --channels 1
```

### 4. Train Model

```bash
# Option A: Direct training
python -m rave train \
    --name pyroviolin \
    --config v2 \
    --db_path /training_data/pyroviolin \
    --out_path /models/pyroviolin \
    --save_every 50000 \
    --gpu 0

# Option B: Use queue for parallel training
python -m morphene queue add \
    --instrument 10 \
    --db-path /training_data/pyroviolin \
    --output /models/pyroviolin

python -m morphene queue start
```

### 5. Audit Checkpoints

```bash
python -m morphene audit \
    --run /models/pyroviolin \
    --all \
    --instrument pyroviolin \
    --musical-ref /sources/pyroviolin/reference_clips/violin_ref.wav \
    --nonmusical-ref /sources/pyroviolin/reference_clips/fire_ref.wav \
    --output /models \
    --checkpoint-filter 50000
```

### 6. Export Final Model

```bash
python -m rave export \
    --run /models/pyroviolin \
    --streaming \
    --fidelity 0.95 \
    --output /models/pyroviolin/final \
    --name pyroviolin
```

## Detailed Workflow

### Source Collection

Each instrument requires:

| Source Type | Minimum Duration | Format |
|-------------|------------------|--------|
| Musical | 1.5 hours | Mono 48kHz WAV |
| Non-musical | 1.5 hours | Mono 48kHz WAV |

#### Acquisition Methods

| Method | Use For | Notes |
|--------|---------|-------|
| Original recording | Field sounds, synths | Full ownership |
| Licensed libraries | Orchestral instruments | Check license |
| Commissioned performance | Rare acoustic | Document agreement |
| Synthesis | Analog synths | Control patches |

#### Processing Steps

1. Convert to mono 48kHz WAV
2. Normalize loudness (-14 LUFS recommended)
3. Trim silence
4. Remove artifacts/noise if needed
5. Split into reasonable file sizes

### Reference Clips

Select two 30-second clips for auditing:

**Musical reference:**
- Representative of source character
- Includes variety of pitches/dynamics
- Clean, well-recorded

**Non-musical reference:**
- Captures essence of source texture
- Avoids extreme transients
- Consistent quality

### Training Configuration

Recommended settings for Morphene:

```bash
python -m rave train \
    --name {instrument_name} \
    --config v2 \
    --db_path {dataset_path} \
    --out_path {output_path} \
    --max_steps 3000000 \
    --val_every 10000 \
    --save_every 50000 \
    --batch 8 \
    --gpu 0
```

#### Config Options

| Config | Use Case | Memory |
|--------|----------|--------|
| v2 | Default, best quality | 16GB |
| v2_small | Faster training, less capacity | 8GB |
| v3 | Higher quality, longer training | 32GB |

### Parallel Training Strategy

Based on POC memory benchmarks:

| Peak Memory | Strategy |
|-------------|----------|
| < 30GB | 3 parallel jobs |
| 30-40GB | 2 parallel jobs |
| > 40GB | Sequential |

### Audit Evaluation

Listen for each checkpoint:

| Output | Check For |
|--------|-----------|
| musical_recon.wav | Faithful reconstruction, recognizable |
| nonmusical_recon.wav | Texture preserved, character intact |
| morph_00.wav | Pure musical quality |
| morph_25.wav | Subtle non-musical influence |
| morph_50.wav | Balanced blend |
| morph_75.wav | Strong non-musical character |
| morph_100.wav | Pure non-musical quality |

**Pass criteria:**
- Smooth transitions between morph points
- No catastrophic artifacts
- Musically usable across range
- Recognizable source characters

### Model Export

Export options for nn~:

```bash
# Streaming mode (real-time, lower latency)
python -m rave export --run /path --streaming --fidelity 0.95

# Non-streaming (higher quality, higher latency)
python -m rave export --run /path --fidelity 0.95
```

### Delivery Package

Each instrument ships with:

```
{instrument_name}/
├── {instrument_name}.ts      # TorchScript model
├── metadata.json             # Full metadata
└── demo.wav                  # 60-second showcase
```

## Memory Monitoring

Track GPU memory during training:

```python
from morphene.training import TrainingQueue

queue = TrainingQueue(db_path='/data/queue.db')
job = queue.get_job(job_id)
stats = queue.get_memory_stats(job_id)

print(f"Peak memory: {stats['max_gb']:.1f} GB")
print(f"Average: {stats['avg_gb']:.1f} GB")
```

## Troubleshooting

### Training Issues

**Out of memory:**
- Reduce batch size: `--batch 4`
- Use v2_small config
- Reduce parallel jobs

**Poor reconstruction:**
- Check source quality
- Verify preprocessing
- Extend training

**Unstable morphing:**
- Balance source durations
- Check latent dimensions
- Try different checkpoint

### Source Issues

**Duration too short:**
- Combine multiple recordings
- Time-stretch (with caution)
- Commission additional material

**Quality inconsistent:**
- Filter problematic files
- Re-record if possible
- Adjust quality rating in database

## Command Reference

### morphene instruments

```bash
# List all instruments
python -m morphene instruments --list

# Filter by tier
python -m morphene instruments --tier flagship

# Show instrument details
python -m morphene instruments --show 10
```

### morphene sources

```bash
# Add source
python -m morphene sources add -i 10 -t musical -n "Name" -o licensed

# List sources
python -m morphene sources list --instrument 10

# Show status
python -m morphene sources status 10

# Summary
python -m morphene sources summary
```

### morphene queue

```bash
# Add job
python -m morphene queue add -i 10 --db-path /data --output /models

# Add entire tier
python -m morphene queue add-tier flagship --db-base /data --output-base /models

# List jobs
python -m morphene queue list
python -m morphene queue list --status running

# Start scheduler
python -m morphene queue start

# Cancel job
python -m morphene queue cancel 1

# Retry failed job
python -m morphene queue retry 1
```

### morphene audit

```bash
# Audit single checkpoint
python -m morphene audit \
    --run /path/to/run \
    --checkpoint /path/to/checkpoint.ckpt \
    --musical-ref /ref/musical.wav \
    --nonmusical-ref /ref/nonmusical.wav \
    --output /audit/output

# Audit all checkpoints
python -m morphene audit \
    --run /path/to/run \
    --all \
    --instrument pyroviolin \
    --musical-ref /ref/musical.wav \
    --nonmusical-ref /ref/nonmusical.wav \
    --output /models \
    --checkpoint-filter 50000
```

### morphene setup

```bash
# POC only
python -m morphene setup --poc /path

# All instruments
python -m morphene setup --all /path

# Specific tier
python -m morphene setup --tier flagship /path

# Specific instrument
python -m morphene setup --instrument 10 /path
```
