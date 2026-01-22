# Morphene Instrument Pipeline

Production pipeline for creating 50 neural audio instruments using RAVE.

## Overview

Morphene is a collection of 50 neural audio instruments built on RAVE, designed for real-time timbre morphing between non-musical and musical sound sources. Each instrument exists as a trained model loadable in nn~ for Max/MSP, PureData, or compatible DAW environments.

## Components

### Instrument Definitions (`instruments/`)

Complete definitions for all 50 instruments across three tiers:

- **Tier 1 - Flagship** (10 instruments): Highest priority, proven pairings
- **Tier 2 - Strong** (20 instruments): Solid, reliable pairings
- **Tier 3 - Experimental** (20 instruments): Creative exploration, higher risk

```python
from morphene.instruments import get_instrument, ALL_INSTRUMENTS, get_poc_instrument

# Get the POC instrument (Pyroviolin)
poc = get_poc_instrument()
print(f"{poc.name}: {poc.musical_source} + {poc.nonmusical_source}")

# Get any instrument by number or name
inst = get_instrument(10)  # or get_instrument("pyroviolin")
```

### Source Tracking (`sources/`)

SQLite-based database for tracking audio source metadata:

- Origin (recorded, licensed, commissioned, synthesized)
- License information and restrictions
- Total duration collected
- Quality ratings
- File locations and processing history

```bash
# Add a source
python -m morphene.scripts.sources_cli add \
    --instrument 10 --type musical --name "Violin samples" \
    --origin licensed --license "Royalty-free" --duration 5400

# Check instrument status
python -m morphene.scripts.sources_cli status 10

# View collection summary
python -m morphene.scripts.sources_cli summary
```

### Training Queue (`training/`)

Parallel training job management with:

- GPU memory monitoring
- Automatic job scheduling
- Failure detection and auto-retry
- Central logging and monitoring

```bash
# Add a training job
python -m morphene.scripts.queue_cli add \
    --instrument 10 --db-path /data/pyroviolin --output /models

# Add all Tier 1 instruments
python -m morphene.scripts.queue_cli add-tier flagship \
    --db-base /data --output-base /models

# Start the scheduler
python -m morphene.scripts.queue_cli start

# Check queue status
python -m morphene.scripts.queue_cli status
```

### Audit System (`scripts/audit.py`)

Checkpoint evaluation for quality control:

- Generates reconstruction audio from musical and non-musical references
- Creates morph interpolations at 0%, 25%, 50%, 75%, 100%
- Saves structured audit outputs for comparison

```bash
# Audit a single checkpoint
python -m morphene.scripts.audit \
    --run /path/to/run --checkpoint /path/to/checkpoint.ckpt \
    --musical-ref /path/to/violin.wav \
    --nonmusical-ref /path/to/fire.wav \
    --output /models/pyroviolin/audit/050000

# Audit all checkpoints
python -m morphene.scripts.audit \
    --run /path/to/run --all \
    --instrument pyroviolin \
    --musical-ref /path/to/violin.wav \
    --nonmusical-ref /path/to/fire.wav \
    --output /models
```

### Project Setup (`scripts/setup_project.py`)

Folder structure initialization:

```bash
# Set up POC only
python -m morphene.scripts.setup_project --poc /path/to/project

# Set up all instruments
python -m morphene.scripts.setup_project --all /path/to/project

# Set up a specific tier
python -m morphene.scripts.setup_project --tier flagship /path/to/project
```

## Folder Structure

```
/models/{instrument_name}/
├── checkpoints/           # Training checkpoints
│   ├── checkpoint_050000.pt
│   ├── checkpoint_100000.pt
│   └── ...
├── audit/                 # Audit outputs per checkpoint
│   ├── 050000/
│   │   ├── musical_recon.wav
│   │   ├── nonmusical_recon.wav
│   │   ├── morph_00.wav
│   │   ├── morph_25.wav
│   │   ├── morph_50.wav
│   │   ├── morph_75.wav
│   │   └── morph_100.wav
│   └── 100000/
│       └── ...
└── final/                 # Final shipped model
    ├── {instrument_name}.ts
    ├── metadata.json
    └── demo.wav
```

## Workflow

### POC (Pyroviolin)

1. **Day 1-2**: Source collection and preprocessing
2. **Day 3**: Launch training, monitor
3. **Day 4**: Run audit, evaluate checkpoints
4. **Day 5**: Export to .ts, test in nn~
5. **Day 6-7**: Document learnings, finalize

### Production

1. **Source Collection**: Use `sources_cli` to track progress
2. **Preprocessing**: Use RAVE's `preprocess.py`
3. **Training**: Add jobs with `queue_cli`, start scheduler
4. **Audit**: Run `audit.py` on checkpoints
5. **Selection**: Listen, compare, select best checkpoint
6. **Export**: Use RAVE's `export.py`
7. **Delivery**: Package with metadata and demo

## Audio Requirements

Per instrument:
- **1.5 hours minimum per source** (3 hours total)
- Mono, 48kHz WAV
- Normalized loudness, trimmed silence
- Full range of pitch, dynamics, articulations

## Training Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | RAVE v2 |
| Sample rate | 48kHz |
| Latent dimensions | 128 (default) |
| Training duration | ~12-24 hours |
| Checkpoint interval | Every 50,000 steps |

## Quality Validation

Before shipping, each model must pass:

| Test | Pass Criteria |
|------|---------------|
| Musical reconstruction | Recognizable, faithful |
| Non-musical reconstruction | Captures source texture |
| Morph progression | Smooth, no artifacts |
| Real-time playback | Stable in nn~ |
| Musicality | Usable across morph range |
