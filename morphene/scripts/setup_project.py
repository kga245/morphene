#!/usr/bin/env python3
"""
Morphene Project Setup

Sets up the folder structure for Morphene instrument development as specified in the PRD:

/models/{instrument_name}/
  /checkpoints/
    checkpoint_050000.pt
    checkpoint_100000.pt
    ...
  /audit/
    /050000/
      musical_recon.wav
      nonmusical_recon.wav
      morph_00.wav
      morph_25.wav
      morph_50.wav
      morph_75.wav
      morph_100.wav
    /100000/
      ...
  /final/
    {instrument_name}.ts
    metadata.json
    demo.wav

Also sets up:
- Source audio directories
- Reference clip directories
- Training data directories
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Ensure morphene module is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from morphene.instruments import (
    ALL_INSTRUMENTS,
    TIER1_FLAGSHIP,
    TIER2_STRONG,
    TIER3_EXPERIMENTAL,
    Tier,
    get_instrument,
    get_poc_instrument,
)


def create_instrument_structure(
    base_path: str,
    instrument_name: str,
    include_sources: bool = True
) -> List[str]:
    """
    Create the folder structure for a single instrument.

    Args:
        base_path: Base path for models
        instrument_name: Name of the instrument
        include_sources: Whether to create source directories

    Returns:
        List of created directories
    """
    created = []

    # Model directories
    model_dirs = [
        f"models/{instrument_name}/checkpoints",
        f"models/{instrument_name}/audit",
        f"models/{instrument_name}/final",
    ]

    # Source directories
    source_dirs = [
        f"sources/{instrument_name}/musical/raw",
        f"sources/{instrument_name}/musical/processed",
        f"sources/{instrument_name}/nonmusical/raw",
        f"sources/{instrument_name}/nonmusical/processed",
        f"sources/{instrument_name}/reference_clips",
    ]

    # Training data directories
    training_dirs = [
        f"training_data/{instrument_name}",
    ]

    all_dirs = model_dirs + training_dirs
    if include_sources:
        all_dirs += source_dirs

    for dir_path in all_dirs:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        created.append(full_path)

    return created


def create_instrument_readme(base_path: str, instrument) -> str:
    """
    Create a README for an instrument.

    Args:
        base_path: Base path for models
        instrument: Instrument object

    Returns:
        Path to created README
    """
    readme_path = os.path.join(base_path, f"models/{instrument.folder_name}/README.md")

    content = f"""# {instrument.name}

## Instrument #{instrument.number}

**Tier:** {instrument.tier.name}

### Sources

| Type | Description |
|------|-------------|
| Musical | {instrument.musical_source} |
| Non-musical | {instrument.nonmusical_source} |

### Musical Source Details
{instrument.musical_source_description or "Not yet specified"}

### Non-musical Source Details
{instrument.nonmusical_source_description or "Not yet specified"}

### Notes
{instrument.notes or "None"}

---

## Folder Structure

```
{instrument.folder_name}/
├── checkpoints/      # Training checkpoints
├── audit/            # Audit outputs per checkpoint
│   ├── 050000/       # Checkpoint step
│   │   ├── musical_recon.wav
│   │   ├── nonmusical_recon.wav
│   │   ├── morph_00.wav
│   │   ├── morph_25.wav
│   │   ├── morph_50.wav
│   │   ├── morph_75.wav
│   │   └── morph_100.wav
│   └── ...
└── final/            # Final exported model
    ├── {instrument.folder_name}.ts
    ├── metadata.json
    └── demo.wav
```

## Training Status

- [ ] Sources collected
- [ ] Reference clips selected
- [ ] Training complete
- [ ] Audit complete
- [ ] Final model exported
- [ ] Demo audio generated

---
Generated: {datetime.now().isoformat()}
"""

    os.makedirs(os.path.dirname(readme_path), exist_ok=True)
    with open(readme_path, 'w') as f:
        f.write(content)

    return readme_path


def create_metadata_template(base_path: str, instrument) -> str:
    """
    Create a metadata.json template for an instrument.

    Args:
        base_path: Base path for models
        instrument: Instrument object

    Returns:
        Path to created metadata file
    """
    metadata_path = os.path.join(
        base_path,
        f"models/{instrument.folder_name}/final/metadata.json"
    )

    metadata = {
        "instrument": {
            "number": instrument.number,
            "name": instrument.name,
            "tier": instrument.tier.name,
        },
        "sources": {
            "musical": {
                "description": instrument.musical_source,
                "details": instrument.musical_source_description,
                "origin": "",
                "license": "",
                "duration_hours": 0.0,
            },
            "nonmusical": {
                "description": instrument.nonmusical_source,
                "details": instrument.nonmusical_source_description,
                "origin": "",
                "license": "",
                "duration_hours": 0.0,
            },
        },
        "training": {
            "config": "v2",
            "selected_checkpoint": "",
            "selection_reasoning": "",
            "total_steps": 0,
            "training_hours": 0.0,
            "peak_memory_gb": 0.0,
        },
        "latent_space": {
            "dimensions": 128,
            "notes": "",
        },
        "suggested_use_cases": [],
        "export": {
            "format": "torchscript",
            "streaming": True,
            "sample_rate": 48000,
        },
        "created_at": "",
        "updated_at": "",
    }

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def setup_poc(base_path: str) -> None:
    """Set up POC (Pyroviolin) structure."""
    poc = get_poc_instrument()

    print(f"\nSetting up POC: {poc.name}")
    print("=" * 50)

    # Create structure
    dirs = create_instrument_structure(base_path, poc.folder_name)
    print(f"Created {len(dirs)} directories")

    # Create README
    readme = create_instrument_readme(base_path, poc)
    print(f"Created README: {readme}")

    # Create metadata template
    metadata = create_metadata_template(base_path, poc)
    print(f"Created metadata template: {metadata}")

    # Create POC checklist
    checklist_path = os.path.join(base_path, "models", poc.folder_name, "POC_CHECKLIST.md")
    checklist = f"""# POC Checklist: {poc.name}

## Day 1
- [ ] Identify violin source (library or record)
- [ ] Identify fire source (Freesound, record, or library)
- [ ] Begin collecting/recording

## Day 2
- [ ] Complete source collection (1.5 hrs each)
- [ ] Preprocess: convert to mono 48kHz WAV
- [ ] Normalize loudness
- [ ] Trim silence
- [ ] Select 30-second reference clips for audit
- [ ] Stage files for training

## Day 3
- [ ] Launch RAVE training on DGX Spark
- [ ] Confirm Tensorboard logging active
- [ ] Monitor first checkpoints
- [ ] Log peak memory usage

## Day 4
- [ ] Training completes (or continues if needed)
- [ ] Run audit script against all checkpoints
- [ ] Listen to reconstruction and morph progression
- [ ] Identify best checkpoint

## Day 5
- [ ] If quality passes: export to .ts
- [ ] Load in nn~ (Max/MSP or PureData)
- [ ] Test real-time playback
- [ ] If quality fails: diagnose, adjust, relaunch training

## Day 6
- [ ] Finalize POC model
- [ ] Document: actual training time, memory usage, issues encountered
- [ ] Confirm parallel training strategy based on benchmarks

## Day 7
- [ ] Buffer for iteration
- [ ] Begin sourcing Tier 1 batch 2 (next 3 instruments)
- [ ] POC complete

---
Generated: {datetime.now().isoformat()}
"""
    with open(checklist_path, 'w') as f:
        f.write(checklist)
    print(f"Created POC checklist: {checklist_path}")


def setup_tier(base_path: str, tier: Tier, include_sources: bool = True) -> None:
    """Set up structure for all instruments in a tier."""
    tier_instruments = {
        Tier.FLAGSHIP: TIER1_FLAGSHIP,
        Tier.STRONG: TIER2_STRONG,
        Tier.EXPERIMENTAL: TIER3_EXPERIMENTAL,
    }

    instruments = tier_instruments[tier]

    print(f"\nSetting up {tier.name} tier ({len(instruments)} instruments)")
    print("=" * 50)

    for inst in instruments:
        dirs = create_instrument_structure(base_path, inst.folder_name, include_sources)
        readme = create_instrument_readme(base_path, inst)
        metadata = create_metadata_template(base_path, inst)
        print(f"  {inst.name}: {len(dirs)} dirs, README, metadata")


def setup_all(base_path: str, include_sources: bool = True) -> None:
    """Set up structure for all 50 instruments."""
    print("\nSetting up Morphene Project Structure")
    print("=" * 50)

    # Create top-level directories
    top_dirs = [
        "models",
        "sources",
        "training_data",
        "data",
        "exports",
    ]

    for d in top_dirs:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)

    # Set up each tier
    for tier in [Tier.FLAGSHIP, Tier.STRONG, Tier.EXPERIMENTAL]:
        setup_tier(base_path, tier, include_sources)

    # Create master index
    create_master_index(base_path)

    print(f"\nSetup complete!")
    print(f"Total instruments: {len(ALL_INSTRUMENTS)}")


def create_master_index(base_path: str) -> str:
    """Create the master index of all instruments."""
    index_path = os.path.join(base_path, "models/INDEX.md")

    lines = [
        "# Morphene Instrument Index",
        "",
        f"Total instruments: {len(ALL_INSTRUMENTS)}",
        "",
        "## Tier 1 - Flagship (10 instruments)",
        "",
        "| # | Name | Musical | Non-musical | Status |",
        "|---|------|---------|-------------|--------|",
    ]

    for inst in TIER1_FLAGSHIP:
        lines.append(
            f"| {inst.number} | {inst.name} | {inst.musical_source} | "
            f"{inst.nonmusical_source} | Pending |"
        )

    lines.extend([
        "",
        "## Tier 2 - Strong (20 instruments)",
        "",
        "| # | Name | Musical | Non-musical | Status |",
        "|---|------|---------|-------------|--------|",
    ])

    for inst in TIER2_STRONG:
        lines.append(
            f"| {inst.number} | {inst.name} | {inst.musical_source} | "
            f"{inst.nonmusical_source} | Pending |"
        )

    lines.extend([
        "",
        "## Tier 3 - Experimental (20 instruments)",
        "",
        "| # | Name | Musical | Non-musical | Status |",
        "|---|------|---------|-------------|--------|",
    ])

    for inst in TIER3_EXPERIMENTAL:
        lines.append(
            f"| {inst.number} | {inst.name} | {inst.musical_source} | "
            f"{inst.nonmusical_source} | Pending |"
        )

    lines.extend([
        "",
        "---",
        f"Generated: {datetime.now().isoformat()}",
    ])

    with open(index_path, 'w') as f:
        f.write('\n'.join(lines))

    return index_path


def main():
    parser = argparse.ArgumentParser(
        description='Set up Morphene project structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set up POC only
  python setup_project.py --poc /path/to/morphene

  # Set up a specific tier
  python setup_project.py --tier flagship /path/to/morphene

  # Set up all instruments
  python setup_project.py --all /path/to/morphene

  # Set up single instrument
  python setup_project.py --instrument 10 /path/to/morphene
        """
    )

    parser.add_argument('base_path', help='Base path for project')
    parser.add_argument('--poc', action='store_true',
                        help='Set up POC (Pyroviolin) only')
    parser.add_argument('--tier', choices=['flagship', 'strong', 'experimental'],
                        help='Set up a specific tier')
    parser.add_argument('--all', action='store_true',
                        help='Set up all instruments')
    parser.add_argument('--instrument', '-i', type=int,
                        help='Set up a specific instrument by number')
    parser.add_argument('--no-sources', action='store_true',
                        help='Skip creating source directories')

    args = parser.parse_args()

    base_path = os.path.abspath(args.base_path)
    include_sources = not args.no_sources

    if args.poc:
        setup_poc(base_path)
    elif args.tier:
        tier_map = {
            'flagship': Tier.FLAGSHIP,
            'strong': Tier.STRONG,
            'experimental': Tier.EXPERIMENTAL,
        }
        setup_tier(base_path, tier_map[args.tier], include_sources)
    elif args.all:
        setup_all(base_path, include_sources)
    elif args.instrument:
        inst = get_instrument(args.instrument)
        if not inst:
            print(f"Error: Unknown instrument number {args.instrument}")
            sys.exit(1)
        print(f"\nSetting up: {inst.name}")
        dirs = create_instrument_structure(base_path, inst.folder_name, include_sources)
        create_instrument_readme(base_path, inst)
        create_metadata_template(base_path, inst)
        print(f"Created {len(dirs)} directories")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
