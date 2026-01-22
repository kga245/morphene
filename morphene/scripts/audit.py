#!/usr/bin/env python3
"""
Morphene Training Audit System

Generates reconstruction and morph interpolation audio from RAVE checkpoints
for quality control and best-checkpoint selection.

For each checkpoint, generates:
- musical_recon.wav: Reconstruction of reference musical clip (30 sec)
- nonmusical_recon.wav: Reconstruction of reference non-musical clip (30 sec)
- morph_00.wav: 0% interpolation (pure musical)
- morph_25.wav: 25% interpolation
- morph_50.wav: 50% interpolation
- morph_75.wav: 75% interpolation
- morph_100.wav: 100% interpolation (pure non-musical)
"""

import os
import sys
import glob
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

import torch
import torchaudio
import gin

# Ensure rave module is importable
try:
    import rave
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import rave

import rave.core
import rave.blocks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint(run_path: str, checkpoint_path: Optional[str] = None) -> Tuple[rave.RAVE, int]:
    """
    Load a RAVE model from a checkpoint.

    Args:
        run_path: Path to the training run directory
        checkpoint_path: Optional specific checkpoint path. If None, uses latest.

    Returns:
        Tuple of (model, step_number)
    """
    # Find and parse config
    config_file = rave.core.search_for_config(run_path)
    if config_file is None:
        raise FileNotFoundError(f"Config file not found in {run_path}")

    gin.clear_config()
    gin.parse_config_file(config_file)

    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = rave.core.search_for_run(run_path)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {run_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Load model
    model = rave.RAVE()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    # Extract step number from checkpoint filename
    step = 0
    try:
        # Checkpoint names are like: epoch=X-step=XXXXXX.ckpt
        basename = os.path.basename(checkpoint_path)
        if 'step=' in basename:
            step = int(basename.split('step=')[1].split('.')[0].split('-')[0])
    except (ValueError, IndexError):
        logger.warning(f"Could not parse step number from {checkpoint_path}")

    return model, step


def load_audio(
    audio_path: str,
    target_sr: int,
    target_length: Optional[int] = None,
    n_channels: int = 1
) -> torch.Tensor:
    """
    Load and preprocess audio file.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        target_length: Target length in samples (truncate or pad)
        n_channels: Number of channels

    Returns:
        Audio tensor of shape (1, n_channels, length)
    """
    audio, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    # Convert to mono if needed
    if audio.shape[0] > n_channels:
        audio = audio[:n_channels]
    elif audio.shape[0] < n_channels:
        audio = audio.repeat(n_channels, 1)[:n_channels]

    # Adjust length
    if target_length is not None:
        if audio.shape[1] > target_length:
            audio = audio[:, :target_length]
        elif audio.shape[1] < target_length:
            padding = target_length - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))

    # Add batch dimension
    return audio.unsqueeze(0)


def reconstruct(model: rave.RAVE, audio: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct audio through the model (encode -> decode).

    Args:
        model: RAVE model
        audio: Input audio tensor (batch, channels, length)

    Returns:
        Reconstructed audio tensor
    """
    with torch.no_grad():
        z = model.encode(audio)
        # Handle different encoder types
        if hasattr(model.encoder, 'reparametrize'):
            z = model.encoder.reparametrize(z)[0]
        y = model.decode(z)
        # Ensure output matches input length
        y = y[..., :audio.shape[-1]]
    return y


def encode_to_latent(model: rave.RAVE, audio: torch.Tensor) -> torch.Tensor:
    """
    Encode audio to latent representation.

    Args:
        model: RAVE model
        audio: Input audio tensor

    Returns:
        Latent tensor
    """
    with torch.no_grad():
        z = model.encode(audio)
        if hasattr(model.encoder, 'reparametrize'):
            z = model.encoder.reparametrize(z)[0]
    return z


def interpolate_latent(
    z1: torch.Tensor,
    z2: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Linearly interpolate between two latent representations.

    Args:
        z1: First latent tensor
        z2: Second latent tensor
        alpha: Interpolation factor (0.0 = z1, 1.0 = z2)

    Returns:
        Interpolated latent tensor
    """
    return (1 - alpha) * z1 + alpha * z2


def generate_morph(
    model: rave.RAVE,
    z_musical: torch.Tensor,
    z_nonmusical: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Generate morph audio at a given interpolation point.

    Args:
        model: RAVE model
        z_musical: Latent representation of musical audio
        z_nonmusical: Latent representation of non-musical audio
        alpha: Interpolation factor

    Returns:
        Morphed audio tensor
    """
    with torch.no_grad():
        z_morph = interpolate_latent(z_musical, z_nonmusical, alpha)
        y = model.decode(z_morph)
    return y


def save_audio(audio: torch.Tensor, path: str, sample_rate: int) -> None:
    """
    Save audio tensor to file.

    Args:
        audio: Audio tensor (batch, channels, length)
        path: Output file path
        sample_rate: Sample rate
    """
    # Remove batch dimension if present
    if audio.dim() == 3:
        audio = audio.squeeze(0)

    # Ensure audio is in valid range
    audio = torch.clamp(audio, -1.0, 1.0)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, audio.cpu(), sample_rate)
    logger.info(f"Saved: {path}")


def run_audit_for_checkpoint(
    run_path: str,
    checkpoint_path: str,
    musical_ref_path: str,
    nonmusical_ref_path: str,
    output_dir: str,
    ref_duration_sec: float = 30.0,
    device: str = 'cpu',
    morph_points: List[float] = None
) -> dict:
    """
    Run full audit for a single checkpoint.

    Args:
        run_path: Path to training run directory
        checkpoint_path: Path to specific checkpoint
        musical_ref_path: Path to musical reference audio
        nonmusical_ref_path: Path to non-musical reference audio
        output_dir: Directory to save audit outputs
        ref_duration_sec: Duration of reference clips in seconds
        device: Device to run on ('cpu' or 'cuda:X')
        morph_points: List of interpolation points (default: [0, 0.25, 0.5, 0.75, 1.0])

    Returns:
        Dict with audit metadata
    """
    if morph_points is None:
        morph_points = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Load model
    model, step = load_checkpoint(run_path, checkpoint_path)
    model = model.to(device)

    sample_rate = model.sr
    n_channels = model.n_channels
    ref_length = int(ref_duration_sec * sample_rate)

    # Load reference audio
    logger.info(f"Loading musical reference: {musical_ref_path}")
    musical_audio = load_audio(
        musical_ref_path,
        sample_rate,
        ref_length,
        n_channels
    ).to(device)

    logger.info(f"Loading non-musical reference: {nonmusical_ref_path}")
    nonmusical_audio = load_audio(
        nonmusical_ref_path,
        sample_rate,
        ref_length,
        n_channels
    ).to(device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate reconstructions
    logger.info("Generating musical reconstruction...")
    musical_recon = reconstruct(model, musical_audio)
    save_audio(
        musical_recon,
        os.path.join(output_dir, 'musical_recon.wav'),
        sample_rate
    )

    logger.info("Generating non-musical reconstruction...")
    nonmusical_recon = reconstruct(model, nonmusical_audio)
    save_audio(
        nonmusical_recon,
        os.path.join(output_dir, 'nonmusical_recon.wav'),
        sample_rate
    )

    # Encode to latent space
    logger.info("Encoding to latent space...")
    z_musical = encode_to_latent(model, musical_audio)
    z_nonmusical = encode_to_latent(model, nonmusical_audio)

    # Generate morph interpolations
    for alpha in morph_points:
        logger.info(f"Generating morph at {int(alpha * 100)}%...")
        morph_audio = generate_morph(model, z_musical, z_nonmusical, alpha)
        # Ensure output matches expected length
        morph_audio = morph_audio[..., :ref_length]
        save_audio(
            morph_audio,
            os.path.join(output_dir, f'morph_{int(alpha * 100):02d}.wav'),
            sample_rate
        )

    # Save audit metadata
    metadata = {
        'checkpoint': checkpoint_path,
        'step': step,
        'musical_reference': musical_ref_path,
        'nonmusical_reference': nonmusical_ref_path,
        'sample_rate': sample_rate,
        'duration_sec': ref_duration_sec,
        'morph_points': morph_points,
        'timestamp': datetime.now().isoformat(),
        'device': device,
    }

    with open(os.path.join(output_dir, 'audit_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def find_checkpoints(run_path: str) -> List[str]:
    """
    Find all checkpoints in a training run directory.

    Args:
        run_path: Path to training run directory

    Returns:
        List of checkpoint paths sorted by step number
    """
    checkpoints = []

    # Check for checkpoints in various locations
    patterns = [
        os.path.join(run_path, '**', '*.ckpt'),
        os.path.join(run_path, 'checkpoints', '*.ckpt'),
        os.path.join(run_path, 'lightning_logs', '**', '*.ckpt'),
    ]

    for pattern in patterns:
        checkpoints.extend(glob.glob(pattern, recursive=True))

    # Remove duplicates and sort by step number
    checkpoints = list(set(checkpoints))

    def get_step(path):
        try:
            basename = os.path.basename(path)
            if 'step=' in basename:
                return int(basename.split('step=')[1].split('.')[0].split('-')[0])
        except (ValueError, IndexError):
            pass
        return 0

    checkpoints.sort(key=get_step)
    return checkpoints


def run_full_audit(
    run_path: str,
    instrument_name: str,
    musical_ref_path: str,
    nonmusical_ref_path: str,
    output_base: str,
    ref_duration_sec: float = 30.0,
    device: str = 'cpu',
    checkpoint_filter: Optional[int] = None
) -> List[dict]:
    """
    Run audit for all checkpoints in a training run.

    Args:
        run_path: Path to training run directory
        instrument_name: Name of the instrument
        musical_ref_path: Path to musical reference audio
        nonmusical_ref_path: Path to non-musical reference audio
        output_base: Base directory for audit outputs
        ref_duration_sec: Duration of reference clips in seconds
        device: Device to run on
        checkpoint_filter: Only audit checkpoints at multiples of this step

    Returns:
        List of audit metadata dicts
    """
    checkpoints = find_checkpoints(run_path)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_path}")

    logger.info(f"Found {len(checkpoints)} checkpoints")

    # Filter checkpoints if requested
    if checkpoint_filter is not None:
        filtered = []
        for ckpt in checkpoints:
            try:
                basename = os.path.basename(ckpt)
                if 'step=' in basename:
                    step = int(basename.split('step=')[1].split('.')[0].split('-')[0])
                    if step % checkpoint_filter == 0:
                        filtered.append(ckpt)
            except (ValueError, IndexError):
                continue
        checkpoints = filtered
        logger.info(f"Filtered to {len(checkpoints)} checkpoints (every {checkpoint_filter} steps)")

    results = []
    for ckpt_path in checkpoints:
        try:
            # Get step number for output directory
            basename = os.path.basename(ckpt_path)
            if 'step=' in basename:
                step = int(basename.split('step=')[1].split('.')[0].split('-')[0])
                step_dir = f'{step:06d}'
            else:
                step_dir = Path(ckpt_path).stem

            output_dir = os.path.join(output_base, instrument_name, 'audit', step_dir)

            logger.info(f"\n{'='*60}")
            logger.info(f"Auditing checkpoint: {ckpt_path}")
            logger.info(f"Output: {output_dir}")
            logger.info('='*60)

            metadata = run_audit_for_checkpoint(
                run_path=run_path,
                checkpoint_path=ckpt_path,
                musical_ref_path=musical_ref_path,
                nonmusical_ref_path=nonmusical_ref_path,
                output_dir=output_dir,
                ref_duration_sec=ref_duration_sec,
                device=device
            )
            results.append(metadata)

        except Exception as e:
            logger.error(f"Failed to audit {ckpt_path}: {e}")
            continue

    return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Morphene Training Audit System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Audit a single checkpoint
  python audit.py --run /path/to/run --checkpoint /path/to/checkpoint.ckpt \\
      --musical-ref /path/to/violin.wav --nonmusical-ref /path/to/fire.wav \\
      --output /models/pyroviolin/audit/050000

  # Audit all checkpoints in a run
  python audit.py --run /path/to/run --all \\
      --instrument pyroviolin \\
      --musical-ref /path/to/violin.wav --nonmusical-ref /path/to/fire.wav \\
      --output /models

  # Audit checkpoints at specific intervals
  python audit.py --run /path/to/run --all --checkpoint-filter 50000 \\
      --instrument pyroviolin \\
      --musical-ref /path/to/violin.wav --nonmusical-ref /path/to/fire.wav \\
      --output /models
        """
    )

    parser.add_argument('--run', required=True, help='Path to training run directory')
    parser.add_argument('--checkpoint', help='Path to specific checkpoint (if not --all)')
    parser.add_argument('--all', action='store_true', help='Audit all checkpoints')
    parser.add_argument('--checkpoint-filter', type=int,
                        help='Only audit checkpoints at multiples of this step')
    parser.add_argument('--instrument', help='Instrument name (required with --all)')
    parser.add_argument('--musical-ref', required=True, help='Path to musical reference audio')
    parser.add_argument('--nonmusical-ref', required=True, help='Path to non-musical reference audio')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Duration of reference clips in seconds (default: 30)')
    parser.add_argument('--device', default='cpu',
                        help='Device to run on (cpu or cuda:X)')

    args = parser.parse_args()

    if args.all:
        if not args.instrument:
            parser.error('--instrument is required when using --all')

        results = run_full_audit(
            run_path=args.run,
            instrument_name=args.instrument,
            musical_ref_path=args.musical_ref,
            nonmusical_ref_path=args.nonmusical_ref,
            output_base=args.output,
            ref_duration_sec=args.duration,
            device=args.device,
            checkpoint_filter=args.checkpoint_filter
        )
        logger.info(f"\nCompleted audit for {len(results)} checkpoints")
    else:
        output_dir = args.output
        if args.checkpoint is None:
            # Use latest checkpoint
            checkpoints = find_checkpoints(args.run)
            if not checkpoints:
                logger.error(f"No checkpoints found in {args.run}")
                sys.exit(1)
            args.checkpoint = checkpoints[-1]
            logger.info(f"Using latest checkpoint: {args.checkpoint}")

        metadata = run_audit_for_checkpoint(
            run_path=args.run,
            checkpoint_path=args.checkpoint,
            musical_ref_path=args.musical_ref,
            nonmusical_ref_path=args.nonmusical_ref,
            output_dir=output_dir,
            ref_duration_sec=args.duration,
            device=args.device
        )
        logger.info(f"\nAudit complete. Output: {output_dir}")


if __name__ == '__main__':
    main()
