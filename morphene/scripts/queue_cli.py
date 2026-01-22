#!/usr/bin/env python3
"""
Morphene Training Queue CLI

Command-line interface for managing the training queue.

Usage:
    python queue_cli.py add --instrument 10 --db-path /data/pyroviolin --output /models
    python queue_cli.py list
    python queue_cli.py status
    python queue_cli.py start
    python queue_cli.py cancel 1
"""

import argparse
import sys
import os
import time
import signal
from pathlib import Path
from typing import Optional

# Ensure morphene module is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from morphene.training import TrainingJob, JobStatus, TrainingQueue, get_queue
from morphene.instruments import get_instrument, ALL_INSTRUMENTS, Tier


def print_table(headers: list, rows: list, col_widths: Optional[list] = None) -> None:
    """Print a formatted table."""
    if not rows:
        print("No data.")
        return

    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2
                      for i in range(len(headers))]

    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    separator = "-" * len(header_line)

    print(header_line)
    print(separator)
    for row in rows:
        print("".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


def cmd_add(args, queue: TrainingQueue) -> None:
    """Add a job to the queue."""
    instrument = get_instrument(args.instrument)
    if not instrument:
        print(f"Error: Unknown instrument number {args.instrument}")
        sys.exit(1)

    job = TrainingJob(
        instrument_number=args.instrument,
        instrument_name=instrument.folder_name,
        config=args.config,
        db_path=args.db_path,
        output_path=args.output,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        priority=args.priority,
        gpu_id=args.gpu,
    )

    job_id = queue.add_job(job)
    print(f"Added job {job_id} for {instrument.name}")
    print(f"  DB path: {args.db_path}")
    print(f"  Output: {args.output}")
    print(f"  Config: {args.config}")
    print(f"  Max steps: {args.max_steps}")


def cmd_add_tier(args, queue: TrainingQueue) -> None:
    """Add all instruments from a tier to the queue."""
    tier_map = {
        '1': Tier.FLAGSHIP,
        'flagship': Tier.FLAGSHIP,
        '2': Tier.STRONG,
        'strong': Tier.STRONG,
        '3': Tier.EXPERIMENTAL,
        'experimental': Tier.EXPERIMENTAL,
    }

    tier = tier_map.get(args.tier.lower())
    if not tier:
        print(f"Error: Unknown tier '{args.tier}'")
        sys.exit(1)

    instruments = [i for i in ALL_INSTRUMENTS if i.tier == tier]
    print(f"Adding {len(instruments)} instruments from {tier.name} tier")

    for instrument in instruments:
        db_path = os.path.join(args.db_base, instrument.folder_name)
        output_path = os.path.join(args.output_base, instrument.folder_name)

        job = TrainingJob(
            instrument_number=instrument.number,
            instrument_name=instrument.folder_name,
            config=args.config,
            db_path=db_path,
            output_path=output_path,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            priority=len(instruments) - instruments.index(instrument),  # Higher priority for earlier instruments
        )

        job_id = queue.add_job(job)
        print(f"  Added job {job_id}: {instrument.name}")


def cmd_list(args, queue: TrainingQueue) -> None:
    """List jobs in the queue."""
    jobs = queue.get_all_jobs()

    if args.status:
        try:
            status = JobStatus(args.status)
            jobs = [j for j in jobs if j.status == status]
        except ValueError:
            print(f"Error: Invalid status '{args.status}'")
            sys.exit(1)

    if not jobs:
        print("No jobs found.")
        return

    print(f"\nTraining Queue ({len(jobs)} jobs)\n")

    headers = ['ID', 'Instrument', 'Status', 'GPU', 'Memory', 'Progress', 'Retries']
    rows = []

    for job in jobs:
        memory_str = f"{job.peak_memory_gb:.1f}GB" if job.peak_memory_gb > 0 else "-"
        gpu_str = str(job.gpu_id) if job.gpu_id is not None else "-"

        rows.append([
            job.id,
            job.instrument_name[:20],
            job.status.value,
            gpu_str,
            memory_str,
            "-",  # Progress would need to be read from logs
            f"{job.retry_count}/{job.max_retries}",
        ])

    print_table(headers, rows)


def cmd_show(args, queue: TrainingQueue) -> None:
    """Show detailed job info."""
    job = queue.get_job(args.id)
    if not job:
        print(f"Error: Job {args.id} not found")
        sys.exit(1)

    instrument = get_instrument(job.instrument_number)

    print(f"\nJob #{job.id}")
    print("=" * 50)
    print(f"Instrument: {instrument.name if instrument else 'Unknown'} (#{job.instrument_number})")
    print(f"Status: {job.status.value}")
    print(f"Config: {job.config}")
    print(f"DB Path: {job.db_path}")
    print(f"Output: {job.output_path}")
    print(f"Max Steps: {job.max_steps}")
    print(f"Batch Size: {job.batch_size}")
    print(f"GPU: {job.gpu_id if job.gpu_id is not None else 'Auto'}")
    print(f"Priority: {job.priority}")
    print(f"Retries: {job.retry_count}/{job.max_retries}")
    print(f"Peak Memory: {job.peak_memory_gb:.2f} GB")
    print(f"PID: {job.pid or 'N/A'}")
    print(f"Created: {job.created_at}")
    print(f"Started: {job.started_at or 'N/A'}")
    print(f"Completed: {job.completed_at or 'N/A'}")

    if job.error_message:
        print(f"Error: {job.error_message}")

    # Show memory stats
    stats = queue.get_memory_stats(job.id)
    if stats['max_gb'] > 0:
        print(f"\nMemory Statistics:")
        print(f"  Min: {stats['min_gb']:.2f} GB")
        print(f"  Max: {stats['max_gb']:.2f} GB")
        print(f"  Avg: {stats['avg_gb']:.2f} GB")

    # Show recent logs
    if args.logs:
        logs = queue.get_job_logs(job.id)
        if logs:
            print(f"\nRecent Logs ({len(logs)} entries):")
            for log in logs[-10:]:
                print(f"  [{log['timestamp']}] {log['level']}: {log['message']}")


def cmd_status(args, queue: TrainingQueue) -> None:
    """Show queue status."""
    status = queue.get_queue_status()

    print("\nTraining Queue Status")
    print("=" * 50)
    print(f"Total jobs: {status['total_jobs']}")
    print(f"Running: {status['running_jobs']}/{status['max_parallel']}")
    print()

    print("Job Status:")
    for s, count in status['status_counts'].items():
        if count > 0:
            print(f"  {s}: {count}")

    print("\nGPU Status:")
    for gpu in status['gpu_status']:
        print(f"  GPU {gpu['gpu_id']}: {gpu['used_gb']:.1f}/{gpu['total_gb']:.1f} GB "
              f"({gpu['available_gb']:.1f} GB available)")


def cmd_start(args, queue: TrainingQueue) -> None:
    """Start the queue scheduler."""
    print("Starting training queue scheduler...")
    print("Press Ctrl+C to stop")

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nStopping scheduler...")
        queue.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    queue.start()

    # Keep running
    try:
        while True:
            time.sleep(10)
            status = queue.get_queue_status()
            running = status['running_jobs']
            pending = status['status_counts']['pending']
            print(f"[{time.strftime('%H:%M:%S')}] Running: {running}, Pending: {pending}")
    except KeyboardInterrupt:
        queue.stop()


def cmd_cancel(args, queue: TrainingQueue) -> None:
    """Cancel a job."""
    job = queue.get_job(args.id)
    if not job:
        print(f"Error: Job {args.id} not found")
        sys.exit(1)

    if not args.yes:
        confirm = input(f"Cancel job {args.id} ({job.instrument_name})? [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return

    if queue.cancel_job(args.id):
        print(f"Cancelled job {args.id}")
    else:
        print(f"Failed to cancel job {args.id}")


def cmd_retry(args, queue: TrainingQueue) -> None:
    """Retry a failed job."""
    job = queue.get_job(args.id)
    if not job:
        print(f"Error: Job {args.id} not found")
        sys.exit(1)

    if job.status not in [JobStatus.FAILED, JobStatus.CANCELLED]:
        print(f"Error: Job {args.id} is not failed or cancelled (status: {job.status.value})")
        sys.exit(1)

    job.status = JobStatus.PENDING
    job.retry_count = 0
    job.error_message = ""
    job.pid = None
    job.started_at = None
    job.completed_at = None
    queue.update_job(job)

    print(f"Job {args.id} reset to pending")


def main():
    parser = argparse.ArgumentParser(
        description='Morphene Training Queue CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--db', default=None,
                        help='Path to queue database (default: data/training_queue.db)')
    parser.add_argument('--max-parallel', type=int, default=3,
                        help='Maximum parallel jobs (default: 3)')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a job')
    add_parser.add_argument('--instrument', '-i', type=int, required=True,
                            help='Instrument number')
    add_parser.add_argument('--db-path', required=True,
                            help='Path to preprocessed dataset')
    add_parser.add_argument('--output', '-o', required=True,
                            help='Output directory')
    add_parser.add_argument('--config', default='v2',
                            help='RAVE config (default: v2)')
    add_parser.add_argument('--max-steps', type=int, default=3_000_000,
                            help='Max training steps (default: 3000000)')
    add_parser.add_argument('--batch-size', type=int, default=8,
                            help='Batch size (default: 8)')
    add_parser.add_argument('--priority', type=int, default=0,
                            help='Job priority (default: 0)')
    add_parser.add_argument('--gpu', type=int,
                            help='GPU ID (default: auto)')

    # Add tier command
    add_tier_parser = subparsers.add_parser('add-tier', help='Add all instruments from a tier')
    add_tier_parser.add_argument('tier', help='Tier: 1/flagship, 2/strong, 3/experimental')
    add_tier_parser.add_argument('--db-base', required=True,
                                 help='Base path for preprocessed datasets')
    add_tier_parser.add_argument('--output-base', required=True,
                                 help='Base output directory')
    add_tier_parser.add_argument('--config', default='v2',
                                 help='RAVE config (default: v2)')
    add_tier_parser.add_argument('--max-steps', type=int, default=3_000_000,
                                 help='Max training steps')
    add_tier_parser.add_argument('--batch-size', type=int, default=8,
                                 help='Batch size')

    # List command
    list_parser = subparsers.add_parser('list', help='List jobs')
    list_parser.add_argument('--status', '-s',
                             choices=['pending', 'running', 'completed', 'failed', 'cancelled'],
                             help='Filter by status')

    # Show command
    show_parser = subparsers.add_parser('show', help='Show job details')
    show_parser.add_argument('id', type=int, help='Job ID')
    show_parser.add_argument('--logs', '-l', action='store_true',
                             help='Show recent logs')

    # Status command
    subparsers.add_parser('status', help='Show queue status')

    # Start command
    subparsers.add_parser('start', help='Start the scheduler')

    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel a job')
    cancel_parser.add_argument('id', type=int, help='Job ID')
    cancel_parser.add_argument('--yes', '-y', action='store_true',
                               help='Skip confirmation')

    # Retry command
    retry_parser = subparsers.add_parser('retry', help='Retry a failed job')
    retry_parser.add_argument('id', type=int, help='Job ID')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Get queue
    queue = get_queue(args.db, max_parallel_jobs=args.max_parallel)

    # Dispatch to command handler
    commands = {
        'add': cmd_add,
        'add-tier': cmd_add_tier,
        'list': cmd_list,
        'show': cmd_show,
        'status': cmd_status,
        'start': cmd_start,
        'cancel': cmd_cancel,
        'retry': cmd_retry,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args, queue)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
