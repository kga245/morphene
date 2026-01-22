#!/usr/bin/env python3
"""
Morphene Source Tracking CLI

Command-line interface for managing audio source metadata.

Usage:
    python sources_cli.py add --instrument 10 --type musical --name "Violin samples" \\
        --origin licensed --license "Royalty-free" --duration 5400
    python sources_cli.py list --instrument 10
    python sources_cli.py status --instrument 10
    python sources_cli.py summary
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Ensure morphene module is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from morphene.sources import (
    SourceDatabase,
    AudioSource,
    SourceOrigin,
    SourceType,
    get_database,
)
from morphene.instruments import get_instrument, ALL_INSTRUMENTS


def print_table(headers: list, rows: list, col_widths: Optional[list] = None) -> None:
    """Print a formatted table."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2
                      for i in range(len(headers))]

    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    separator = "-" * len(header_line)

    print(header_line)
    print(separator)
    for row in rows:
        print("".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


def cmd_add(args, db: SourceDatabase) -> None:
    """Add a new source."""
    # Validate instrument
    instrument = get_instrument(args.instrument)
    if not instrument:
        print(f"Error: Unknown instrument number {args.instrument}")
        sys.exit(1)

    # Parse source type
    try:
        source_type = SourceType(args.type)
    except ValueError:
        print(f"Error: Invalid source type '{args.type}'. Use 'musical' or 'nonmusical'")
        sys.exit(1)

    # Parse origin
    try:
        origin = SourceOrigin(args.origin)
    except ValueError:
        valid = ', '.join(o.value for o in SourceOrigin)
        print(f"Error: Invalid origin '{args.origin}'. Use one of: {valid}")
        sys.exit(1)

    # Parse file paths
    file_paths = args.files.split(',') if args.files else []

    # Parse processing
    processing = args.processing.split(',') if args.processing else []

    source = AudioSource(
        instrument_number=args.instrument,
        source_type=source_type,
        name=args.name,
        origin=origin,
        license_type=args.license or "",
        license_restrictions=args.restrictions or "",
        total_duration_sec=args.duration or 0.0,
        quality_rating=args.quality or 0,
        processing_applied=processing,
        file_paths=file_paths,
        notes=args.notes or "",
    )

    source_id = db.add_source(source)
    print(f"Added source with ID {source_id}")
    print(f"  Instrument: {instrument.name} (#{args.instrument})")
    print(f"  Type: {source_type.value}")
    print(f"  Name: {args.name}")
    print(f"  Duration: {args.duration / 3600:.2f} hours" if args.duration else "  Duration: Not set")


def cmd_update(args, db: SourceDatabase) -> None:
    """Update an existing source."""
    source = db.get_source(args.id)
    if not source:
        print(f"Error: Source with ID {args.id} not found")
        sys.exit(1)

    if args.name:
        source.name = args.name
    if args.license:
        source.license_type = args.license
    if args.restrictions:
        source.license_restrictions = args.restrictions
    if args.duration is not None:
        source.total_duration_sec = args.duration
    if args.quality is not None:
        source.quality_rating = args.quality
    if args.notes:
        source.notes = args.notes
    if args.files:
        source.file_paths = args.files.split(',')
    if args.processing:
        source.processing_applied = args.processing.split(',')

    db.update_source(source)
    print(f"Updated source {args.id}")


def cmd_list(args, db: SourceDatabase) -> None:
    """List sources."""
    if args.instrument:
        sources = db.get_sources_for_instrument(args.instrument)
        instrument = get_instrument(args.instrument)
        if instrument:
            print(f"\nSources for {instrument.name} (#{args.instrument}):\n")
    else:
        sources = db.get_all_sources()
        print(f"\nAll sources ({len(sources)} total):\n")

    if not sources:
        print("No sources found.")
        return

    headers = ['ID', 'Inst#', 'Type', 'Name', 'Origin', 'Duration (h)', 'Quality']
    rows = []
    for s in sources:
        rows.append([
            s.id,
            s.instrument_number,
            s.source_type.value[:3],
            s.name[:30] + ('...' if len(s.name) > 30 else ''),
            s.origin.value[:8],
            f"{s.duration_hours:.2f}",
            s.quality_rating or '-'
        ])

    print_table(headers, rows)


def cmd_show(args, db: SourceDatabase) -> None:
    """Show detailed info about a source."""
    source = db.get_source(args.id)
    if not source:
        print(f"Error: Source with ID {args.id} not found")
        sys.exit(1)

    instrument = get_instrument(source.instrument_number)

    print(f"\nSource #{source.id}")
    print("=" * 50)
    print(f"Instrument: {instrument.name if instrument else 'Unknown'} (#{source.instrument_number})")
    print(f"Type: {source.source_type.value}")
    print(f"Name: {source.name}")
    print(f"Origin: {source.origin.value}")
    print(f"License: {source.license_type or 'Not specified'}")
    print(f"Restrictions: {source.license_restrictions or 'None'}")
    print(f"Duration: {source.duration_hours:.2f} hours ({source.total_duration_sec:.0f} sec)")
    print(f"Meets minimum (1.5h): {'Yes' if source.meets_minimum_duration else 'No'}")
    print(f"Quality rating: {source.quality_rating or 'Not rated'}")
    print(f"Processing: {', '.join(source.processing_applied) or 'None'}")
    print(f"Files: {len(source.file_paths)} file(s)")
    for f in source.file_paths:
        print(f"  - {f}")
    print(f"Notes: {source.notes or 'None'}")
    print(f"Created: {source.created_at}")
    print(f"Updated: {source.updated_at}")

    # Show reference clips
    clips = db.get_reference_clips(source.id)
    if clips:
        print(f"\nReference clips ({len(clips)}):")
        for clip in clips:
            print(f"  - {clip['file_path']} ({clip['duration_sec']:.1f}s)")


def cmd_status(args, db: SourceDatabase) -> None:
    """Show status for an instrument."""
    instrument = get_instrument(args.instrument)
    if not instrument:
        print(f"Error: Unknown instrument number {args.instrument}")
        sys.exit(1)

    status = db.get_instrument_status(args.instrument)

    print(f"\nStatus for {instrument.name} (#{args.instrument})")
    print("=" * 50)
    print(f"Musical source: {instrument.musical_source}")
    print(f"Non-musical source: {instrument.nonmusical_source}")
    print()
    print(f"Musical audio collected: {status['musical_duration_hours']:.2f} / 1.50 hours")
    print(f"  Ready: {'Yes' if status['musical_ready'] else 'No'}")
    print(f"  Sources: {status['musical_sources_count']}")
    print()
    print(f"Non-musical audio collected: {status['nonmusical_duration_hours']:.2f} / 1.50 hours")
    print(f"  Ready: {'Yes' if status['nonmusical_ready'] else 'No'}")
    print(f"  Sources: {status['nonmusical_sources_count']}")
    print()
    print(f"READY FOR TRAINING: {'YES' if status['ready_for_training'] else 'NO'}")


def cmd_summary(args, db: SourceDatabase) -> None:
    """Show overall collection summary."""
    summary = db.get_collection_summary()

    print("\nMorphene Source Collection Summary")
    print("=" * 50)
    print(f"Total instruments: {summary['total_instruments']}")
    print(f"  Ready for training: {summary['ready_for_training']}")
    print(f"  In progress: {summary['in_progress']}")
    print(f"  Not started: {summary['not_started']}")
    print()
    print(f"Total sources: {summary['total_sources']}")
    print(f"Total audio: {summary['total_duration_hours']:.1f} hours")

    # Show per-tier breakdown
    print("\nBy Tier:")
    for tier_name, tier_range in [("Flagship", range(1, 11)),
                                   ("Strong", range(11, 31)),
                                   ("Experimental", range(31, 51))]:
        tier_instruments = [i for i in ALL_INSTRUMENTS if i.number in tier_range]
        ready = sum(1 for i in tier_instruments
                    if db.get_instrument_status(i.number)['ready_for_training'])
        print(f"  {tier_name}: {ready}/{len(tier_instruments)} ready")


def cmd_add_ref(args, db: SourceDatabase) -> None:
    """Add a reference clip."""
    source = db.get_source(args.source_id)
    if not source:
        print(f"Error: Source with ID {args.source_id} not found")
        sys.exit(1)

    clip_id = db.add_reference_clip(
        source_id=args.source_id,
        file_path=args.file,
        duration_sec=args.duration,
        description=args.description or ""
    )
    print(f"Added reference clip with ID {clip_id}")


def cmd_delete(args, db: SourceDatabase) -> None:
    """Delete a source."""
    source = db.get_source(args.id)
    if not source:
        print(f"Error: Source with ID {args.id} not found")
        sys.exit(1)

    if not args.yes:
        confirm = input(f"Delete source '{source.name}' (ID {args.id})? [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return

    db.delete_source(args.id)
    print(f"Deleted source {args.id}")


def main():
    parser = argparse.ArgumentParser(
        description='Morphene Source Tracking CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--db', default=None,
                        help='Path to database file (default: data/sources.db)')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new source')
    add_parser.add_argument('--instrument', '-i', type=int, required=True,
                            help='Instrument number (1-50)')
    add_parser.add_argument('--type', '-t', required=True,
                            choices=['musical', 'nonmusical'],
                            help='Source type')
    add_parser.add_argument('--name', '-n', required=True,
                            help='Source name')
    add_parser.add_argument('--origin', '-o', required=True,
                            choices=['recorded', 'licensed', 'commissioned', 'synthesized'],
                            help='Source origin')
    add_parser.add_argument('--license', '-l', help='License type')
    add_parser.add_argument('--restrictions', help='License restrictions')
    add_parser.add_argument('--duration', '-d', type=float,
                            help='Total duration in seconds')
    add_parser.add_argument('--quality', '-q', type=int, choices=[1, 2, 3, 4, 5],
                            help='Quality rating (1-5)')
    add_parser.add_argument('--files', '-f', help='Comma-separated file paths')
    add_parser.add_argument('--processing', '-p',
                            help='Comma-separated processing steps')
    add_parser.add_argument('--notes', help='Additional notes')

    # Update command
    update_parser = subparsers.add_parser('update', help='Update a source')
    update_parser.add_argument('id', type=int, help='Source ID')
    update_parser.add_argument('--name', '-n', help='Source name')
    update_parser.add_argument('--license', '-l', help='License type')
    update_parser.add_argument('--restrictions', help='License restrictions')
    update_parser.add_argument('--duration', '-d', type=float,
                               help='Total duration in seconds')
    update_parser.add_argument('--quality', '-q', type=int, choices=[1, 2, 3, 4, 5],
                               help='Quality rating (1-5)')
    update_parser.add_argument('--files', '-f', help='Comma-separated file paths')
    update_parser.add_argument('--processing', '-p',
                               help='Comma-separated processing steps')
    update_parser.add_argument('--notes', help='Additional notes')

    # List command
    list_parser = subparsers.add_parser('list', help='List sources')
    list_parser.add_argument('--instrument', '-i', type=int,
                             help='Filter by instrument number')

    # Show command
    show_parser = subparsers.add_parser('show', help='Show source details')
    show_parser.add_argument('id', type=int, help='Source ID')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show instrument status')
    status_parser.add_argument('instrument', type=int, help='Instrument number')

    # Summary command
    subparsers.add_parser('summary', help='Show collection summary')

    # Add reference clip command
    addref_parser = subparsers.add_parser('add-ref', help='Add a reference clip')
    addref_parser.add_argument('source_id', type=int, help='Source ID')
    addref_parser.add_argument('--file', '-f', required=True, help='File path')
    addref_parser.add_argument('--duration', '-d', type=float, required=True,
                               help='Duration in seconds')
    addref_parser.add_argument('--description', help='Description')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a source')
    delete_parser.add_argument('id', type=int, help='Source ID')
    delete_parser.add_argument('--yes', '-y', action='store_true',
                               help='Skip confirmation')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Get database
    db = get_database(args.db)

    # Dispatch to command handler
    commands = {
        'add': cmd_add,
        'update': cmd_update,
        'list': cmd_list,
        'show': cmd_show,
        'status': cmd_status,
        'summary': cmd_summary,
        'add-ref': cmd_add_ref,
        'delete': cmd_delete,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args, db)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
