#!/usr/bin/env python3
"""
Morphene CLI entry point.

Usage:
    python -m morphene <command> [options]

Commands:
    instruments     List instrument definitions
    sources         Manage source tracking database
    queue           Manage training queue
    audit           Run checkpoint audits
    setup           Set up project structure
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Morphene Instrument Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  instruments   List and query instrument definitions
  sources       Manage audio source tracking (add, list, status)
  queue         Manage training queue (add, list, start, cancel)
  audit         Run checkpoint quality audits
  setup         Set up project folder structure

Examples:
  python -m morphene instruments --list
  python -m morphene sources add -i 10 -t musical -n "Violin" -o licensed
  python -m morphene queue add -i 10 --db-path /data/pyroviolin --output /models
  python -m morphene audit --run /path/to/run --all --instrument pyroviolin
  python -m morphene setup --poc /path/to/project
        """
    )

    parser.add_argument('command', nargs='?',
                        choices=['instruments', 'sources', 'queue', 'audit', 'setup'],
                        help='Command to run')
    parser.add_argument('--version', action='store_true',
                        help='Show version')

    # Parse just the first argument to determine command
    args, remaining = parser.parse_known_args()

    if args.version:
        from morphene import __version__
        print(f"morphene {__version__}")
        return

    if not args.command:
        parser.print_help()
        return

    # Dispatch to appropriate CLI
    if args.command == 'instruments':
        from morphene.instruments import list_instruments, get_instrument, Tier

        # Simple instrument listing
        sub_parser = argparse.ArgumentParser(description='List instruments')
        sub_parser.add_argument('--list', '-l', action='store_true',
                                help='List all instruments')
        sub_parser.add_argument('--tier', '-t',
                                choices=['flagship', 'strong', 'experimental'],
                                help='Filter by tier')
        sub_parser.add_argument('--show', '-s', type=int,
                                help='Show details for instrument number')

        sub_args = sub_parser.parse_args(remaining)

        if sub_args.show:
            inst = get_instrument(sub_args.show)
            if inst:
                print(f"\n{inst.display_name}")
                print("=" * 40)
                print(f"Tier: {inst.tier.name}")
                print(f"Musical: {inst.musical_source}")
                print(f"  {inst.musical_source_description or 'No details'}")
                print(f"Non-musical: {inst.nonmusical_source}")
                print(f"  {inst.nonmusical_source_description or 'No details'}")
                if inst.notes:
                    print(f"Notes: {inst.notes}")
            else:
                print(f"Instrument {sub_args.show} not found")
        else:
            tier = None
            if sub_args.tier:
                tier_map = {
                    'flagship': Tier.FLAGSHIP,
                    'strong': Tier.STRONG,
                    'experimental': Tier.EXPERIMENTAL,
                }
                tier = tier_map[sub_args.tier]
            list_instruments(tier)

    elif args.command == 'sources':
        # Run sources CLI with remaining args
        sys.argv = ['sources_cli'] + remaining
        from morphene.scripts import sources_cli
        sources_cli.main()

    elif args.command == 'queue':
        # Run queue CLI with remaining args
        sys.argv = ['queue_cli'] + remaining
        from morphene.scripts import queue_cli
        queue_cli.main()

    elif args.command == 'audit':
        # Run audit script with remaining args
        sys.argv = ['audit'] + remaining
        from morphene.scripts import audit
        audit.main()

    elif args.command == 'setup':
        # Run setup script with remaining args
        sys.argv = ['setup_project'] + remaining
        from morphene.scripts import setup_project
        setup_project.main()


if __name__ == '__main__':
    main()
