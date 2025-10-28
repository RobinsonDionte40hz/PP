#!/usr/bin/env python3
"""
Run a test campaign with 10 proteins

This script runs the large-scale validation campaign on 10 selected proteins
with full QCPP-UBF integration.

Usage:
    python run_10_protein_test.py
    
Or with custom config:
    python run_10_protein_test.py --config validation/configs/test_10_proteins.json
"""

import sys
import argparse
from pathlib import Path

# Add validation to path
sys.path.insert(0, str(Path(__file__).parent / "validation"))

from validation.run_validation_campaign import main as run_campaign


def main():
    """Run 10-protein test campaign."""
    
    parser = argparse.ArgumentParser(
        description='Run validation campaign on 10 proteins',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='validation/configs/test_10_proteins.json',
        help='Path to campaign configuration file'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode (approve each phase)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous checkpoint'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("10-PROTEIN VALIDATION CAMPAIGN")
    print("="*70)
    print(f"\nConfiguration: {args.config}")
    print(f"Interactive mode: {args.interactive}")
    print(f"Resume: {args.resume}")
    print()
    
    # Build arguments for campaign runner
    campaign_args = [
        '--config', args.config,
    ]
    
    if args.interactive:
        campaign_args.append('--interactive')
    
    if args.resume:
        campaign_args.append('--resume')
    
    # Run the campaign
    sys.argv = ['run_validation_campaign.py'] + campaign_args
    
    try:
        return run_campaign()
    except KeyboardInterrupt:
        print("\n\n⚠️  Campaign interrupted by user")
        print("You can resume later with: python run_10_protein_test.py --resume")
        return 1
    except Exception as e:
        print(f"\n\n❌ Campaign failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
