#!/usr/bin/env python3
"""
Wrapper script for backward compatibility.

The actual implementation is in ubf_protein/cli_predict.py.
Run predictions using:
  python test_protein.py --pdb 1UBQ
  python test_protein.py --sequence ACDEFGH...

Or use the package CLI:
  python -m ubf_protein.cli_predict --pdb 1UBQ
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the actual CLI
from ubf_protein.cli_predict import main

if __name__ == "__main__":
    main()
