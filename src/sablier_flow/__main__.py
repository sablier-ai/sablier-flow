"""Allow ``python -m sablier_flow`` to invoke the CLI."""
from __future__ import annotations

import sys

from sablier_flow.cli import main

if __name__ == "__main__":
    sys.exit(main())
