"""Launch a native-editor job with the project's engine on the import path."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tonehound.native_bridge import main

if __name__ == '__main__':
    raise SystemExit(main())
