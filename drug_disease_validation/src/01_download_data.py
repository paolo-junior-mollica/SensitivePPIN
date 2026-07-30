"""Step 1 entry point for downloading and preparing source data."""

from pathlib import Path
import sys


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from drug_disease_validation.src.source_data import build_step01_parser, run_step01


def main() -> None:
    parser = build_step01_parser(_REPO_ROOT)
    args = parser.parse_args()
    run_step01(args)


if __name__ == "__main__":
    main()
