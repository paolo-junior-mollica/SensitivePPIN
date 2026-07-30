"""Step 2 entry point for building positive drug-target/disease-protein pairs."""

from pathlib import Path
import sys


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from drug_disease_validation.src.pair_building import build_step02_parser, run_step02


def main() -> None:
    parser = build_step02_parser(_REPO_ROOT)
    args = parser.parse_args()
    run_step02(args)


if __name__ == "__main__":
    main()
