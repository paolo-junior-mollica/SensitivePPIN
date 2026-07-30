"""Step 3 entry point for mapping protein pairs to Reactome pathways."""

from pathlib import Path
import sys


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from drug_disease_validation.src.mapping_pairs import build_step03_parser, run_step03


def main() -> None:
    parser = build_step03_parser(_REPO_ROOT)
    args = parser.parse_args()
    run_step03(args)


if __name__ == "__main__":
    main()
