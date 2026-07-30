"""Step 4 entry point for building pathway-based BioGRID subgraphs."""

from pathlib import Path
import sys


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from drug_disease_validation.src.subgraph_building import build_step04_parser, run_step04


def main() -> None:
    parser = build_step04_parser(_REPO_ROOT)
    args = parser.parse_args()
    run_step04(args)


if __name__ == "__main__":
    main()
