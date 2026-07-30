"""Step 4.1 entry point for compacting Step 4 subgraph pickle files into shards."""

from pathlib import Path
import sys


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from drug_disease_validation.src.subgraph_sharding import build_step04_1_parser, run_step04_1


def main() -> None:
    parser = build_step04_1_parser(_REPO_ROOT)
    args = parser.parse_args()
    run_step04_1(args)


if __name__ == "__main__":
    main()
