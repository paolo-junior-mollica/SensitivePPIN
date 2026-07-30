from pathlib import Path
import sys


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from drug_disease_validation.src.source_data import build_step01_parser, run_step01  # noqa: E402


def main() -> None:
    parser = build_step01_parser(_REPO_ROOT)
    args = parser.parse_args()
    run_step01(args)


if __name__ == "__main__":
    main()
