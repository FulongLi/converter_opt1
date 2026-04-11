import sys

from .cli import main_pareto, main_train_ann


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("train-ann", "train_ann", "ann"):
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main_train_ann()
    else:
        if len(sys.argv) > 1 and sys.argv[1] in ("pareto", "opt"):
            sys.argv = [sys.argv[0]] + sys.argv[2:]
        main_pareto()


if __name__ == "__main__":
    main()
