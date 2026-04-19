import importlib
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <experiment> [args...]")
        print("Example: python main.py train_gpt --lr 3e-4 --batch_size 32")
        sys.exit(1)

    experiment = sys.argv[1]
    sys.argv = [experiment + ".py"] + sys.argv[2:]

    try:
        module = importlib.import_module(f"experiments.{experiment}")
    except ModuleNotFoundError:
        print(f"Error: no training script found for experiment '{experiment}'")
        print(f"Expected: experiments/{experiment}.py")
        sys.exit(1)

    if not hasattr(module, "main"):
        print(f"Error: experiments/{experiment}.py has no main() function")
        sys.exit(1)

    module.main()


if __name__ == "__main__":
    main()
