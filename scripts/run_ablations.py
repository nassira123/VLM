import argparse

from src.utils.config import load_config, resolve_config
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    _ = resolve_config(load_config(args.config))
    logger = setup_logger()
    logger.warning("Ablation runner is scaffolded but not implemented yet. Implement after MVP.")


if __name__ == "__main__":
    main()
