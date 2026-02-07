import argparse

from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logger = setup_logger()
    logger.warning("Report generation is scaffolded but not implemented yet. Implement after MVP.")


if __name__ == "__main__":
    main()
