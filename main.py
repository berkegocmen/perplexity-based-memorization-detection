import logging
import argparse

from pipeline import ExperimentPipeline

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config file")
    args = parser.parse_args()
    pipeline = ExperimentPipeline(args.config)
    pipeline.run()
