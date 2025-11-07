import json
import os
import argparse
from omegaconf import OmegaConf
import hydra
from typing import List, Dict, Any

from src.core.dsl import DSL
from src.learning.dataset import SyntheticARCProblemGenerator
from src.utils.logging import get_logger
from src.utils.seed import set_deterministic_seed

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ARC-like tasks.")
    parser.add_argument("--output_path", type=str, default="data/synthetic/synthetic_tasks.json", help="Path to save the generated synthetic tasks.")
    parser.add_argument("--config_path", type=str, default="config/config.yaml", help="Path to the Hydra configuration file for training settings.")
    parser.add_argument("--num_tasks", type=int, default=5000, help="Number of synthetic tasks to generate.")
    args = parser.parse_args()

    # Initialize Hydra to load configuration
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=os.path.dirname(args.config_path), job_name="generate_synthetic_data")
    cfg = hydra.compose(config_name=os.path.basename(args.config_path))

    set_deterministic_seed(cfg.seed)

    dsl = DSL()
    synthetic_generator = hydra.utils.instantiate(cfg.training.synthetic_generator, dsl=dsl)

    logger.info(f"Generating {args.num_tasks} synthetic tasks...")
    generated_tasks = []
    for i in range(args.num_tasks):
        # Use the last stage's complexity and primitives for general synthetic data generation
        last_stage = cfg.training.curriculum.stages[-1]
        task = synthetic_generator.generate_task(
            complexity=last_stage.task_complexity_max,
            primitives_subset=last_stage.primitives_subset
        )
        generated_tasks.append(task)
        if (i + 1) % 1000 == 0:
            logger.info(f"Generated {i+1} tasks.")

    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, 'w') as f:
        json.dump(generated_tasks, f, indent=4)
    logger.info(f"Generated {len(generated_tasks)} synthetic tasks saved to {args.output_path}")

if __name__ == "__main__":
    main()
