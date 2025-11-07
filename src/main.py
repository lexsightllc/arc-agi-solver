import typer
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import mlflow

from src.utils.seed import set_deterministic_seed
from src.learning.trainer import CurriculumTrainer
from src.inference.predictor import ARCPredictor
from src.validation.preflight import PreflightValidator
from src.utils.logging import get_logger
from src.utils.visualization import visualize_program_execution

logger = get_logger(__name__)
app = typer.Typer()

@app.command()
def train(
    config_path: str = typer.Option("config/config.yaml", help="Path to the Hydra configuration file."),
    overrides: list[str] = typer.Option([], "-o", "--override", help="Hydra overrides (e.g., 'training.epochs=200').")
):
    """Train the ARC AGI solver using a curriculum learning approach."""
    # Initialize Hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=os.path.dirname(config_path), job_name="train_arc_solver")
    cfg = hydra.compose(config_name=os.path.basename(config_path), overrides=overrides)

    logger.info(f"Starting training with configuration:\n{OmegaConf.to_yaml(cfg)}")
    set_deterministic_seed(cfg.seed)

    with mlflow.start_run(experiment_id=mlflow.set_experiment(cfg.mlflow.experiment_name)) as run:
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        trainer: CurriculumTrainer = hydra.utils.instantiate(cfg.training.trainer, cfg=cfg)
        trainer.train()

        logger.info("Training complete. Model saved.")
        # Log best model artifact
        mlflow.log_artifact(os.path.join(cfg.output_dir, "best_model.pt"))

@app.command()
def evaluate(
    model_path: str = typer.Option(..., help="Path to the trained model checkpoint."),
    dataset_path: str = typer.Option(..., help="Path to the evaluation dataset (e.g., processed ARC tasks)."),
    config_path: str = typer.Option("config/config.yaml", help="Path to the Hydra configuration file."),
    overrides: list[str] = typer.Option([], "-o", "--override", help="Hydra overrides.")
):
    """Evaluate the trained ARC AGI solver on a given dataset."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=os.path.dirname(config_path), job_name="evaluate_arc_solver")
    cfg = hydra.compose(config_name=os.path.basename(config_path), overrides=overrides)

    logger.info(f"Starting evaluation with configuration:\n{OmegaConf.to_yaml(cfg)}")
    set_deterministic_seed(cfg.seed)

    with mlflow.start_run(experiment_id=mlflow.set_experiment(cfg.mlflow.experiment_name)) as run:
        mlflow.log_params({"evaluation_model_path": model_path, "evaluation_dataset_path": dataset_path})

        predictor: ARCPredictor = hydra.utils.instantiate(cfg.inference.predictor, cfg=cfg, model_path=model_path)
        # Load evaluation tasks from dataset_path
        # For simplicity, assume dataset_path points to a JSON file of ARC tasks
        import json
        with open(dataset_path, 'r') as f:
            evaluation_tasks = json.load(f)

        results = predictor.evaluate(evaluation_tasks)
        logger.info(f"Evaluation results: {results}")
        mlflow.log_metrics(results)

@app.command()
def predict(
    input_path: str = typer.Option(..., help="Path to the input ARC challenges JSON file."),
    output_path: str = typer.Option("submission.json", help="Path to save the submission JSON file."),
    model_path: str = typer.Option(..., help="Path to the trained model checkpoint."),
    config_path: str = typer.Option("config/config.yaml", help="Path to the Hydra configuration file."),
    overrides: list[str] = typer.Option([], "-o", "--override", help="Hydra overrides.")
):
    """Generate predictions for ARC tasks and create a submission.json file."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=os.path.dirname(config_path), job_name="predict_arc_solver")
    cfg = hydra.compose(config_name=os.path.basename(config_path), overrides=overrides)

    logger.info(f"Starting prediction with configuration:\n{OmegaConf.to_yaml(cfg)}")
    set_deterministic_seed(cfg.seed)

    predictor: ARCPredictor = hydra.utils.instantiate(cfg.inference.predictor, cfg=cfg, model_path=model_path)
    predictor.generate_submission(input_path, output_path)
    logger.info(f"Submission file generated at {output_path}")

@app.command()
def validate(
    submission_path: str = typer.Option(..., help="Path to the submission JSON file."),
    challenges_path: str = typer.Option(..., help="Path to the original ARC challenges JSON file."),
    config_path: str = typer.Option("config/config.yaml", help="Path to the Hydra configuration file."),
    overrides: list[str] = typer.Option([], "-o", "--override", help="Hydra overrides.")
):
    """Run preflight validation on a submission file."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=os.path.dirname(config_path), job_name="validate_submission")
    cfg = hydra.compose(config_name=os.path.basename(config_path), overrides=overrides)

    logger.info(f"Starting preflight validation with configuration:\n{OmegaConf.to_yaml(cfg)}")
    validator: PreflightValidator = hydra.utils.instantiate(cfg.validation.validator, cfg=cfg)
    is_valid = validator.validate_submission(submission_path, challenges_path)

    if is_valid:
        logger.info(f"Submission file '{submission_path}' passed all preflight checks.")
    else:
        logger.error(f"Submission file '{submission_path}' failed preflight checks. See logs for details.")
        raise typer.Exit(code=1)

@app.command()
def interpret(
    task_id: str = typer.Option(..., help="ID of the ARC task to interpret."),
    program_path: str = typer.Option(..., help="Path to the discovered program JSON file."),
    output_dir: str = typer.Option("interpret_output", help="Directory to save visualizations and logs."),
    config_path: str = typer.Option("config/config.yaml", help="Path to the Hydra configuration file."),
    overrides: list[str] = typer.Option([], "-o", "--override", help="Hydra overrides.")
):
    """Render discovered programs, visualize intermediate grids, and explain invariant firing."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=os.path.dirname(config_path), job_name="interpret_arc_solver")
    cfg = hydra.compose(config_name=os.path.basename(config_path), overrides=overrides)

    logger.info(f"Starting interpretation for task {task_id} with configuration:\n{OmegaConf.to_yaml(cfg)}")
    set_deterministic_seed(cfg.seed)

    # Load the program and task data
    import json
    from src.core.program import Program
    from src.core.grid import ARCGrid

    with open(program_path, 'r') as f:
        program_data = json.load(f)
    program = Program.from_json(program_data) # Assuming Program has a from_json method

    # Placeholder for loading the actual task data (input/output examples)
    # In a real scenario, this would load from the ARC dataset based on task_id
    # For demonstration, let's create a dummy task input
    dummy_input_grid = ARCGrid.from_array([[0,0,0],[0,1,0],[0,0,0]]) # Example 3x3 grid with center 1
    dummy_output_grid = ARCGrid.from_array([[1,1,1],[1,1,1],[1,1,1]]) # Example 3x3 grid all 1s
    task_examples = [{"input": dummy_input_grid, "output": dummy_output_grid}]

    os.makedirs(output_dir, exist_ok=True)

    visualize_program_execution(
        program=program,
        task_examples=task_examples,
        output_dir=output_dir,
        task_id=task_id,
        cfg=cfg.interpretability
    )
    logger.info(f"Interpretation results saved to {output_dir}")


if __name__ == "__main__":
    app()
