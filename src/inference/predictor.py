import torch
import json
import time
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig

from src.core.grid import ARCGrid
from src.core.dsl import DSL, Program
from src.learning.models import PolicyNet, ValueNet
from src.learning.prior import ProgramPrior
from src.solver.verifier import ProgramVerifier
from src.solver.constraints import ConstraintSolver
from src.solver.repair import ProgramRepairer
from src.solver.portfolio import SolverPortfolio
from src.utils.logging import get_logger
from src.utils.metrics import evaluate_predictions

logger = get_logger(__name__)

class ARCPredictor:
    """Orchestrates the inference process, including Attempt 1 and Attempt 2 predictions."""
    def __init__(self, cfg: DictConfig, model_path: str):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dsl = DSL()

        # Initialize models and components
        self.policy_net: PolicyNet = hydra.utils.instantiate(cfg.model.policy_net).to(self.device)
        self.value_net: ValueNet = hydra.utils.instantiate(cfg.model.value_net).to(self.device)
        self.program_prior: ProgramPrior = hydra.utils.instantiate(cfg.model.program_prior, dsl=self.dsl)

        # Load trained weights
        self._load_models(model_path)

        self.verifier = hydra.utils.instantiate(cfg.solver.verifier)
        self.constraint_solver = hydra.utils.instantiate(cfg.solver.constraints)
        self.repairer = hydra.utils.instantiate(cfg.solver.repair, dsl=self.dsl, verifier=self.verifier, constraint_solver=self.constraint_solver)
        self.solver_portfolio = SolverPortfolio(
            dsl=self.dsl,
            policy_net=self.policy_net,
            value_net=self.value_net,
            program_prior=self.program_prior,
            verifier=self.verifier,
            constraint_solver=self.constraint_solver,
            repairer=self.repairer,
            cfg=cfg.solver
        )

    def _load_models(self, model_path: str):
        """Loads trained model weights."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.program_prior.load_state_dict(checkpoint['program_prior_state_dict'])
        self.policy_net.eval()
        self.value_net.eval()
        logger.info(f"Models loaded from {model_path}")

    def predict_single_task(self, task_id: str, input_grid: ARCGrid, test_examples: List[Dict[str, Any]]) -> Dict[str, List[List[int]]]:
        """Generates two predictions for a single ARC task (input_grid, test_examples)."""
        predictions = {}
        compute_budget_ms = self.cfg.compute_budget.inference_time_per_task_ms
        node_expansion_budget = self.cfg.compute_budget.node_expansion_budget

        # Attempt 1: Deterministic and Conservative
        logger.info(f"  Task {task_id}: Starting Attempt 1 (Deterministic)")
        start_time_attempt1 = time.time()
        attempt1_program = self.solver_portfolio.solve_task(
            task_id, input_grid, test_examples[0]["output"], # Use first test output as target for search
            compute_budget_ms // 2, # Allocate half budget for attempt 1
            node_expansion_budget // 2,
            self.cfg.solver.inference_strategy.attempt1
        )
        time_taken_attempt1 = (time.time() - start_time_attempt1) * 1000
        logger.info(f"  Task {task_id}: Attempt 1 finished in {time_taken_attempt1:.2f}ms.")

        attempt1_output_grid = None
        if attempt1_program:
            try:
                attempt1_output_grid = attempt1_program.execute(input_grid)
            except Exception as e:
                logger.error(f"Error executing Attempt 1 program for task {task_id}: {e}")
        
        if attempt1_output_grid is None or not isinstance(attempt1_output_grid, ARCGrid):
            attempt1_output_grid = self.solver_portfolio.get_fallback_program(task_id, input_grid, self.cfg.compute_budget).execute(input_grid)
            logger.warning(f"  Task {task_id}: Attempt 1 failed or returned non-grid, used fallback.")

        predictions["attempt1"] = attempt1_output_grid.to_array()

        # Attempt 2: Diverse yet Principled
        logger.info(f"  Task {task_id}: Starting Attempt 2 (Diverse)")
        start_time_attempt2 = time.time()
        attempt2_program = self.solver_portfolio.solve_task(
            task_id, input_grid, test_examples[0]["output"], # Use first test output as target for search
            compute_budget_ms // 2, # Allocate remaining half budget for attempt 2
            node_expansion_budget // 2,
            self.cfg.solver.inference_strategy.attempt2
        )
        time_taken_attempt2 = (time.time() - start_time_attempt2) * 1000
        logger.info(f"  Task {task_id}: Attempt 2 finished in {time_taken_attempt2:.2f}ms.")

        attempt2_output_grid = None
        if attempt2_program:
            try:
                attempt2_output_grid = attempt2_program.execute(input_grid)
            except Exception as e:
                logger.error(f"Error executing Attempt 2 program for task {task_id}: {e}")

        if attempt2_output_grid is None or not isinstance(attempt2_output_grid, ARCGrid):
            attempt2_output_grid = self.solver_portfolio.get_fallback_program(task_id, input_grid, self.cfg.compute_budget).execute(input_grid)
            logger.warning(f"  Task {task_id}: Attempt 2 failed or returned non-grid, used fallback.")

        predictions["attempt2"] = attempt2_output_grid.to_array()

        return predictions

    def generate_submission(self, input_path: str, output_path: str):
        """Generates the submission.json file for all tasks in the input_path."""
        from src.inference.submission import SubmissionGenerator
        submission_generator = SubmissionGenerator(self)
        submission_generator.generate(input_path, output_path)

    def evaluate(self, evaluation_tasks: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates the predictor on a set of tasks and returns metrics."""
        all_predictions = {}
        runtimes = []

        for task_id, task_data in evaluation_tasks.items():
            logger.info(f"Processing evaluation task: {task_id}")
            input_grid = ARCGrid.from_array(task_data["train"][0]["input"]) # Use first train input as base
            test_examples = task_data["test"]

            task_start_time = time.time()
            task_predictions = {}
            for i, test_example in enumerate(test_examples):
                # For evaluation, we need to predict for each test input
                # The current predict_single_task assumes one input/output pair for search target.
                # This needs to be adapted for multiple test inputs.
                # For simplicity, we'll use the first test input's output as the target for search.
                # A more robust evaluation would run search for each test input independently.
                logger.debug(f"  Predicting for test example {i} of task {task_id}")
                predictions_for_test_input = self.predict_single_task(task_id, ARCGrid.from_array(test_example["input"]), [test_example])
                task_predictions[f"output_{i}"] = predictions_for_test_input
            
            all_predictions[task_id] = task_predictions
            runtimes.append(time.time() - task_start_time)

        metrics = evaluate_predictions(all_predictions, evaluation_tasks)
        metrics["avg_runtime_per_task_s"] = np.mean(runtimes).item()
        metrics["max_runtime_per_task_s"] = np.max(runtimes).item()
        return metrics
