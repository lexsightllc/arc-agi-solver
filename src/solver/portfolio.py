import time
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig

from src.core.grid import ARCGrid
from src.core.dsl import DSL, Program
from src.learning.models import PolicyNet, ValueNet
from src.learning.prior import ProgramPrior
from src.solver.search import MCTSSearch
from src.solver.verifier import ProgramVerifier
from src.solver.constraints import ConstraintSolver
from src.solver.repair import ProgramRepairer
from src.utils.logging import get_logger

logger = get_logger(__name__)

class SolverPortfolio:
    """Manages a portfolio of different search strategies and heuristics."""
    def __init__(self, dsl: DSL, policy_net: PolicyNet, value_net: ValueNet, program_prior: ProgramPrior,
                 verifier: ProgramVerifier, constraint_solver: ConstraintSolver, repairer: ProgramRepairer, cfg: DictConfig):
        self.dsl = dsl
        self.policy_net = policy_net
        self.value_net = value_net
        self.program_prior = program_prior
        self.verifier = verifier
        self.constraint_solver = constraint_solver
        self.repairer = repairer
        self.cfg = cfg

        self.solvers: Dict[str, MCTSSearch] = {}
        self._initialize_solvers()

    def _initialize_solvers(self):
        """Initializes different MCTS instances based on configuration."""
        for solver_cfg in self.cfg.solvers:
            # Create a specific MCTS instance for each solver strategy
            # This allows different search parameters, policy guidance, etc.
            # For simplicity, we'll reuse the main policy/value/prior but could swap them.
            search_params = self.cfg.search.copy()
            if solver_cfg.get("policy_guided", True) == False:
                # Disable policy guidance for this solver (e.g., purely random rollout)
                search_params.num_simulations = search_params.num_simulations // 2 # Less simulations for random
                search_params.c_puct = 2.0 # More exploration

            # Instantiate a new MCTS searcher with potentially modified parameters
            solver_instance = MCTSSearch(
                dsl=self.dsl,
                policy_net=self.policy_net,
                value_net=self.value_net,
                program_prior=self.program_prior,
                verifier=self.verifier,
                constraint_solver=self.constraint_solver,
                cfg=search_params # Pass the specific search config
            )
            self.solvers[solver_cfg.name] = solver_instance
            logger.info(f"Initialized solver: {solver_cfg.name}")

    def solve_task(self, task_id: str, input_grid: ARCGrid, output_grid: ARCGrid, compute_budget_ms: int, node_expansion_budget: int, strategy_cfg: DictConfig) -> Optional[Program]:
        """Attempts to solve a single ARC task using a specified strategy."""
        solver_name = strategy_cfg.solver_name
        solver = self.solvers.get(solver_name)
        if not solver:
            logger.error(f"Solver '{solver_name}' not found in portfolio.")
            return None

        logger.info(f"Solving task {task_id} with strategy '{solver_name}'. Budget: {compute_budget_ms}ms, {node_expansion_budget} nodes.")
        start_time = time.time()

        # Adjust solver parameters based on strategy_cfg (e.g., temperature, random restarts)
        original_temperature = solver.temperature
        original_c_puct = solver.c_puct
        solver.temperature = strategy_cfg.get("temperature", original_temperature)
        solver.c_puct = strategy_cfg.get("c_puct", original_c_puct)

        best_program = None
        num_restarts = strategy_cfg.get("random_restarts", 0) + 1

        for restart_idx in range(num_restarts):
            remaining_budget_ms = compute_budget_ms - int((time.time() - start_time) * 1000)
            if remaining_budget_ms <= 0: break

            logger.debug(f"  Attempt {restart_idx+1}/{num_restarts} for solver '{solver_name}'. Remaining budget: {remaining_budget_ms}ms")
            program = solver.search(input_grid, output_grid, remaining_budget_ms, node_expansion_budget // num_restarts)

            if program:
                # Attempt repair if program is found but not fully verified (e.g., if constraints were relaxed)
                if not self.verifier.verify_program_on_examples(program, [
                    {"input": input_grid, "output": output_grid}
                ]):
                    logger.debug(f"  Program found but failed strict verification. Attempting repair.")
                    repaired_program = self.repairer.repair_program(program, input_grid, output_grid)
                    if repaired_program:
                        program = repaired_program
                        logger.debug("  Program successfully repaired.")

                if self.verifier.verify_program_on_examples(program, [
                    {"input": input_grid, "output": output_grid}
                ]):
                    best_program = program
                    logger.info(f"  Solver '{solver_name}' found a verified program on attempt {restart_idx+1}.")
                    break # Found a solution, no need for more restarts

        # Reset solver parameters
        solver.temperature = original_temperature
        solver.c_puct = original_c_puct

        return best_program

    def get_fallback_program(self, task_id: str, input_grid: ARCGrid, cfg_compute_budget: DictConfig) -> Optional[Program]:
        """Returns a fallback program if the budget is exhausted or no solution is found.
        Options: return the best partial program, or an empty grid.
        """
        fallback_strategy = cfg_compute_budget.fallback_strategy
        if fallback_strategy == "return_best_partial":
            # This would require the search algorithm to keep track of the 'best partial' program
            # (e.g., one that matches most pixels, or satisfies most constraints).
            # For now, we'll return a simple identity program or a default.
            logger.warning(f"Budget exhausted for task {task_id}. Returning identity program as fallback.")
            # A simple identity program: copy input to output
            from src.core.dsl import Copy, ProgramStep
            identity_program = Program([ProgramStep(Copy(), {"source_grid": input_grid, "r_offset": 0, "c_offset": 0})], self.dsl)
            return identity_program
        elif fallback_strategy == "return_empty_grid":
            logger.warning(f"Budget exhausted for task {task_id}. Returning empty grid as fallback.")
            return Program([], self.dsl) # Program that produces an empty grid (or a grid of background color)
        else:
            logger.error(f"Unknown fallback strategy: {fallback_strategy}")
            return None
