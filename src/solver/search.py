# SPDX-License-Identifier: MPL-2.0
import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

from src.core.grid import ARCGrid
from src.core.dsl import DSL, Program, ProgramStep, Primitive
from src.core.ir import ARCIntermediateRepresentation
from src.learning.models import PolicyNet, ValueNet
from src.learning.prior import ProgramPrior
from src.solver.verifier import ProgramVerifier
from src.solver.constraints import ConstraintSolver
from src.solver.utils import generate_random_args
from src.utils.logging import get_logger

logger = get_logger(__name__)

class SearchNode:
    """Represents a node in the MCTS search tree."""
    def __init__(self, grid: ARCGrid, program: Program, parent: Optional['SearchNode'] = None, action: Optional[Tuple[Primitive, Dict[str, Any]]] = None):
        self.grid = grid
        self.program = program # Program leading to this state
        self.parent = parent
        self.action = action # (primitive, args) that led to this node
        self.children: List['SearchNode'] = []
        self.visits = 0
        self.value_sum = 0.0 # Sum of rewards from simulations through this node
        self.policy_probs: Optional[Dict[Tuple[Primitive, Tuple], float]] = None # Policy probabilities for child actions
        self.is_terminal = False
        self.is_solved = False

    def ucb_score(self, c_puct: float, total_visits: int) -> float:
        if self.visits == 0: return float('inf') # Prioritize unvisited nodes
        q_value = self.value_sum / self.visits
        # Add exploration term based on parent's policy prediction (if available)
        # For simplicity, using a generic exploration term here.
        exploration_term = c_puct * np.sqrt(np.log(total_visits) / self.visits)
        return q_value + exploration_term

    def add_child(self, child_node: 'SearchNode'):
        self.children.append(child_node)

    def get_program_length(self) -> int:
        return len(self.program)


class MCTSSearch:
    """Monte Carlo Tree Search for ARC program induction."""
    def __init__(self, dsl: DSL, policy_net: PolicyNet, value_net: ValueNet, program_prior: ProgramPrior,
                 verifier: ProgramVerifier, constraint_solver: ConstraintSolver, cfg: DictConfig):
        self.dsl = dsl
        self.policy_net = policy_net
        self.value_net = value_net
        self.program_prior = program_prior
        self.verifier = verifier
        self.constraint_solver = constraint_solver
        self.cfg = cfg

        self.num_simulations = cfg.num_simulations
        self.c_puct = cfg.c_puct
        self.temperature = cfg.temperature
        self.max_depth = cfg.max_depth
        self.rollout_budget = cfg.rollout_budget
        self.mdl_weight = cfg.mdl_weight
        self.device = policy_net.device # Assuming policy_net has a device attribute

        self.all_primitives = list(self.dsl.get_all_primitives().values())
        self.primitive_names = self.dsl.get_primitive_names()

    def _grid_to_tensor(self, grid: ARCGrid) -> torch.Tensor:
        """Converts an ARCGrid to a one-hot encoded tensor for model input."""
        num_colors = 10 # ARC colors 0-9
        grid_array = grid.grid
        one_hot = np.eye(num_colors)[grid_array]
        # Add batch dimension and permute to (B, C, H, W)
        return torch.from_numpy(one_hot).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

    def _expand_node(self, node: SearchNode, task_input_grid: ARCGrid, task_output_grid: ARCGrid) -> None:
        """Expands a node by generating possible next actions (primitives + args)."""
        if node.get_program_length() >= self.max_depth: # Max program length reached
            node.is_terminal = True
            return

        # Get policy predictions from neural network
        input_tensor = self._grid_to_tensor(node.grid)
        policy_logits, arg_logits = self.policy_net(input_tensor)

        # Convert logits to probabilities and sample actions
        primitive_probs = F.softmax(policy_logits / self.temperature, dim=-1).squeeze(0)
        # For simplicity, we'll just take top-k primitives and generate random args for now.
        # A more sophisticated approach would involve arg_logits to guide arg sampling.
        top_k_primitives_indices = torch.topk(primitive_probs, k=min(len(self.all_primitives), 5)).indices.tolist()

        possible_actions = []
        for idx in top_k_primitives_indices:
            primitive_cls = self.all_primitives[idx]
            primitive_instance = primitive_cls()
            # Generate arguments (this is a critical and complex part)
            # For now, generate random valid arguments. In a real system, arg_logits would guide this.
            args = generate_random_args(primitive_instance, node.grid, self.primitive_names)
            possible_actions.append((primitive_instance, args))

        node.policy_probs = {}
        for primitive, args in possible_actions:
            # Execute action to get next state
            try:
                next_grid_output = primitive.apply(node.grid, **args)
                if not isinstance(next_grid_output, ARCGrid): # Handle non-grid outputs like CountColor
                    # If it's the last step and it's a value, it's a terminal state.
                    # For ARC, final output is always a grid, so this is an intermediate value.
                    # For now, we'll skip non-grid intermediate outputs.
                    continue

                next_program_steps = node.program.steps + [ProgramStep(primitive, args)]
                next_program = Program(next_program_steps, self.dsl)

                # Check constraints (e.g., if the program is already violating invariants)
                if not self.constraint_solver.check_program_invariants(next_program, task_input_grid, task_output_grid):
                    continue # Skip invalid programs

                child_node = SearchNode(next_grid_output, next_program, parent=node, action=(primitive, args))
                node.add_child(child_node)
                # Store policy probability for this action (simplified)
                node.policy_probs[(primitive, tuple(sorted(args.items())))] = primitive_probs[self.primitive_names.index(primitive.name)].item()
            except Exception as e:
                logger.debug(f"Error applying primitive {primitive.name} with args {args}: {e}")
                continue

        if not node.children: # If no valid children, it's a terminal node
            node.is_terminal = True

    def _simulate_rollout(self, node: SearchNode, task_input_grid: ARCGrid, task_output_grid: ARCGrid) -> float:
        """Performs a random rollout from the current node to estimate value."""
        current_grid = node.grid.copy()
        current_program_steps = list(node.program.steps) # Copy steps
        reward = 0.0

        for _ in range(self.rollout_budget):
            if len(current_program_steps) >= self.max_depth: break

            # Randomly pick a primitive and generate random args
            primitive_cls = np.random.choice(self.all_primitives)
            primitive_instance = primitive_cls()
            args = generate_random_args(primitive_instance, current_grid, self.primitive_names)

            try:
                next_grid_output = primitive_instance.apply(current_grid, **args)
                if not isinstance(next_grid_output, ARCGrid): continue

                current_program_steps.append(ProgramStep(primitive_instance, args))
                current_grid = next_grid_output

                # Check for solution during rollout (early exit)
                if self.verifier.verify(current_grid, task_output_grid):
                    reward = 1.0 # Solved
                    break
            except Exception:
                pass # Ignore errors during rollout

        # If not solved, assign reward based on similarity or value network prediction
        if reward == 0.0:
            # Use value network for a more informed rollout termination reward
            value_tensor = self.value_net(self._grid_to_tensor(current_grid))
            reward = torch.sigmoid(value_tensor).item() # Convert logit to probability

        # Incorporate MDL as a penalty
        final_program = Program(current_program_steps, self.dsl)
        mdl_cost = final_program.calculate_mdl()
        reward -= self.mdl_weight * mdl_cost

        return reward

    def _backpropagate(self, node: SearchNode, reward: float) -> None:
        """Updates visit counts and value sums up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.value_sum += reward
            current = current.parent

    def search(self, task_input_grid: ARCGrid, task_output_grid: ARCGrid, compute_budget_ms: int, node_expansion_budget: int) -> Optional[Program]:
        """Performs MCTS to find a program."""
        start_time = time.time()
        root = SearchNode(task_input_grid, Program([], self.dsl))
        best_program_found: Optional[Program] = None
        best_mdl = float('inf')
        best_reward = -float('inf')

        num_expansions = 0

        for i in range(self.num_simulations):
            if (time.time() - start_time) * 1000 > compute_budget_ms or num_expansions >= node_expansion_budget:
                logger.info(f"MCTS budget exhausted after {i} simulations.")
                break

            # Selection: Traverse tree using UCB to find a leaf node
            node = root
            path = [node]
            while node.children and not node.is_terminal:
                # Select child with highest UCB score
                total_visits = sum(child.visits for child in node.children)
                if total_visits == 0: # All children unvisited, pick one randomly
                    node = np.random.choice(node.children)
                else:
                    node = max(node.children, key=lambda child: child.ucb_score(self.c_puct, total_visits))
                path.append(node)

            # Expansion: Expand the selected leaf node
            if not node.is_terminal:
                self._expand_node(node, task_input_grid, task_output_grid)
                num_expansions += 1
                # If node has children after expansion, pick one to simulate from
                if node.children:
                    node = np.random.choice(node.children) # Pick a new child for simulation
                    path.append(node)

            # Simulation: Rollout from the expanded node (or selected leaf if not expanded)
            reward = self._simulate_rollout(node, task_input_grid, task_output_grid)

            # Backpropagation: Update values and visits up the path
            self._backpropagate(node, reward)

            # Check if current node's program is a solution and update best_program_found
            if self.verifier.verify(node.grid, task_output_grid):
                node.is_solved = True
                current_mdl = node.program.calculate_mdl()
                # Incorporate learned prior for tie-breaking
                # For simplicity, assume program_prior.get_program_log_likelihood needs an embedding
                # Here, we'll use a dummy embedding or just rely on MDL for now.
                prior_log_likelihood = self.program_prior.get_program_log_likelihood(node.program, program_embedding=None)
                # Combine MDL and prior for a scoring function
                program_score = -current_mdl + prior_log_likelihood # Higher is better

                if best_program_found is None or program_score > best_reward:
                    best_program_found = node.program
                    best_mdl = current_mdl
                    best_reward = program_score
                    logger.debug(f"Found new best program (MDL: {best_mdl:.2f}, Prior: {prior_log_likelihood:.2f}) at simulation {i}")

        if best_program_found is None:
            logger.warning("MCTS completed, but no verified program found. Returning best program by value if any.")
            # Fallback: if no verified program, return the program from the node with highest average value
            # This requires traversing the tree to find the best node.
            # For simplicity, we'll just return None if no verified solution.
            # A more robust fallback would be to return the program from the node with the highest Q-value
            # that also satisfies basic constraints.
            return None

        logger.info(f"MCTS finished. Best program found with MDL: {best_mdl:.2f}")
        return best_program_found
