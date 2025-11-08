# SPDX-License-Identifier: MPL-2.0
import pytest
import torch
from omegaconf import OmegaConf

from src.core.grid import ARCGrid
from src.core.dsl import DSL, Program, ProgramStep, Paint, Rotate, ReflectHorizontal
from src.learning.models import PolicyNet, ValueNet, ProgramPriorModel
from src.learning.prior import ProgramPrior
from src.solver.verifier import ProgramVerifier
from src.solver.constraints import ConstraintSolver
from src.solver.repair import ProgramRepairer
from src.solver.portfolio import SolverPortfolio
from src.solver.search import MCTSSearch
from src.utils.seed import set_deterministic_seed

@pytest.fixture(scope="module")
def setup_solver_components():
    set_deterministic_seed(42)
    device = torch.device("cpu")
    dsl = DSL()

    # Mock models
    policy_net = PolicyNet(input_channels=10, hidden_dim=64, num_layers=2, output_dim=len(dsl.get_primitive_names())).to(device)
    value_net = ValueNet(input_channels=10, hidden_dim=64, num_layers=2, output_dim=1).to(device)
    program_prior_model = ProgramPriorModel(input_dim=64, hidden_dim=32).to(device)
    program_prior = ProgramPrior(dsl, prior_type="neural", neural_prior_model=program_prior_model)

    # Mock config for solver components
    cfg_solver = OmegaConf.create({
        "verifier": {"strict_equality": True},
        "constraints": {
            "enable_color_invariants": True,
            "enable_shape_invariants": True,
            "enable_cardinality_invariants": True,
            "z3_timeout_ms": 100
        },
        "repair": {
            "max_repair_attempts": 2,
            "edit_types": ["argument_change"],
            "repair_budget_per_attempt_ms": 100
        },
        "search": {
            "num_simulations": 50,
            "c_puct": 1.0,
            "temperature": 0.1,
            "max_depth": 5,
            "rollout_budget": 10,
            "mdl_weight": 0.01
        },
        "solvers": [
            {"name": "mcts_learned_prior", "strategy": "mcts", "policy_guided": True, "prior_guided": True, "mdl_enabled": True, "constraints_enabled": True},
            {"name": "mcts_symmetry_focused", "strategy": "mcts", "policy_guided": False, "prior_guided": False, "mdl_enabled": True, "constraints_enabled": True}
        ],
        "inference_strategy": {
            "attempt1": {"solver_name": "mcts_learned_prior", "temperature": 0.01, "random_restarts": 0, "strict_constraints": True},
            "attempt2": {"solver_name": "mcts_symmetry_focused", "temperature": 0.5, "random_restarts": 1, "strict_constraints": False}
        }
    })

    verifier = ProgramVerifier(strict_equality=cfg_solver.verifier.strict_equality)
    constraint_solver = ConstraintSolver(**cfg_solver.constraints)
    repairer = ProgramRepairer(dsl, verifier, constraint_solver, **cfg_solver.repair)

    return dsl, policy_net, value_net, program_prior, verifier, constraint_solver, repairer, cfg_solver, device

def test_mcts_solves_simple_task(setup_solver_components):
    dsl, policy_net, value_net, program_prior, verifier, constraint_solver, repairer, cfg_solver, device = setup_solver_components

    # Define a simple task: change one pixel color
    input_grid = ARCGrid.from_array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    output_grid = ARCGrid.from_array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    # Mock policy/value net to guide towards Paint(1,1,1)
    # This is a very basic mock. In reality, the network would learn this.
    class MockPolicyNet(PolicyNet):
        def forward(self, grid_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # Prioritize 'paint' primitive
            primitive_names = dsl.get_primitive_names()
            paint_idx = primitive_names.index("paint")
            logits = torch.zeros(1, len(primitive_names), device=device)
            logits[0, paint_idx] = 10.0 # High logit for paint

            # Mock arg logits to guide towards (1,1,1)
            # This is highly simplified; real arg prediction is complex.
            arg_logits = torch.zeros(1, 10, device=device) # Dummy arg logits
            return logits, arg_logits

    class MockValueNet(ValueNet):
        def forward(self, grid_tensor: torch.Tensor) -> torch.Tensor:
            # High value for grids that are close to the target
            # For simplicity, always return a high value to encourage search
            return torch.tensor([[1.0]], device=device)

    mcts_solver = MCTSSearch(
        dsl=dsl,
        policy_net=MockPolicyNet(10, 64, 2, len(dsl.get_primitive_names()), architecture="cnn").to(device),
        value_net=MockValueNet(10, 64, 2, 1, architecture="cnn").to(device),
        program_prior=program_prior,
        verifier=verifier,
        constraint_solver=constraint_solver,
        cfg=cfg_solver.search
    )

    # Increase budget for test stability
    compute_budget_ms = 5000 # 5 seconds
    node_expansion_budget = 500

    solved_program = mcts_solver.search(input_grid, output_grid, compute_budget_ms, node_expansion_budget)

    assert solved_program is not None
    assert verifier.verify_program_on_examples(solved_program, [
        {"input": input_grid, "output": output_grid}
    ])
    # Check if the program is roughly what we expect (Paint(1,1,1))
    assert len(solved_program.steps) == 1
    assert solved_program.steps[0].primitive.name == "paint"
    assert solved_program.steps[0].args["r"] == 1
    assert solved_program.steps[0].args["c"] == 1
    assert solved_program.steps[0].args["color"] == 1

def test_solver_portfolio_attempt1_and_attempt2(setup_solver_components):
    dsl, policy_net, value_net, program_prior, verifier, constraint_solver, repairer, cfg_solver, device = setup_solver_components

    # Define a slightly more complex task: reflect horizontally
    input_grid = ARCGrid.from_array([[1, 2, 3], [4, 5, 6]])
    output_grid = ARCGrid.from_array([[3, 2, 1], [6, 5, 4]])

    # Mock policy/value net to guide towards ReflectHorizontal
    class MockPolicyNetReflect(PolicyNet):
        def forward(self, grid_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            primitive_names = dsl.get_primitive_names()
            reflect_h_idx = primitive_names.index("reflect_horizontal")
            logits = torch.zeros(1, len(primitive_names), device=device)
            logits[0, reflect_h_idx] = 10.0 # High logit for reflect_horizontal
            arg_logits = torch.zeros(1, 10, device=device) # Dummy arg logits
            return logits, arg_logits

    class MockValueNetReflect(ValueNet):
        def forward(self, grid_tensor: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[1.0]], device=device)

    # Re-initialize portfolio with mock networks
    solver_portfolio = SolverPortfolio(
        dsl=dsl,
        policy_net=MockPolicyNetReflect(10, 64, 2, len(dsl.get_primitive_names()), architecture="cnn").to(device),
        value_net=MockValueNetReflect(10, 64, 2, 1, architecture="cnn").to(device),
        program_prior=program_prior,
        verifier=verifier,
        constraint_solver=constraint_solver,
        repairer=repairer,
        cfg=cfg_solver
    )

    # Increase budget for test stability
    compute_budget_ms = 10000 # 10 seconds
    node_expansion_budget = 1000

    # Attempt 1
    attempt1_program = solver_portfolio.solve_task(
        "test_reflect_task", input_grid, output_grid,
        compute_budget_ms // 2, node_expansion_budget // 2,
        cfg_solver.inference_strategy.attempt1
    )

    assert attempt1_program is not None
    assert verifier.verify_program_on_examples(attempt1_program, [
        {"input": input_grid, "output": output_grid}
    ])
    assert len(attempt1_program.steps) == 1
    assert attempt1_program.steps[0].primitive.name == "reflect_horizontal"

    # Attempt 2 (should also find the same solution given the strong mock policy)
    attempt2_program = solver_portfolio.solve_task(
        "test_reflect_task", input_grid, output_grid,
        compute_budget_ms // 2, node_expansion_budget // 2,
        cfg_solver.inference_strategy.attempt2
    )

    assert attempt2_program is not None
    assert verifier.verify_program_on_examples(attempt2_program, [
        {"input": input_grid, "output": output_grid}
    ])
    assert len(attempt2_program.steps) == 1
    assert attempt2_program.steps[0].primitive.name == "reflect_horizontal"

    # Verify that the outputs are identical for this simple case
    assert attempt1_program.execute(input_grid) == attempt2_program.execute(input_grid)
