import torch
import torch.nn as nn
from typing import List, Dict, Any
from src.core.dsl import Program, DSL
from src.learning.models import ProgramPriorModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

class ProgramPrior:
    """Manages a learned prior distribution over DSL programs."""
    def __init__(self, dsl: DSL, prior_type: str = "frequency", neural_prior_model: ProgramPriorModel = None):
        self.dsl = dsl
        self.prior_type = prior_type
        self.program_counts: Dict[str, int] = {}
        self.total_programs = 0

        if self.prior_type == "neural":
            if neural_prior_model is None:
                raise ValueError("Neural prior model must be provided for 'neural' prior_type.")
            self.neural_prior_model = neural_prior_model
            self.optimizer = torch.optim.Adam(self.neural_prior_model.parameters(), lr=0.001)
            self.loss_fn = nn.BCEWithLogitsLoss() # Or other suitable loss
        else:
            self.neural_prior_model = None

    def update_prior(self, solved_programs: List[Program]):
        """Updates the prior based on newly solved programs."""
        if self.prior_type == "frequency":
            for program in solved_programs:
                program_str = str(program) # Simple string representation for frequency counting
                self.program_counts[program_str] = self.program_counts.get(program_str, 0) + 1
                self.total_programs += 1
            logger.debug(f"Frequency prior updated. Total unique programs: {len(self.program_counts)}")
        elif self.prior_type == "neural":
            # For neural prior, this would involve training the neural_prior_model
            # on embeddings of solved programs, perhaps with a positive label.
            # This is a placeholder for a more complex training loop.
            logger.debug("Neural prior update triggered (training placeholder).")
            # Example: Create dummy embeddings for solved programs
            # In a real scenario, program embeddings would come from a program encoder.
            dummy_embeddings = torch.randn(len(solved_programs), self.neural_prior_model.fc1.in_features)
            targets = torch.ones(len(solved_programs), 1)

            self.optimizer.zero_grad()
            logits = self.neural_prior_model(dummy_embeddings)
            loss = self.loss_fn(logits, targets)
            loss.backward()
            self.optimizer.step()
            logger.debug(f"Neural prior trained with loss: {loss.item():.4f}")

    def get_program_log_likelihood(self, program: Program, program_embedding: torch.Tensor = None) -> float:
        """Returns the log-likelihood of a program according to the prior."""
        if self.prior_type == "frequency":
            program_str = str(program)
            count = self.program_counts.get(program_str, 0)
            if self.total_programs == 0: return -float('inf') # Avoid division by zero
            # Add Laplace smoothing to avoid zero probabilities
            return torch.log(torch.tensor((count + 1) / (self.total_programs + len(self.program_counts)))).item()
        elif self.prior_type == "neural":
            if program_embedding is None:
                raise ValueError("Program embedding must be provided for neural prior.")
            self.neural_prior_model.eval()
            with torch.no_grad():
                logits = self.neural_prior_model(program_embedding.unsqueeze(0)) # Add batch dim
                # Convert logits to log-probabilities (e.g., using sigmoid and log)
                return F.logsigmoid(logits).item()
        return -float('inf') # Default for unknown prior type

    def get_state_dict(self) -> Dict[str, Any]:
        state = {
            "prior_type": self.prior_type,
            "program_counts": self.program_counts,
            "total_programs": self.total_programs,
        }
        if self.prior_type == "neural" and self.neural_prior_model:
            state["neural_prior_model_state_dict"] = self.neural_prior_model.state_dict()
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        self.prior_type = state["prior_type"]
        self.program_counts = state["program_counts"]
        self.total_programs = state["total_programs"]
        if self.prior_type == "neural" and self.neural_prior_model:
            self.neural_prior_model.load_state_dict(state["neural_prior_model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
