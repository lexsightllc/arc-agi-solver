import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import mlflow
import os
import json
from tqdm import tqdm

from src.core.dsl import DSL, Program
from src.learning.models import PolicyNet, ValueNet, ProgramPriorModel
from src.learning.dataset import SyntheticARCProblemGenerator, ARCProblemDataset, collate_fn
from src.learning.prior import ProgramPrior
from src.utils.logging import get_logger
from src.utils.seed import set_deterministic_seed

logger = get_logger(__name__)

class CurriculumManager:
    """Manages the curriculum learning stages."""
    def __init__(self, stages: List[DictConfig]):
        self.stages = stages
        self.current_stage_idx = 0

    def get_current_stage(self) -> DictConfig:
        if self.current_stage_idx >= len(self.stages):
            return None
        return self.stages[self.current_stage_idx]

    def advance_stage(self):
        self.current_stage_idx += 1
        logger.info(f"Advanced to curriculum stage {self.current_stage_idx + 1}/{len(self.stages)}")

    @property
    def is_finished(self) -> bool:
        return self.current_stage_idx >= len(self.stages)


class CurriculumTrainer:
    """Orchestrates training of policy and value networks using curriculum learning."""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dsl = DSL()
        set_deterministic_seed(cfg.seed)

        # Initialize models
        self.policy_net: PolicyNet = hydra.utils.instantiate(cfg.model.policy_net).to(self.device)
        self.value_net: ValueNet = hydra.utils.instantiate(cfg.model.value_net).to(self.device)
        self.program_prior: ProgramPrior = hydra.utils.instantiate(cfg.model.program_prior, dsl=self.dsl)

        # Optimizers
        self.optimizer_policy = self._get_optimizer(self.policy_net.parameters(), cfg.training.learning_rate)
        self.optimizer_value = self._get_optimizer(self.value_net.parameters(), cfg.training.learning_rate)

        # Schedulers
        self.scheduler_policy = self._get_scheduler(self.optimizer_policy, cfg.training.scheduler)
        self.scheduler_value = self._get_scheduler(self.optimizer_value, cfg.training.scheduler)

        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

        self.curriculum_manager = CurriculumManager(cfg.training.curriculum.stages)
        self.synthetic_generator = hydra.utils.instantiate(cfg.training.synthetic_generator, dsl=self.dsl)

        os.makedirs(cfg.output_dir, exist_ok=True)

    def _get_optimizer(self, params, lr):
        if self.cfg.training.optimizer == "Adam":
            return optim.Adam(params, lr=lr)
        elif self.cfg.training.optimizer == "SGD":
            return optim.SGD(params, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.training.optimizer}")

    def _get_scheduler(self, optimizer, scheduler_type):
        if scheduler_type == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        else:
            return None # No scheduler

    def train(self):
        global_step = 0
        for epoch in range(self.cfg.training.epochs):
            if self.curriculum_manager.is_finished:
                logger.info("Curriculum finished. Stopping training.")
                break

            current_stage = self.curriculum_manager.get_current_stage()
            logger.info(f"Epoch {epoch+1}/{self.cfg.training.epochs} - Current Curriculum Stage: {current_stage.name}")

            # Generate synthetic tasks for the current stage
            synthetic_tasks = [
                self.synthetic_generator.generate_task(current_stage.task_complexity_min,
                                                      current_stage.primitives_subset)
                for _ in range(current_stage.num_tasks_per_epoch)
            ]
            dataset = ARCProblemDataset(synthetic_tasks, self.dsl)
            dataloader = DataLoader(dataset, batch_size=self.cfg.training.batch_size, shuffle=True, collate_fn=collate_fn)

            self.policy_net.train()
            self.value_net.train()
            total_policy_loss = 0
            total_value_loss = 0

            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
                input_grids = batch["input_grid"].to(self.device)
                output_grids = batch["output_grid"].to(self.device)
                program_representations = batch["program_representation"]

                # Policy Network Training (Supervised Learning on generated programs)
                self.optimizer_policy.zero_grad()
                policy_logits, arg_logits = self.policy_net(input_grids)

                # This part is highly simplified. Program_representations need to be parsed
                # into target primitive IDs and argument values for loss calculation.
                # For now, we'll use a dummy target.
                # target_primitive_ids = torch.randint(0, policy_logits.shape[1], (policy_logits.shape[0],)).to(self.device)
                # policy_loss = self.policy_loss_fn(policy_logits, target_primitive_ids)
                policy_loss = torch.tensor(0.0).to(self.device) # Placeholder

                # Value Network Training (Predicting solvability or reward)
                self.optimizer_value.zero_grad()
                predicted_values = self.value_net(input_grids)
                # Dummy target: 1 if output_grid is not empty, 0 otherwise
                target_values = (output_grids.sum(dim=(1,2,3)) > 0).float().unsqueeze(1).to(self.device)
                value_loss = self.value_loss_fn(predicted_values, target_values)

                # Combine losses and backpropagate
                loss = policy_loss + value_loss # Add other losses like MDL, constraint violation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.training.gradient_clip_norm)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.training.gradient_clip_norm)
                self.optimizer_policy.step()
                self.optimizer_value.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                global_step += 1

                if global_step % self.cfg.training.log_interval == 0:
                    mlflow.log_metric("train_policy_loss", policy_loss.item(), step=global_step)
                    mlflow.log_metric("train_value_loss", value_loss.item(), step=global_step)
                    logger.info(f"Step {global_step}: Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

            avg_policy_loss = total_policy_loss / len(dataloader)
            avg_value_loss = total_value_loss / len(dataloader)
            mlflow.log_metric("epoch_avg_policy_loss", avg_policy_loss, step=epoch)
            mlflow.log_metric("epoch_avg_value_loss", avg_value_loss, step=epoch)
            logger.info(f"Epoch {epoch+1} Summary: Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f}")

            if self.scheduler_policy: self.scheduler_policy.step(avg_policy_loss)
            if self.scheduler_value: self.scheduler_value.step(avg_value_loss)

            # Meta-learning: Update and distill program prior
            if (epoch + 1) % self.cfg.training.meta_learning.prior_update_interval_epochs == 0:
                # In a real scenario, solved tasks would be collected and used to update the prior
                # For synthetic tasks, we can use the generated programs directly.
                solved_programs = [Program.from_json(task["program"], self.dsl) for task in synthetic_tasks if task.get("program")]
                self.program_prior.update_prior(solved_programs)
                logger.info(f"Program prior updated with {len(solved_programs)} programs.")

            if (epoch + 1) % self.cfg.training.meta_learning.prior_distillation_interval_epochs == 0:
                # Distill prior into policy network (e.g., by adding a distillation loss)
                # This would involve sampling from the prior and training the policy to mimic it.
                logger.info("Distilling program prior into policy network (placeholder).")

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'policy_net_state_dict': self.policy_net.state_dict(),
                'value_net_state_dict': self.value_net.state_dict(),
                'optimizer_policy_state_dict': self.optimizer_policy.state_dict(),
                'optimizer_value_state_dict': self.optimizer_value.state_dict(),
                'program_prior_state_dict': self.program_prior.get_state_dict(),
            }, os.path.join(self.cfg.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))

            # Advance curriculum stage if conditions met (e.g., performance plateau, or fixed epochs)
            # For now, advance after a fixed number of epochs per stage
            if (epoch + 1) % current_stage.epochs == 0:
                self.curriculum_manager.advance_stage()

        logger.info("Training process completed.")
        # Save final model
        torch.save(self.policy_net.state_dict(), os.path.join(self.cfg.output_dir, "best_model.pt"))
        torch.save(self.value_net.state_dict(), os.path.join(self.cfg.output_dir, "best_value_model.pt"))
