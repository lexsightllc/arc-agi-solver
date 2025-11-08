<!-- SPDX-License-Identifier: MPL-2.0 -->
# ARC AGI Solver

This project implements a neuro-symbolic program induction system designed to solve tasks from the Abstraction and Reasoning Corpus (ARC) AGI challenge. It aims to substantially advance beyond heuristic pattern matching by learning to invent and select compact grid transformation programs that generalize to unseen tasks.

## Core Goal
To develop a neuro-symbolic program induction system that learns to invent and select compact grid transformation programs that generalize to unseen tasks.

## Technologies Used

*   **Learned Components:** PyTorch
*   **High-Performance Grid Operations:** NumPy, Numba
*   **Graph Representations:** NetworkX
*   **Constraint Satisfaction:** Z3
*   **E-graph Rewriting (Conceptual):** Inspired by egglog for algebraic simplification of DSL programs.
*   **Custom Kernels:** Triton (for profiling-indicated bottlenecks)
*   **Configuration Management:** Hydra
*   **Experiment Tracking:** MLflow
*   **Data and Artifact Versioning:** DVC
*   **Command Line Interface:** Typer
*   **Containerization:** Docker
*   **Continuous Integration:** GitHub Actions
*   **Testing:** pytest, Hypothesis
*   **Determinism:** Documented random seed discipline

## Architecture Overview

1.  **Input Layer:** Parses colored grids into a typed intermediate representation (IR) with explicit symmetries and object proposals.
2.  **Solver Portfolio:** Includes a learned policy and value network to guide a tree search over a compact Domain Specific Language (DSL) of grid primitives (e.g., `paint`, `copy`, `reflect`, `rotate`, `flood_fill`, `count`, `compress`, `relational_compose`).
3.  **Constraint Layer:** Enforces color, shape, and cardinality invariants inferred from training examples using Z3.
4.  **Minimum Description Length (MDL) Objective:** Guides program selection by preferring short, consistent programs.
5.  **Reflective Self-Repair Loop:** Attempts small edits to candidate programs when verification fails.
6.  **Verifier:** Executes candidate programs against training examples and checks for exact equality with expected outputs.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/arc-agi-solver.git
    cd arc-agi-solver
    ```
2.  **Install with Docker (Recommended for reproducibility):**
    ```bash
    docker build -t arc-agi-solver .
    docker run -it arc-agi-solver /bin/bash
    ```
3.  **Install locally (Python 3.9+):**
    ```bash
    pip install -e .
    # Or install dependencies directly:
    pip install -r requirements.txt
    ```

## Usage

This project uses `Typer` for its command-line interface.

*   **Train the model:**
    ```bash
    python src/main.py train --config-path config/training/curriculum.yaml
    ```
*   **Evaluate the model:**
    ```bash
    python src/main.py evaluate --model-path runs/best_model.pt --dataset-path data/processed/evaluation_tasks.json
    ```
*   **Generate predictions for ARC tasks:**
    ```bash
    python src/main.py predict --input-path data/raw/arc-dataset/evaluation_challenges.json --output-path submission.json
    ```
*   **Run preflight validation on a submission file:**
    ```bash
    python src/main.py validate --submission-path submission.json --challenges-path data/raw/arc-dataset/evaluation_challenges.json
    ```

## Experiment Tracking and Reproducibility

*   **MLflow:** Used for logging parameters, metrics, and artifacts (models, plots) for each training run. Access the UI with `mlflow ui`.
*   **DVC:** Manages data and model versioning. Use `dvc pull` to get data and `dvc push` to store changes.
*   **Hydra:** Provides structured configuration management, allowing easy override of parameters via the command line.
*   **Random Seed Discipline:** All random operations are seeded for full reproducibility, controlled via `src/utils/seed.py` and configured in `config/config.yaml`.

## Testing

Run unit, integration, and property-based tests using `pytest`:

```bash
pytest
```

## Interpretability

An interpretability protocol is provided to render discovered programs, visualize intermediate grid transformations, and explain constraint firing. This is available via a dedicated notebook or CLI command:

```bash
python src/main.py interpret --task-id <task_id> --program-path <program_file>
```

## Deliverables

Upon successful execution and evaluation, the following artifacts will be produced:

*   **Container Image:** `arc-agi-solver:latest` (Docker image).
*   **Trained Weights:** `runs/<run_id>/artifacts/model.pt` (PyTorch model state dict).
*   **Configuration Files:** `runs/<run_id>/.hydra/config.yaml` (Hydra config for the best performing run).
*   **Reproducibility Archive:** A DVC-tracked archive containing fixed seeds and environment hashes.
*   **Experiment Tracking Report:** An MLflow UI link and a summary report generated from the best run.
*   **Interpretability Notebook:** `notebooks/interpretability.ipynb` for qualitative inspection of solved tasks.
*   **Final Submission File:** `submission.json` produced by the validated pipeline.

## Expected Performance and Future Work

This system is designed to achieve substantial zero-shot generalization, aiming for a top-tier performance in ARC-like tasks by moving beyond simple pattern matching. We anticipate outperforming purely heuristic or purely neural approaches on compositional tasks.

**Next Steps to Close the Gap:**

1.  **DSL Expansion:** Systematically expand the DSL with more powerful relational primitives.
2.  **Meta-Learning for DSL Invention:** Explore meta-learning techniques to dynamically invent or adapt DSL primitives for novel task families.
3.  **Improved Program Prior:** Integrate more sophisticated graph neural networks or transformer-based models for a richer learned prior over programs.
4.  **Distributed Search:** Implement distributed MCTS for higher throughput and deeper search within budget.

**Scientific Novelty:**

This design's novelty lies in its tightly integrated neuro-symbolic architecture, where a learned policy guides a symbolic search over a compositional DSL, with robust constraint satisfaction and a self-repair mechanism. The learned prior, updated through meta-learning on synthetic tasks, allows for efficient generalization without relying on evaluation tasks. This approach enables the system to *invent* and *verify* programs, fostering transferability to new problems by learning underlying compositional structures rather than merely memorizing known patterns.

## License

The ARC AGI Solver is available under the [Mozilla Public License 2.0](LICENSE). Any file distributed under the MPL-2.0 must retain its license header, and if you distribute modified versions of MPL-covered files, you must make those modifications available under MPL-2.0 as well. Larger works that merely aggregate or interface with this project may remain under their own licenses, provided the MPL terms continue to be satisfied.

All distribution artifacts must include the accompanying [NOTICE](NOTICE) file so that third-party acknowledgments remain intact.

## Credits

ProjectName © 2025 Augusto "Guto" Ochoa Ughini. The project builds upon a vibrant open-source ecosystem; please consult the dependency metadata for upstream acknowledgments and licensing details.