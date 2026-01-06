# Project Overview

This project is a modular and scalable template for Deep Learning projects, primarily built with **PyTorch Lightning** for model training and **Hydra** for configuration management. It emphasizes clear separation of concerns, reusability, and ease of experimentation, making it suitable as a boilerplate for various deep learning tasks, especially those involving Natural Language Processing (given the `conllu_parser.py` utility).

**Key Technologies:**
*   **PyTorch Lightning:** For structured and scalable deep learning model training, validation, and testing. It handles boilerplate code, distributed training, and mixed-precision training.
*   **Hydra:** A powerful configuration management framework that allows dynamic composition of configurations, making experiments highly reproducible and manageable.
*   **PyTorch:** The underlying deep learning library.
*   **rootutils:** A utility for consistent project root detection and path management.

**Core Idea and Structure for Reusability:**

The project is designed as a template to facilitate the development of new deep learning projects by providing a well-organized and configurable structure.

### `configs/` Directory: The Heart of Configuration Management

This directory is crucial for the project's modularity and reusability, leveraging Hydra's capabilities. Each subdirectory represents a distinct aspect of the deep learning pipeline, and configurations can be composed dynamically.

*   **`train.yaml` & `eval.yaml`**: These are the main entry points for defining training and evaluation runs. They use Hydra's `defaults` keyword to compose other configuration files.
*   **`data/`**: Defines data modules (`LightningDataModule`). For example, `mnist.yaml` configures the MNIST dataset. To add a new dataset, you would create a new YAML file here and implement the corresponding `LightningDataModule` in `src/data/`.
*   **`model/`**: Configures the `LightningModule` and its associated components (optimizer, learning rate scheduler, and the neural network architecture itself). `mnist.yaml` links to `src/models/mnist_module.py` and `src/models/components/simple_dense_net.py`.
*   **`trainer/`**: Specifies PyTorch Lightning `Trainer` parameters, such as `accelerator`, `devices`, `max_epochs`, `strategy` (e.g., DDP for distributed training). This allows easy switching between CPU, GPU, or multi-GPU setups.
*   **`callbacks/`**: Defines callbacks for the training process (e.g., `ModelCheckpoint`, `EarlyStopping`, `RichProgressBar`). These can be enabled or disabled, and their parameters configured, through simple YAML files.
*   **`logger/`**: Manages different experiment loggers (e.g., WandB, TensorBoard, MLflow, Aim). You can easily enable multiple loggers using `many_loggers.yaml` or switch between them.
*   **`experiment/`**: Contains specific experiment configurations that override or extend the default settings. This is ideal for defining sets of hyperparameters for a particular model or dataset.
*   **`hparams_search/`**: Provides configurations for hyperparameter optimization tools like Optuna, integrated with Hydra's sweeper.
*   **`debug/`**: Offers pre-configured settings for debugging, such as `fast_dev_run` or limiting data, accelerating the debugging process.
*   **`paths/`**: Defines project-specific paths (`root_dir`, `data_dir`, `log_dir`, `output_dir`), ensuring consistency across different environments.
*   **`hydra/`**: Contains Hydra's own configuration, including logging and output directory patterns for runs and sweeps.

This structure allows users to define a new experiment by creating a single YAML file in `configs/experiment/` that combines existing `data`, `model`, `trainer`, `callbacks`, and `logger` configurations, only overriding what's necessary.

### `src/` Directory: Modular Code Implementation

The `src/` directory houses the core Python code, organized into logical modules.

*   **`src/data/`**:
    *   **`mnist_datamodule.py`**: An example of a `LightningDataModule` that handles downloading, splitting, and preparing the MNIST dataset. This is where all dataset-specific logic resides. For a new dataset, a new `LightningDataModule` file would be created here.
    *   **`conllu_parser.py`**: A utility for parsing CoNLL-U format files, indicating the project's potential in NLP tasks. This showcases how custom data processing utilities can be integrated.
*   **`src/models/`**:
    *   **`mnist_module.py`**: An example of a `LightningModule` that encapsulates the model, loss function, optimization logic, and metric calculation for the MNIST classification task. This is the central piece for defining any deep learning model.
    *   **`components/simple_dense_net.py`**: Defines a basic `torch.nn.Module` (the actual neural network architecture). `LightningModule`s typically wrap these `torch.nn.Module`s.
*   **`src/utils/`**: Contains various helper functions that are generic and can be reused across different parts of the project.
    *   **`instantiators.py`**: Facilitates the dynamic creation of objects (callbacks, loggers) from their configurations using Hydra's `instantiate` method.
    *   **`pylogger.py`**, **`logging_utils.py`**: Custom logging utilities, including a `RankedLogger` for multi-GPU setups.
    *   **`rich_utils.py`**, **`utils.py`**: General utilities for tasks like printing configuration trees, enforcing tags for experiment tracking, and handling warnings.

This separation ensures that data loading, model architecture, training logic, and utility functions are independently developed and easily swappable, promoting maintainability and reusability.

### Building and Running

The primary entry points for running the project are `train.py` and `eval.py`.

*   **Training:**
    ```bash
    python src/train.py
    # or with a specific experiment config
    python src/train.py experiment=example
    # or override parameters from command line
    python src/train.py trainer.max_epochs=5 model.optimizer.lr=0.005
    ```
*   **Evaluation:**
    ```bash
    python src/eval.py ckpt_path=/path/to/your/checkpoint.ckpt
    ```
*   **Hyperparameter Search (with Optuna example):**
    ```bash
    python src/train.py -m hparams_search=mnist_optuna experiment=example
    ```
*   **Debugging:**
    ```bash
    python src/train.py debug=default
    python src/train.py debug=fdr # fast dev run
    ```

The project utilizes `rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)` to automatically add the project root to `PYTHONPATH` and set up the `PROJECT_ROOT` environment variable. This simplifies imports and path handling.

### Development Conventions

*   **Configuration-driven Development:** All major aspects of a run (data, model, trainer, callbacks, loggers) are controlled via YAML configurations managed by Hydra.
*   **PyTorch Lightning Structure:** Adherence to `LightningDataModule` and `LightningModule` patterns for clear and concise deep learning code.
*   **Modularity:** Emphasis on creating small, focused modules and configuration files that can be combined and reused.
*   **Experiment Tracking:** Integration with various loggers (WandB, TensorBoard, MLflow, Aim) for comprehensive experiment tracking.
*   **Consistent Logging:** Usage of a `RankedLogger` to handle logging gracefully in distributed environments.
*   **Path Management:** `rootutils` ensures all paths are relative to the project root, enhancing portability.
*   **Type Hinting:** (Inferred from `src` files) Use of type hints for improved code readability and maintainability.
*   **Code Quality:** The presence of `.pre-commit-config.yaml` and `.github/workflows` (for `code-quality-main.yaml`, `code-quality-pr.yaml`) suggests a focus on automated code quality checks and CI/CD.

This project structure provides a robust foundation for deep learning research and development, allowing for quick iteration and organized experimentation.