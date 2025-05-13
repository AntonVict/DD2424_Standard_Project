# Data Storage and Logging Structure

This document describes how experiment runs are logged and stored in this project, including the structure and purpose of each storage location.

## 1. `raw-results.md`
- **Purpose:**
  - Serves as a cumulative, human-readable record of all experiment runs.
- **Contents:**
  - Each run appends a new section with a heading indicating the stage/problem, timestamp, and strategy (if relevant).
  - Links to the generated plots (loss, accuracy, confusion matrix) are included as images.
  - The path to the saved model (if applicable) is printed.
  - The full log output for the run is appended for traceability and debugging.
- **Example Entry:**
  ```markdown
  ## Stage 3 Imbalanced Classification Run (2025-05-13_16-56-07)
  ![](logs/plots/stage3-imbalance-2025-05-13_16-56-07-loss.png)
  ![](logs/plots/stage3-imbalance-2025-05-13_16-56-07-acc.png)
  ![](logs/plots/stage3-imbalance-2025-05-13_16-56-07-confmat.png)
  
  **Model saved at:** `models/stage3-imbalance-2025-05-13_16-56-07.pt`
  
  **Log:**
  [Telemetry] ...
  ```

## 2. `logs/`
- **Purpose:**
  - Stores the raw log files for each run.
- **Contents:**
  - Each run creates a log file named `logs-YYYY-MM-DD_HH-MM-SS.log`.
  - All telemetry, progress, and results for the run are written here.
- **Example:**
  - `logs/logs-2025-05-13_16-56-07.log`

## 3. `logs/plots/`
- **Purpose:**
  - Stores all generated plots for each run.
- **Contents:**
  - Loss and accuracy curves: `stageX-<desc>-YYYY-MM-DD_HH-MM-SS-loss.png` and `...-acc.png`.
  - Confusion matrices (if applicable): `stageX-<desc>-YYYY-MM-DD_HH-MM-SS-confmat.png`.
- **Example:**
  - `logs/plots/stage3-imbalance-2025-05-13_16-56-07-loss.png`
  - `logs/plots/stage3-imbalance-2025-05-13_16-56-07-acc.png`
  - `logs/plots/stage3-imbalance-2025-05-13_16-56-07-confmat.png`

## 4. `models/`
- **Purpose:**
  - Stores the trained model weights for each run.
- **Contents:**
  - Each run saves a model file named with the stage, description, and timestamp, e.g. `stage3-imbalance-2025-05-13_16-56-07.pt`.
- **Example:**
  - `models/stage3-imbalance-2025-05-13_16-56-07.pt`

## 5. Loading a Stored Model
- **How to Load:**
  1. Instantiate the same model architecture as used during training (e.g., ResNet34 with the correct number of output classes).
  2. Load the saved weights using `torch.load` and `load_state_dict`.
- **Example Code:**
  ```python
  import torch
  import torchvision.models as models
  import torch.nn as nn

  # Instantiate the model architecture
  model = models.resnet34(weights=None)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, 37)  # Use the correct number of classes

  # Load the saved weights
  model.load_state_dict(torch.load('models/stage3-imbalance-2025-05-13_16-56-07.pt', map_location='cpu'))
  model.eval()  # Set to eval mode for inference
  ```
- **Notes:**
  - The model architecture must match exactly (including the number of output classes).
  - For further training, set the model to `train()` mode and attach an optimizer.

---
This structure ensures that all experiment results, logs, plots, and models are reproducible, traceable, and easy to review or reuse.
