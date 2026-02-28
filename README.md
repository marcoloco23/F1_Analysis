# F1 Race Points Prediction

Predict Formula 1 race finishing points from free practice session telemetry using XGBoost.

## How It Works

1. **Data extraction** — Pulls lap-by-lap telemetry (timing, speeds, tire data) from FP1/FP2/FP3 via [FastF1](https://docs.fastf1.dev/)
2. **Feature engineering** — Aggregates 12 metrics per driver (mean/max/min/std → 48 features)
3. **Scoring** — Computes composite performance scores across pace, consistency, sector times, and speed traps
4. **Prediction** — Trains an XGBoost regressor to predict race points from practice performance

## Setup

```bash
pip install .                    # core dependencies
pip install ".[notebook]"        # + Jupyter support
pip install ".[dev]"             # + linting/testing
```

Requires Python 3.10+.

## Usage

### Compute Practice Session Scores

```bash
python get_scores.py --year 2024 --gp Bahrain
```

### Build Training Dataset & Train Model

```bash
# Build dataset from a full season (downloads ~60 sessions via fastf1)
python train_ai.py --year 2024 --prepare

# Train on existing dataset
python train_ai.py

# Train with 5-fold cross-validation
python train_ai.py --cv
```

### Interactive Analysis

Open `F1.ipynb` in Jupyter for visualized predictions and feature importance analysis.

## Project Structure

```
├── constants.py      # Telemetry columns, score weights, configuration
├── get_data.py       # FastF1 data extraction and per-driver aggregation
├── get_scores.py     # Composite performance scoring system
├── train_ai.py       # Dataset preparation and XGBoost training
├── F1.ipynb          # Interactive analysis notebook
└── pyproject.toml    # Dependencies and project metadata
```

## Data Pipeline

```
FastF1 API (FP1/FP2/FP3 telemetry)
  → Per-lap metrics: times, speeds, tire compound, tire life
  → Per-driver aggregation: mean/max/min/std (48 features)
  → MinMax normalization
  → XGBoost regression → predicted race points
```

## Score Dimensions

| Category | Metrics | Weight |
|---|---|---|
| Best lap time | Fastest lap vs field | 2.0× |
| Sector pace | Best sector times (×3) | 2.0× each |
| Mean pace | Average lap + sector times | 1.0× |
| Consistency | Lap + sector time std dev | 0.5× |
| Top speed | Max across 4 speed traps | 1.0× |
