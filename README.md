# dob-hocbf

Companion code for the paper:

**Disturbance-Observer-Based CBF Safety Filters with State-Dependent Error Certification**

This repository contains the dynamic-bicycle obstacle-avoidance simulation used
to evaluate disturbance-observer-based CBF/HOCBF safety filters with
state-dependent observer-error certification. The experiment compares a nominal
CLF-ECBF-QP controller, a global-certificate DOB baseline, a tunable-margin DOB
benchmark, and the proposed state-dependent DOB safety filter under continuous
body-frame disturbances.

## Running the Experiment

Run the full dynamic-bicycle simulation with `uv`:

```bash
uv run dob-hocbf-bicycle6d
```

The script writes outputs to:

```text
outputs/bicycle6d/
```

Generated files include the trajectory plot, safety/ECBF traces, DOB error and
margin plots, CLF performance plot, metric bar chart, and `summary.csv`.

## Repository Layout

```text
src/bicycle6d/        simulation, controller, observer, metrics, and plotting code
outputs/bicycle6d/    generated experiment outputs
pyproject.toml        package metadata and dependencies
```
