# FALCON Software MVP

FALCON = Fault Analysis Logic for Component Optimization & Burn-in

This is the first software-only prototype for SIH26170.

## Pipeline

Input time-series -> preprocessing -> Gaussian Process -> posterior trajectories
-> decision instability -> adaptive measurement schedule -> AUC + tau_63
-> normal reference -> Mahalanobis anomaly score -> NORMAL / ANOMALOUS

The prototype compares fixed-rate and adaptive sampling using safe RC-like
synthetic trajectories. It does not represent semiconductor reliability physics.

## Run

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py

Windows:
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python run_demo.py

Results are written to results/.

## Important scope

- GP predictive variance is NOT treated as a direct anomaly score.
- The primary adaptive trigger is classification decision instability under
  GP posterior resampling.
- Isolation Forest is not used as an uncertainty proxy.
- KWW/stretched-exponential fitting is not used.
- Physical stress is not dynamically controlled.
- No destructive testing is required.
- Simulation results are not industrial performance claims.
