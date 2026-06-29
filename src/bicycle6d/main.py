"""Run the dynamic-bicycle continuous-disturbance DOB-CLF-ECBF-QP experiment."""

from __future__ import annotations

from .config import default_config
from .controller import local_initial_error_bounds
from .metrics import compute_metrics, write_summary
from .plotting import (
    plot_clf_performance,
    plot_dob_estimation,
    plot_metric_bars,
    plot_safety,
    plot_trajectory,
)
from .simulation import METHODS, run_all


def main() -> None:
    cfg = default_config()
    initial_error_bounds = local_initial_error_bounds(cfg.bicycle.state0, 0.0, cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    results = run_all(cfg)
    metrics = [
        compute_metrics(
            results[spec.key],
            cfg.bicycle.dt,
            cfg.bicycle.finish_x,
        )
        for spec in METHODS
    ]
    write_summary(metrics, cfg.output_dir / "summary.csv")
    plot_trajectory(results, cfg, cfg.output_dir / "trajectory.png")
    plot_safety(results, cfg, cfg.output_dir / "safety_ecbf.png")
    plot_dob_estimation(results, cfg, cfg.output_dir / "dob_ecbf.png")
    plot_clf_performance(results, cfg, cfg.output_dir / "clf_performance.png")
    plot_metric_bars(metrics, cfg.output_dir / "metric_bars.png")

    print("Continuous-disturbance DOB-CLF-ECBF-QP dynamic-bicycle experiment")
    print(f"outputs: {cfg.output_dir}")
    print(
        "observer: "
        f"k={cfg.observer.gain:.3f}, "
        f"beta_h={cfg.observer.b_h_bound:.3f}, "
        f"steady_error={cfg.observer.steady_error_bound:.3f}, "
        f"e0_max={initial_error_bounds.max():.3f}"
    )
    for item in metrics:
        print(
            f"{item.name:34s} "
            f"min_h={item.min_h:8.3f} "
            f"viol={item.violation_pct:5.1f}% "
            f"max_err={item.max_observer_error:7.3f} "
            f"max_ebar={item.max_error_bound:7.3f} "
            f"bound_viol={item.error_bound_violation_pct:5.1f}% "
            f"mean_rho={item.mean_rho:7.3f} "
            f"max_rho={item.max_rho:7.3f} "
            f"slack_int={item.clf_slack_integral:8.3f} "
            f"max_slack={item.max_clf_slack:8.3f} "
            f"min_margin={item.min_qp_margin:8.3f} "
            f"smooth={item.control_variation:7.3f} "
            f"infeas={item.infeasible_count} "
            f"done={item.completed} "
            f"t_done={item.completion_time:5.2f}"
        )


if __name__ == "__main__":
    main()
