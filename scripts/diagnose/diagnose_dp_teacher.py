#!/usr/bin/env python
"""
In-memory diagnostics for the CURRENT DP teacher.

Does NOT regenerate trajectories, train models, or overwrite checkpoints.
Writes only runs/eval/dp_teacher_diagnostics.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ts_jepa.config import load_config, project_root
from ts_jepa.data.trajectory_generator import build_env_and_teacher
from ts_jepa.control.dp_teacher import DPControlTeacher


def _act_candidates(teacher: DPControlTeacher, state: np.ndarray) -> np.ndarray:
    """Mirror DPControlTeacher.act() candidate construction (read-only)."""
    state = np.asarray(state, dtype=np.float64).reshape(4)
    table_u = float(teacher.policy[teacher._index_of(state)])
    deltas = np.array([0.0, -4.0, 4.0, -8.0, 8.0], dtype=np.float64)
    local = np.clip(table_u + deltas, teacher.forces[0], teacher.forces[-1])
    coarse = teacher.forces[:: max(1, len(teacher.forces) // 5)]
    return np.unique(np.concatenate([local, coarse]))


def _global_discrete_argmin(teacher: DPControlTeacher, state: np.ndarray) -> tuple[float, float, dict[str, float]]:
    state = np.asarray(state, dtype=np.float64).reshape(4)
    q_by_u: dict[str, float] = {}
    best_u = float(teacher.forces[0])
    best_q = float("inf")
    for u in teacher.forces:
        q = float(teacher._bellman_cost(state, float(u)))
        q_by_u[str(float(u))] = q
        if q < best_q - 1e-12 or (abs(q - best_q) <= 1e-12 and abs(u) < abs(best_u)):
            best_q = q
            best_u = float(u)
    return best_u, best_q, q_by_u


def main() -> None:
    config = load_config()
    t0 = time.perf_counter()
    env, teacher = build_env_and_teacher(config)
    build_s = time.perf_counter() - t0

    report: dict = {
        "build_time_s": round(build_s, 3),
        "formulation": {
            "physics_dt": float(env.ode.dt),
            "dp_substeps": int(teacher.dp_substeps),
            "effective_dp_dt": float(teacher.effective_dp_dt),
            "stage_cost": "sum_{j=1..N} 0.5||s_j-s*||^2 + 0.5 R u^2  (control once per DP decision)",
            "control_cost_charge": "once_per_dp_decision",
            "control_cost_paper_status": (
                "Paper is silent; current implementation uses one control-cost charge per DP decision."
            ),
            "bellman": "Q(s,u)=stage(s,u)+gamma*V(quantize(s_N))",
            "desired_state": teacher.desired_state.tolist(),
            "R": float(teacher.control_effort_weight),
            "gamma": float(teacher.discount),
            "vi_iters": int(teacher.value_iteration_iters),
        },
        "force_bins": [float(u) for u in teacher.forces],
        "grid_shape": list(teacher.grid.shape),
        "grid_axes": {
            "x": [float(teacher.grid.x[0]), float(teacher.grid.x[-1]), int(len(teacher.grid.x))],
            "x_dot": [float(teacher.grid.x_dot[0]), float(teacher.grid.x_dot[-1]), int(len(teacher.grid.x_dot))],
            "theta": [float(teacher.grid.theta[0]), float(teacher.grid.theta[-1]), int(len(teacher.grid.theta))],
            "theta_dot": [
                float(teacher.grid.theta_dot[0]),
                float(teacher.grid.theta_dot[-1]),
                int(len(teacher.grid.theta_dot)),
            ],
        },
    }

    # ------------------------------------------------------------------ A
    uniq = np.unique(teacher.policy)
    n_zero = int(np.sum(teacher.policy == 0.0))
    n_pos = int(np.sum(teacher.policy > 0.0))
    n_neg = int(np.sum(teacher.policy < 0.0))
    report["policy"] = {
        "unique_actions": [float(u) for u in uniq],
        "num_unique": int(uniq.size),
        "n_zero": n_zero,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "fraction_zero": float(n_zero / teacher.policy.size),
        "percentage_zero": float(100.0 * n_zero / teacher.policy.size),
    }
    pass_a = bool(uniq.size >= 3 and n_pos > 0 and n_neg > 0)

    # ------------------------------------------------------------------ B
    reps = {
        "+0.25": np.array([0.0, 0.0, 0.25, 0.0], dtype=np.float64),
        "-0.25": np.array([0.0, 0.0, -0.25, 0.0], dtype=np.float64),
        "+0.10": np.array([0.0, 0.0, 0.10, 0.0], dtype=np.float64),
        "-0.10": np.array([0.0, 0.0, -0.10, 0.0], dtype=np.float64),
        "eq": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    }
    rep_report = {}
    for name, state in reps.items():
        table_u = float(teacher.policy[teacher._index_of(state)])
        act_u = float(teacher.act(state))
        rep_report[name] = {
            "state": state.tolist(),
            "nearest_cell": list(teacher._index_of(state)),
            "table_policy_u": table_u,
            "act_u": act_u,
        }
    report["representative_states"] = rep_report

    # Dynamics-aligned check via full discrete argmin (same Bellman cost), not theta-sign heuristic.
    dyn = {}
    for name in ("+0.25", "-0.25", "+0.10", "-0.10"):
        u_star, _, _ = _global_discrete_argmin(teacher, reps[name])
        dyn[name] = float(u_star)
    report["dynamics_preferred_actions_full_discrete"] = dyn

    act_pos = rep_report["+0.25"]["act_u"]
    act_neg = rep_report["-0.25"]["act_u"]
    nonzero_pair = act_pos != 0.0 and act_neg != 0.0
    opposite = np.sign(act_pos) * np.sign(act_neg) < 0
    match_dyn_pos = np.sign(act_pos) == np.sign(dyn["+0.25"]) and act_pos != 0.0
    match_dyn_neg = np.sign(act_neg) == np.sign(dyn["-0.25"]) and act_neg != 0.0
    small_nonzero = rep_report["+0.10"]["act_u"] != 0.0 and rep_report["-0.10"]["act_u"] != 0.0
    small_opposite = np.sign(rep_report["+0.10"]["act_u"]) * np.sign(rep_report["-0.10"]["act_u"]) < 0
    pass_b = bool(
        nonzero_pair and opposite and match_dyn_pos and match_dyn_neg and small_nonzero and small_opposite
    )
    report["diagnostics_B"] = {
        "nonzero_pm_0.25": nonzero_pair,
        "opposite_pm_0.25": bool(opposite),
        "matches_dynamics_+0.25": bool(match_dyn_pos),
        "matches_dynamics_-0.25": bool(match_dyn_neg),
        "nonzero_pm_0.10": small_nonzero,
        "opposite_pm_0.10": bool(small_opposite),
    }

    # ------------------------------------------------------------------ C
    s = reps["+0.25"]
    probe_u = [-20.0, -4.0, 0.0, 4.0, 20.0]
    successor = {}
    cells = []
    v_list = []
    q_list = []
    for u in probe_u:
        final, stage = teacher._rollout_constant_force(s, float(u))
        assert isinstance(stage, float)
        cell = teacher._index_of(np.asarray(final, dtype=np.float64))
        v_cont = float(teacher.value[cell])
        q = float(stage + teacher.discount * v_cont)
        successor[str(float(u))] = {
            "final_s_N": [float(x) for x in np.asarray(final).reshape(4)],
            "quantized_cell": list(cell),
            "integrated_stage_cost": float(stage),
            "continuation_value": v_cont,
            "bellman_Q": q,
        }
        cells.append(tuple(cell))
        v_list.append(v_cont)
        q_list.append(q)
    report["successor_diagnostics_at_+0.25"] = {
        "by_force": successor,
        "num_unique_successor_cells": int(len(set(cells))),
        "continuation_value_min": float(min(v_list)),
        "continuation_value_max": float(max(v_list)),
        "Q_min": float(min(q_list)),
        "Q_max": float(max(q_list)),
    }
    pass_c = len(set(cells)) >= 2 and (max(v_list) - min(v_list)) > 0.0 and (max(q_list) - min(q_list)) > 0.0

    # ------------------------------------------------------------------ D
    act_consistency = {}
    pass_d_items = []
    for name, state in reps.items():
        global_u, global_q, q_by_u = _global_discrete_argmin(teacher, state)
        candidates = _act_candidates(teacher, state)
        act_u = float(teacher.act(state))
        # Best among act() candidates
        cand_best_u = float(candidates[0])
        cand_best_q = float("inf")
        for u in candidates:
            q = float(teacher._bellman_cost(state, float(u)))
            if q < cand_best_q - 1e-12 or (abs(q - cand_best_q) <= 1e-12 and abs(u) < abs(cand_best_u)):
                cand_best_q = q
                cand_best_u = float(u)
        global_in_candidates = bool(np.any(np.isclose(candidates, global_u)))
        act_matches_candidate_argmin = abs(act_u - cand_best_u) <= 1e-9
        act_consistency[name] = {
            "global_discrete_argmin_u": global_u,
            "global_discrete_argmin_Q": global_q,
            "act_u": act_u,
            "act_candidate_set": [float(u) for u in candidates],
            "act_candidate_argmin_u": cand_best_u,
            "global_optimum_in_act_candidates": global_in_candidates,
            "act_matches_candidate_argmin": act_matches_candidate_argmin,
            "Q_all_forces": q_by_u,
        }
        pass_d_items.append(act_matches_candidate_argmin)
    report["act_consistency"] = act_consistency
    # D pass: act() is consistent with its documented candidate search (not necessarily global).
    pass_d = all(pass_d_items)
    # Also keep a lean-state Q check: act should beat u=0 when +0.25
    q0 = float(teacher._bellman_cost(s, 0.0))
    q_act = float(teacher._bellman_cost(s, rep_report["+0.25"]["act_u"]))
    report["Q_at_+0.25"] = {"u0": q0, "u_act": q_act, "act_beats_zero": bool(q_act < q0 - 1e-9)}
    pass_d = bool(pass_d and (q_act < q0 - 1e-9))

    # ------------------------------------------------------------------ E
    from ts_jepa.evaluation.metrics import control_score

    pos_tol = float(config.get("evaluation", {}).get("control_position_tol", 0.05))
    ang_tol = float(config.get("evaluation", {}).get("control_angle_tol", 0.05))
    xd = float(np.asarray(config["simulation"]["desired_state"], dtype=np.float64)[0])

    def _eq28_scores(states_arr: np.ndarray) -> dict[str, float]:
        scores = [
            control_score(s, desired_x=xd, position_tol=pos_tol, angle_tol=ang_tol)
            for s in np.asarray(states_arr)
        ]
        return {
            "mean_control_score": float(np.mean(scores)) if scores else float("nan"),
            "num_steps": int(len(scores)),
            "num_in_band": int(np.sum(scores)),
        }

    traj = env.rollout(teacher, steps=100, seed=10000)
    cmds = traj["commands"]
    states = traj["states"]
    finite = bool(np.isfinite(states).all() and np.isfinite(cmds).all())
    eq28 = _eq28_scores(states)
    report["rollout_seed_10000"] = {
        "command_min": float(cmds.min()),
        "command_max": float(cmds.max()),
        "command_mean": float(cmds.mean()),
        "command_std": float(cmds.std()),
        "command_unique": [float(u) for u in np.unique(cmds)],
        "max_abs_x": float(np.max(np.abs(states[:, 0]))),
        "max_abs_theta": float(np.max(np.abs(states[:, 2]))),
        "final_state": [float(x) for x in states[-1]],
        "theta_before": float(states[0, 2]),
        "theta_after": float(states[-1, 2]),
        "x_before": float(states[0, 0]),
        "x_after": float(states[-1, 0]),
        "numerically_bounded": finite,
        "eq28": eq28,
    }
    pass_e = bool(
        float(cmds.std()) > 0.0
        and not np.array_equal(np.unique(cmds), np.array([0.0]))
        and finite
    )

    # Extra perturbed seeds (sanity only)
    pert = []
    for seed in range(10000, 10010):
        tr = env.rollout(teacher, steps=100, seed=seed)
        pert.append(
            {
                "seed": seed,
                "theta_before": float(tr["states"][0, 2]),
                "theta_after": float(tr["states"][-1, 2]),
                "x_before": float(tr["states"][0, 0]),
                "x_after": float(tr["states"][-1, 0]),
                "max_abs_theta": float(np.max(np.abs(tr["states"][:, 2]))),
                "max_abs_x": float(np.max(np.abs(tr["states"][0:, 0]))),
                "command_mean": float(tr["commands"].mean()),
                "command_std": float(tr["commands"].std()),
                "command_unique": [float(u) for u in np.unique(tr["commands"])],
                "eq28_mean_control_score": _eq28_scores(tr["states"])["mean_control_score"],
            }
        )
    report["perturbed_rollouts"] = pert
    report["perturbed_eq28_mean"] = float(np.mean([p["eq28_mean_control_score"] for p in pert]))

    report["pass"] = {
        "A_policy_diversity": pass_a,
        "B_representative_dynamics_aligned": pass_b,
        "C_successor_distinguishable": pass_c,
        "D_act_consistency": pass_d,
        "E_rollout_nontrivial_commands": pass_e,
    }
    report["overall_pass"] = all(report["pass"].values())
    report["diagnostic_wall_time_s"] = round(time.perf_counter() - t0, 3)

    out = project_root(config) / "runs" / "eval" / "dp_teacher_diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
