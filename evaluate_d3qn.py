"""Deterministically evaluate D3QN checkpoints without continuous Actor loading."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from D3QN import D3QNAgent, DuelingQNetwork
from World import World
from discrete_actions import DiscreteActionCodec
from evaluate_policy import plot_trajectory, rollout


ROOT = Path(__file__).resolve().parent
D3QN_STATE_SCHEMA = (
    "v4: uav xy, target-minus-uav xy, pending payload, outage usage, "
    "A2G SNR, G2A outage probability"
)
D3QN_G2A_STATE_SCHEMA = (
    "v4: uav xy, target-minus-uav xy, pending payload, outage usage, "
    "G2A outage probability"
)
ACTION_SCHEMA = "hover + 16 relative headings at 0.125 + 16 relative headings at 0.25"
REQUIRED_CONFIG_FIELDS = {
    "algorithm", "data_level", "episodes", "map_hashes", "net_width", "order",
    "outage_max_s", "payloads_mb", "state_dim", "state_schema", "uav_h", "variant",
    "action_count", "action_schema",
}


@dataclass(frozen=True)
class D3QNRunContext:
    run_dir: Path
    config: dict
    algorithm: str
    state_dim: int
    net_width: int
    variant: str
    action_count: int
    order: tuple
    payloads_mb: np.ndarray
    outage_max_s: float
    total_steps: int = 1000
    length: float = 20.0
    width: float = 20.0
    delta_t: float = 0.5
    dist_max: float = 0.25


def _map_hashes():
    data_dir = ROOT / "results" / "datas"
    return {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ("Radio_datas.npz", "Radio_datas_A2G.npz")
    }


def load_d3qn_run_context(run_dir):
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"run_config.json not found in {run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_CONFIG_FIELDS - config.keys())
    if missing:
        raise ValueError(f"D3QN run_config.json missing required fields: {', '.join(missing)}")
    if config["algorithm"] != "d3qn":
        raise ValueError("D3QN evaluator only accepts algorithm='d3qn'")
    a2g_blind = bool(config.get("a2g_blind", False))
    expected_dim = 7 if a2g_blind else 8
    expected_schema = D3QN_G2A_STATE_SCHEMA if a2g_blind else D3QN_STATE_SCHEMA
    if config["variant"] != "A" or int(config["state_dim"]) != expected_dim:
        raise ValueError(f"D3QN baseline requires Variant A and the {expected_dim}D observation")
    if config["state_schema"] != expected_schema:
        raise ValueError("D3QN state schema is incompatible with the current World")
    if int(config["action_count"]) != 33 or config["action_schema"] != ACTION_SCHEMA:
        raise ValueError("D3QN action schema is incompatible with the 33-action codec")
    order = tuple(int(value) for value in config["order"])
    payloads = np.asarray(config["payloads_mb"], dtype=float)
    if len(order) != len(payloads) or sorted(order) != list(range(1, len(payloads) + 1)):
        raise ValueError("D3QN run_config.json has inconsistent order and payloads")
    if config["map_hashes"] != _map_hashes():
        raise ValueError("fixed radio maps do not match this D3QN run")
    return D3QNRunContext(
        run_dir=run_dir,
        config=config,
        algorithm="d3qn",
        state_dim=expected_dim,
        net_width=int(config["net_width"]),
        variant="A",
        action_count=33,
        order=order,
        payloads_mb=payloads,
        outage_max_s=float(config["outage_max_s"]),
    )


def d3qn_checkpoint_paths(run_dir):
    model_dir = Path(run_dir) / "models"
    return {
        "final": model_dir / "d3qn_final.pt",
        "training_best": model_dir / "d3qn_training_best.pt",
        "eval_best": model_dir / "d3qn_eval_best.pt",
    }


def load_d3qn_checkpoint(checkpoint, state_dim, action_count, net_width):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"D3QN checkpoint not found: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != D3QNAgent.CHECKPOINT_SCHEMA
        or int(payload.get("state_dim", -1)) != int(state_dim)
        or int(payload.get("action_dim", -1)) != int(action_count)
        or int(payload.get("hidden_dim", -1)) != int(net_width)
    ):
        raise ValueError(f"D3QN checkpoint is incompatible with run_config.json: {checkpoint}")
    network = DuelingQNetwork(state_dim, action_count, net_width).cpu()
    try:
        network.load_state_dict(payload["online_state_dict"], strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"D3QN checkpoint is incompatible with run_config.json: {checkpoint}") from error
    network.eval()
    return network, dict(payload.get("metadata", {}))


def _build_world(context):
    return World(
        length=context.length,
        width=context.width,
        uav_num=1,
        dist_max=context.dist_max,
        delta_t=context.delta_t,
        t=context.total_steps,
        uav_h=float(context.config["uav_h"]),
        data_size=300,
        data_sizes=context.payloads_mb,
        outage_max_s=context.outage_max_s,
        ini_loc=[2.62, 2.65],
        end_loc=[18.95, 11.12],
        users_name=str(ROOT / "results" / "datas" / "Users_6.txt"),
        BS_loc=np.array(
            [[4.77, 4.56, 0.25], [14.89, 4.89, 0.25],
             [4.63, 13.78, 0.25], [13.48, 14.89, 0.25]]
        ),
        traverse_sequence=list(context.order),
        variant="A",
        include_a2g_snr=not bool(context.config.get("a2g_blind", False)),
    )


def evaluate_checkpoint(context, checkpoint):
    network, metadata = load_d3qn_checkpoint(
        checkpoint, context.state_dim, context.action_count, context.net_width
    )
    codec = DiscreteActionCodec()

    def act(state):
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action_index = int(network(tensor).argmax(dim=1).item())
        return codec.decode(action_index)

    report, trace = rollout(context, _build_world(context), act)
    report["checkpoint"] = str(Path(checkpoint).resolve())
    report["checkpoint_metadata"] = metadata
    report["policy_mode"] = "deterministic D3QN argmax (epsilon=0, no MapGuide)"
    return report, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    context = load_d3qn_run_context(args.run_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else context.run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for label, checkpoint in d3qn_checkpoint_paths(context.run_dir).items():
        if not checkpoint.is_file():
            continue
        report, trace = evaluate_checkpoint(context, checkpoint)
        target = output_dir / label
        target.mkdir(parents=True, exist_ok=True)
        (target / "metrics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        np.savez(target / "trajectory.npz", **trace)
        plot_trajectory(context, trace, target / "trajectory_dual_maps",
                        f"D3QN {label}: {report['termination_reason']}")
        reports[label] = report
    if not reports:
        raise FileNotFoundError(f"no D3QN checkpoints found in {context.run_dir / 'models'}")
    summary = {
        "run_dir": str(context.run_dir),
        "policy_mode": "deterministic D3QN argmax (epsilon=0, no MapGuide)",
        "evaluation": reports,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
