"""Measure batch-one inference and headless game-step latency."""

import argparse
import json
import math
import os
import platform
import random
import statistics
import sys
import time

import torch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import snake_ai  # noqa: E402


def percentile(sorted_values, fraction):
    return sorted_values[math.ceil(len(sorted_values) * fraction) - 1]


def measure(fn, runs):
    for _ in range(min(1000, runs)):
        fn()

    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - start)

    samples.sort()
    return {
        "mean_us": statistics.mean(samples) / 1000.0,
        "median_us": statistics.median(samples) / 1000.0,
        "p99_us": percentile(samples, 0.99) / 1000.0,
        "min_us": samples[0] / 1000.0,
        "max_us": samples[-1] / 1000.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the Snake DQN decision loop.",
    )
    parser.add_argument("--runs", type=int, default=20000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.threads < 1:
        parser.error("--runs and --threads must be positive")

    torch.set_num_threads(args.threads)

    model_path = snake_ai.resolve_model(snake_ai.LATEST_MODEL)
    if not os.path.exists(model_path):
        raise SystemExit(
            "No checkpoint found. Train first:\n"
            "    python3 src/snake_ai.py train --episodes 500"
        )

    net = snake_ai.QNetwork()
    net.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    net.eval()

    parameters = sum(p.numel() for p in net.parameters())
    seed = 7
    game = snake_ai.SnakeGame(random.Random(seed))
    sample = torch.from_numpy(game.get_state()).unsqueeze(0)

    def forward_only():
        with torch.no_grad():
            net(sample)

    def forward_argmax():
        with torch.no_grad():
            net(sample).argmax(dim=1)

    def decision_only():
        current = game.get_state()
        tensor = torch.from_numpy(current).float().unsqueeze(0)
        with torch.no_grad():
            net(tensor).argmax(dim=1)

    def full_loop():
        state = game.get_state()
        tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action = int(net(tensor).argmax(dim=1).item())
        _, _, done, _ = game.step(action)
        if done:
            game.reset()

    results = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": args.threads,
        },
        "model": {
            "path": model_path,
            "parameters": parameters,
            "bytes_fp32": parameters * 4,
            "inputs": snake_ai.LAYER_SIZES[0],
            "outputs": snake_ai.LAYER_SIZES[-1],
        },
        "runs_per_stage": args.runs,
        "seed": seed,
        "forward_only": measure(forward_only, args.runs),
        "forward_argmax": measure(forward_argmax, args.runs),
        "decision_only": measure(decision_only, args.runs),
        "full_loop": measure(full_loop, args.runs),
    }

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"Model:      {model_path}")
    print(
        f"Parameters: {parameters:,} "
        f"({parameters * 4 / 1024:.1f} KiB at fp32)"
    )
    print(
        f"Machine:    {results['machine']['platform']} | "
        f"torch {torch.__version__} | threads {args.threads}"
    )
    print()

    rows = (
        ("forward_only", "network forward"),
        ("forward_argmax", "forward + argmax"),
        ("decision_only", "features + forward + argmax"),
        ("full_loop", "full loop incl. game.step"),
    )
    for key, label in rows:
        item = results[key]
        print(
            f"{label:32s} "
            f"mean {item['mean_us']:7.1f} us | "
            f"median {item['median_us']:7.1f} us | "
            f"p99 {item['p99_us']:7.1f} us"
        )

    print()
    print(
        "Throughput: "
        f"{1_000_000 / results['decision_only']['mean_us']:,.0f} "
        f"decisions/second ({args.threads} CPU thread{'s' if args.threads != 1 else ''})"
    )


if __name__ == "__main__":
    main()
