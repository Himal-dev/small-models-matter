"""Evaluate the released DQN and a greedy rule on repeatable Snake games."""

import json
import os
import random
import sys
from pathlib import Path
from statistics import mean, median

import torch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuron_watch import scripted_action  # noqa: E402
from snake_ai import QNetwork, SnakeGame  # noqa: E402


def evaluate(policy, seeds, step_limit):
    scores = []
    capped = 0
    for seed in seeds:
        game = SnakeGame(random.Random(seed))
        state = game.get_state()
        for _ in range(step_limit):
            state, _, done, _ = game.step(policy(game, state))
            if done:
                break
        else:
            capped += 1
        scores.append(game.score)
    return {
        "mean_score": mean(scores),
        "median_score": median(scores),
        "best_score": max(scores),
        "capped_games": capped,
    }


def main():
    torch.set_num_threads(1)
    model = QNetwork().eval()
    model.load_state_dict(torch.load(
        ROOT / "models" / "snake_ai_lab_latest.pt",
        map_location="cpu",
        weights_only=True,
    ))

    def dqn(_game, state):
        return int(model(torch.from_numpy(state)).argmax())

    seeds = range(10_000, 10_100)
    step_limit = 5_000
    with torch.inference_mode():
        results = {
            "seed_start": seeds.start,
            "games_per_policy": len(seeds),
            "step_limit": step_limit,
            "dqn": evaluate(dqn, seeds, step_limit),
            "greedy_manhattan": evaluate(
                lambda game, _state: scripted_action(game), seeds, step_limit
            ),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
