"""Build the static dashboard from the published checkpoint and training log."""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import torch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import neuron_watch as watch  # noqa: E402
import snake_ai  # noqa: E402


def replay(model, length=120):
    game = snake_ai.SnakeGame(random.Random(7))
    frames = []
    for _ in range(length):
        features = game.get_state()
        action = int(model(torch.from_numpy(features)).argmax())
        frame = watch.snapshot(game, features)
        next_state, reward, done, info = game.step(action)
        frame.update(
            next_features=next_state.tolist(),
            reward=reward,
            done=done,
            event=info["event"],
        )
        frames.append(frame)
        if done:
            game.reset()
    return frames


def build(output):
    model = snake_ai.QNetwork().eval()
    state = torch.load(
        ROOT / "models" / "snake_ai_lab_latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    with torch.inference_mode():
        frames = replay(model)
    history = watch.read_stats_history(watch.RELEASED_STATS)
    if not history:
        raise ValueError("The static demo requires the released training log")
    data = {
        "weights": [state[f"layers.{layer}.weight"].tolist() for layer in watch.MODULES],
        "biases": [None] + [state[f"layers.{layer}.bias"].tolist() for layer in watch.MODULES],
        "sizes": watch.LAYER_SIZES,
        "layer_labels": watch.LAYER_LABELS,
        "action_names": watch.ACTION_NAMES,
        "scenarios": watch.build_fixed_scenarios(),
        "frames": frames,
        "history": history,
        "learning": watch.LEARNING_SETTINGS,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name in ("dashboard.css", "network.js", "dashboard.js", "replay.js"):
        shutil.copyfile(ROOT / "web" / name, output / name)
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    html = html.replace('data-mode="live"', 'data-mode="replay"')
    html = html.replace("Auto demo (live snake)", "Recorded game replay")
    html = html.replace(
        '<script src="./dashboard.js"></script>',
        '<script src="./replay.js"></script>\n<script src="./dashboard.js"></script>',
    )
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "demo.json").write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    (output / ".nojekyll").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    build(parser.parse_args().output)
