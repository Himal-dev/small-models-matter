"""
Full-network live watcher for the Snake AI Laboratory.

The trainer saves snake_ai_lab_latest.pt at the end of every episode.
This server reads that checkpoint and serves an interactive dashboard that
shows the whole network (24 -> 128 -> 128 -> 3), its activations, weights,
biases and how they change over training.

Run from the repository root:
    python3 src/neuron_watch.py --port 8765

Then open:
    http://127.0.0.1:8765
"""

import argparse
import csv
import json
import logging
import math
import os
import pickle
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import snake_ai

LOGGER = logging.getLogger(__name__)

WEB_DIR = os.path.join(snake_ai.ROOT_DIR, "web")


def checkpoint_path():
    """Prefer the live training checkpoint, then the shipped demo model."""
    return snake_ai.resolve_model(snake_ai.LATEST_MODEL)


RELEASED_STATS = os.path.join(snake_ai.ROOT_DIR, "data", "snake_training_stats.csv")

FEATURE_NAMES = [
    "danger straight",
    "danger right",
    "danger left",
    "heading right",
    "heading down",
    "heading left",
    "heading up",
    "food left",
    "food right",
    "food up",
    "food down",
    "wall straight",
    "wall right",
    "wall left",
    "body straight",
    "body right",
    "body left",
    "food dx",
    "food dy",
    "snake length",
    "idle ratio",
    "same column",
    "same row",
    "score",
]

ACTION_NAMES = list(snake_ai.ACTION_NAMES.values())
LAYER_SIZES = snake_ai.LAYER_SIZES
MODULES = [0, 2, 4]

REWARD_TABLE = [
    {"event": "ATE FOOD", "value": 10.0, "why": "the real objective"},
    {"event": "DEATH", "value": -10.0, "why": "survival matters"},
    {"event": "CLOSER", "value": 0.05, "why": "dense hint toward food"},
    {"event": "FARTHER", "value": -0.02, "why": "small discouragement"},
    {"event": "MOVED", "value": 0.0, "why": "no preference"},
]

LEARNING_SETTINGS = {
    "gamma": snake_ai.GAMMA,
    "learning_rate": snake_ai.LEARNING_RATE,
    "batch_size": snake_ai.BATCH_SIZE,
    "memory_size": snake_ai.MEMORY_SIZE,
    "target_sync": snake_ai.TARGET_SYNC,
    "grad_clip": snake_ai.GRAD_CLIP,
    "rewards": REWARD_TABLE,
}

# Human labels for node layers: 0=input, 1=hidden1, 2=hidden2, 3=output
LAYER_LABELS = [
    "Input features",
    "Hidden layer 1",
    "Hidden layer 2",
    "Output (Q-values)",
]

SCENARIO_LABELS = {
    "auto": "Auto demo (live snake)",
    "start": "Game start",
    "mid": "Mid-game (scripted)",
}

STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/network.js": ("network.js", "application/javascript; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "application/javascript; charset=utf-8"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_stats_history(path, limit=1000):
    """Per-episode score and loss, for the training-progress charts."""
    try:
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
    except FileNotFoundError:
        return []

    history = []
    for row in rows[-limit:]:
        history.append({
            "episode": _as_int(row.get("episode")),
            "score": _as_int(row.get("score")),
            "average_score": _as_float(row.get("average_score")),
            "best_score": _as_int(row.get("best_score")),
            "epsilon": _as_float(row.get("epsilon")),
            "average_loss": _as_float(row.get("average_loss")),
        })
    return history


def _as_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value):
    number = _as_float(value)
    return int(number) if number is not None else None


def scripted_action(game):
    """Pick a safe action that reduces the Manhattan distance to the food."""
    candidates = []

    for action in (0, 1, 2):
        direction = game.direction
        if action == 1:
            direction = (direction + 1) % 4
        elif action == 2:
            direction = (direction - 1) % 4

        dx, dy = snake_ai.DIRS[direction]
        head = (game.snake[0][0] + dx, game.snake[0][1] + dy)

        if game.danger(head):
            continue

        if game.food is None:
            distance = 0
        else:
            distance = abs(head[0] - game.food[0]) + \
                       abs(head[1] - game.food[1])

        candidates.append((distance, action))

    if not candidates:
        return 0

    candidates.sort()
    return candidates[0][1]


def snapshot(game, state, action=None, q=None, event=""):
    return {
        "features": np.asarray(state, dtype=float).tolist(),
        "snake": [[int(x), int(y)] for x, y in game.snake],
        "food": [int(game.food[0]), int(game.food[1])]
                if game.food else None,
        "score": int(game.score),
        "steps": int(game.steps),
        "action": None if action is None else int(action),
        "q": None if q is None else np.asarray(q, dtype=float).tolist(),
        "event": event,
    }


def build_fixed_scenarios():
    """Two deterministic states so the page has something to show instantly."""
    scenarios = {}

    for name, seed, steps in (("start", 1234, 0), ("mid", 7, 18)):
        game = snake_ai.SnakeGame(random.Random(seed))
        state = game.reset()

        for _ in range(steps):
            action = scripted_action(game)
            state, _, done, _ = game.step(action)
            if done:
                state = game.reset()

        scenarios[name] = snapshot(game, state)

    return scenarios


# ---------------------------------------------------------------------------
# Demo game (separate from the trainer's pygame window)
# ---------------------------------------------------------------------------

class DemoRunner(threading.Thread):
    def __init__(self, watcher):
        super().__init__(daemon=True)
        self.watcher = watcher
        self.lock = threading.Lock()
        self.game = snake_ai.SnakeGame()
        self.state = self.game.get_state()
        self.frame = None
        self.stopped = threading.Event()

    def run(self):
        while not self.stopped.is_set():
            self.watcher.refresh()
            with self.watcher.lock:
                version = self.watcher.version
                net = self.watcher.net
            if net is not None:
                self.step(net, version)
            self.stopped.wait(0.30)

    @torch.inference_mode()
    def step(self, net, version):
        q = net(torch.from_numpy(self.state)).numpy()
        action = int(np.argmax(q))
        board = snapshot(self.game, self.state, action, q)
        next_state, reward, done, info = self.game.step(action)
        board["event"] = info["event"]
        next_q = net(torch.from_numpy(next_state)).numpy()
        best_future = float(np.max(next_q))
        predicted = float(q[action])
        target = reward if done else reward + snake_ai.GAMMA * best_future
        error = target - predicted
        transition = {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "reward": reward,
            "event": info["event"],
            "done": done,
            "q": q.tolist(),
            "q_next": next_q.tolist(),
            "best_future": best_future,
            "target": target,
            "predicted": predicted,
            "td_error": error,
            "squared_error": error ** 2,
        }
        self.state = self.game.reset() if done else next_state
        with self.lock:
            self.frame = (board, transition, version)

    def snapshot(self):
        with self.lock:
            return self.frame


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------

class Watcher:
    def __init__(self, total_episodes):
        self.lock = threading.Lock()
        self.state = None
        self.prev_state = None
        self.refresh_lock = threading.Lock()
        self.version = None
        self.error = None
        self.history = {}
        self.layer_history = []
        self.training_history = []
        self.total_episodes = total_episodes
        self.net = None
        self.layer_stats = []
        self.background_edges = []
        self.demo = None
        self.scenarios = build_fixed_scenarios()

        self.refresh(force=True)

    # -- checkpoint handling ------------------------------------------------

    def refresh(self, force=False):
        with self.refresh_lock:
            return self._refresh(force)

    def _refresh(self, force):
        path = checkpoint_path()
        try:
            stat = os.stat(path)
        except OSError:
            return False

        source = "training" if path == snake_ai.LATEST_MODEL else "shipped"
        version = (str(stat.st_mtime_ns), stat.st_size, source)
        if not force and version == self.version:
            return False

        try:
            with open(path, "rb") as handle:
                opened = os.fstat(handle.fileno())
                if (stat.st_ino, stat.st_mtime_ns) != (opened.st_ino, opened.st_mtime_ns):
                    return False
                state = torch.load(handle, map_location="cpu", weights_only=True)
            net = snake_ai.QNetwork().eval()
            net.load_state_dict(state)
            if not all(torch.isfinite(parameter).all() for parameter in net.parameters()):
                raise ValueError("checkpoint contains non-finite parameters")
            history = read_stats_history(
                snake_ai.STATS_FILE if source == "training" else RELEASED_STATS
            )
        except (
            OSError, EOFError, pickle.UnpicklingError, RuntimeError,
            TypeError, ValueError, csv.Error,
        ) as exc:
            with self.lock:
                self.error = f"checkpoint read failed (retrying): {exc}"
            return False

        with self.lock:
            self.prev_state = self.state
            self.state = state
            self.net = net
            self.version = version
            self.error = None
            self.training_history = history
            self.layer_stats = self._layer_stats()
            self.background_edges = self._background_edges()
            self._record_history()

        return True

    def _episode_number(self):
        return (self.training_history[-1]["episode"] or 0) if self.training_history else 0

    def _weights_bias(self, layer, neuron, state=None):
        if state is None:
            state = self.state
        if state is None:
            return None, None

        module = MODULES[layer]
        weight_key = f"layers.{module}.weight"
        bias_key = f"layers.{module}.bias"
        if weight_key not in state:
            return None, None

        weights = state[weight_key][neuron].numpy().astype(float)
        bias = float(state[bias_key][neuron])
        return weights, bias

    # -- history ------------------------------------------------------------

    def _record_history(self):
        episode = self._episode_number()

        for key in list(self.history):
            layer, neuron = key
            weights, bias = self._weights_bias(layer, neuron)
            if weights is None:
                continue

            self.history[key].append({
                "episode": episode,
                "bias": float(bias),
                "weight_norm": float(np.linalg.norm(weights)),
            })
            self.history[key] = self.history[key][-240:]

        self.layer_history.append({
            "episode": episode,
            "mean_abs": [item["mean_abs"] for item in self.layer_stats],
            "max_abs": [item["max_abs"] for item in self.layer_stats],
        })
        self.layer_history = self.layer_history[-240:]

    # -- forward pass -------------------------------------------------------

    def _forward(self, features):
        net = self.net
        tensor = torch.from_numpy(
            np.asarray(features, dtype=np.float32)
        ).unsqueeze(0)

        with torch.no_grad():
            a0 = tensor
            z1 = net.layers[0](a0)
            a1 = torch.relu(z1)
            z2 = net.layers[2](a1)
            a2 = torch.relu(z2)
            q = net.layers[4](a2)

        activations = [
            a0[0].numpy().astype(float),
            a1[0].numpy().astype(float),
            a2[0].numpy().astype(float),
            q[0].numpy().astype(float),
        ]
        zs = [
            None,
            z1[0].numpy().astype(float),
            z2[0].numpy().astype(float),
            q[0].numpy().astype(float),
        ]
        return activations, zs

    # -- stats --------------------------------------------------------------

    def _layer_stats(self, state=None):
        if state is None:
            state = self.state

        result = []
        for module in MODULES:
            weights = state[f"layers.{module}.weight"].numpy()
            biases = state[f"layers.{module}.bias"].numpy()
            result.append({
                "mean_abs": float(np.mean(np.abs(weights))),
                "max_abs": float(np.max(np.abs(weights))),
                "bias_mean": float(np.mean(biases)),
                "bias_abs_mean": float(np.mean(np.abs(biases))),
            })
        return result

    # -- graph edges --------------------------------------------------------

    def _background_edges(self):
        edges = []
        for layer, module in enumerate(MODULES):
            for target, row in enumerate(self.state[f"layers.{module}.weight"].numpy()):
                edges.extend(
                    {"f": layer, "s": int(source), "t": target, "w": float(row[source])}
                    for source in np.argsort(-np.abs(row))[:2]
                )
        return edges

    def _edges(self, layer, neuron, weights):
        # The sparse view omits edges already highlighted by the inspector.
        edges = [
            edge for edge in self.background_edges
            if (edge["f"], edge["t"]) != (layer, neuron)
            and (edge["f"], edge["s"]) != (layer + 1, neuron)
        ]
        focus_in = [{"s": index, "w": weight} for index, weight in enumerate(weights)]
        focus_out = []
        if layer + 1 < len(MODULES):
            next_weights = self.state[f"layers.{MODULES[layer + 1]}.weight"][:, neuron]
            focus_out = [
                {"t": index, "w": weight}
                for index, weight in enumerate(next_weights.tolist())
            ]
        return edges, focus_in, focus_out

    # -- selected neuron ----------------------------------------------------

    def _selected(self, layer, neuron, activations):
        weights, bias = self._weights_bias(layer, neuron)
        inputs = activations[layer]

        if layer == 0:
            names = list(FEATURE_NAMES)
        else:
            names = [f"h{layer}[{i}]" for i in range(len(inputs))]

        z = float(np.dot(weights, inputs) + bias)
        activation = z if layer == 2 else max(0.0, z)

        prev_weights, prev_bias = (
            self._weights_bias(layer, neuron, self.prev_state)
            if self.prev_state is not None else (None, None)
        )

        if prev_weights is None:
            weight_deltas = None
            bias_delta = None
            norm_delta = None
        else:
            weight_deltas = (weights - prev_weights).tolist()
            bias_delta = float(bias - prev_bias)
            norm_delta = float(
                np.linalg.norm(weights) - np.linalg.norm(prev_weights)
            )

        return {
            "layer": layer,
            "neuron": neuron,
            "layer_label": LAYER_LABELS[layer + 1],
            "input_label": LAYER_LABELS[layer],
            "is_output": layer == 2,
            "bias": float(bias),
            "bias_delta": bias_delta,
            "weight_norm": float(np.linalg.norm(weights)),
            "weight_norm_delta": norm_delta,
            "weights": weights.tolist(),
            "weight_deltas": weight_deltas,
            "input_names": names,
            "inputs": inputs.tolist(),
            "contributions": (weights * inputs).tolist(),
            "z": z,
            "activation": activation,
        }

    # -- API payload --------------------------------------------------------

    def payload(self, layer, neuron, scenario_name, edge_version=None):
        self.refresh()

        with self.lock:
            if self.state is None:
                return {
                    "ok": False,
                    "error": self.error or "waiting for the first checkpoint",
                }

            scenario = self.scenarios.get(scenario_name, self.scenarios["start"])
            demo_transition = None
            if scenario_name == "auto" and self.demo is not None:
                frame = self.demo.snapshot()
                if frame is not None and frame[2] == self.version:
                    scenario, demo_transition, _ = frame

            activations, zs = self._forward(scenario["features"])
            selected = self._selected(layer, neuron, activations)
            edges, focus_in, focus_out = self._edges(layer, neuron, selected["weights"])

            layer_stats = self.layer_stats
            prev_stats = (
                self._layer_stats(self.prev_state)
                if self.prev_state is not None
                else None
            )

            stats_out = []
            for index, item in enumerate(layer_stats):
                delta = None
                if prev_stats is not None:
                    delta = {
                        key: item[key] - prev_stats[index][key]
                        for key in item
                    }
                stats_out.append({**item, "delta": delta})

            biases = [
                None,
                self.state["layers.0.bias"].numpy().astype(float).tolist(),
                self.state["layers.2.bias"].numpy().astype(float).tolist(),
                self.state["layers.4.bias"].numpy().astype(float).tolist(),
            ]

            key = (layer, neuron)
            if key not in self.history:
                self.history[key] = [{
                    "episode": self._episode_number(),
                    "bias": selected["bias"],
                    "weight_norm": selected["weight_norm"],
                }]

            stats = self.training_history[-1] if self.training_history else {}
            age = max(0, time.time() - int(self.version[0]) / 1e9)

            chosen_action = int(np.argmax(activations[3]))
            active = {
                "h1": int(np.sum(activations[1] > 0)),
                "h2": int(np.sum(activations[2] > 0)),
                "output": chosen_action,
            }

            board = {
                key: value
                for key, value in scenario.items()
                if key != "features"
            }

            return {
                "ok": True,
                "version": list(self.version),
                "all_weights": (
                    [self.state[f"layers.{module}.weight"].tolist()
                     for module in MODULES]
                    if edge_version is not None
                    and edge_version != ",".join(map(str, self.version))
                    else None
                ),
                "checkpoint_age": age,
                "error": self.error,
                "episode": self._episode_number(),
                "total_episodes": self.total_episodes,
                "stats": {
                    key: stats.get(key) for key in (
                        "score", "average_score", "best_score", "epsilon", "average_loss"
                    )
                },
                "sizes": LAYER_SIZES,
                "layer_labels": LAYER_LABELS,
                "action_names": ACTION_NAMES,
                "parameter_counts": [
                    LAYER_SIZES[0] * LAYER_SIZES[1] + LAYER_SIZES[1],
                    LAYER_SIZES[1] * LAYER_SIZES[2] + LAYER_SIZES[2],
                    LAYER_SIZES[2] * LAYER_SIZES[3] + LAYER_SIZES[3],
                ],
                "scenario": scenario_name,
                "scenario_label": SCENARIO_LABELS.get(
                    scenario_name,
                    scenario_name,
                ),
                "board": board,
                "activations": [item.tolist() for item in activations],
                "zs": [
                    None if item is None else item.tolist()
                    for item in zs
                ],
                "biases": biases,
                "active": active,
                "chosen_action": chosen_action,
                "selected": selected,
                "edges": edges,
                "focus_in": focus_in,
                "focus_out": focus_out,
                "layer_stats": stats_out,
                "history": list(self.history[key]),
                "layer_history": list(self.layer_history),
                "training_history": self.training_history,
                "learning": {
                    **LEARNING_SETTINGS,
                    "epsilon": stats.get("epsilon"),
                    "average_loss": stats.get("average_loss"),
                    "demo": demo_transition,
                },
            }


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    watcher = None

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in STATIC_FILES:
            name, content_type = STATIC_FILES[parsed.path]
            path = os.path.join(WEB_DIR, name)
            try:
                with open(path, "rb") as handle:
                    body = handle.read()
            except OSError:
                self._json(
                    {"ok": False, "error": f"missing {name}"},
                    500,
                )
                return

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if parsed.path == "/api/network":
            query = parse_qs(parsed.query)

            try:
                layer = int(query.get("layer", ["0"])[0])
                neuron = int(query.get("neuron", ["0"])[0])
            except ValueError:
                self._json(
                    {"ok": False, "error": "bad layer/neuron"},
                    400,
                )
                return

            scenario = query.get("scenario", ["auto"])[0]

            if layer not in (0, 1, 2):
                self._json(
                    {"ok": False, "error": "layer must be 0, 1 or 2"},
                    400,
                )
                return

            max_neuron = 2 if layer == 2 else 127
            if neuron < 0 or neuron > max_neuron:
                self._json(
                    {"ok": False, "error": f"neuron must be 0..{max_neuron}"},
                    400,
                )
                return

            if scenario not in SCENARIO_LABELS:
                self._json(
                    {"ok": False, "error": "unknown scenario"},
                    400,
                )
                return

            try:
                edge_version = query.get("edge_version", [None])[0]
                payload = self.watcher.payload(layer, neuron, scenario, edge_version)
            except Exception:
                LOGGER.exception("Failed to build network response")
                self._json({"ok": False, "error": "Could not build network response"}, 500)
                return

            self._json(payload, 200)
            return

        self._json({"ok": False, "error": "not found"}, 404)

    def _json(self, payload, status):
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Full-network live watcher",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    watcher = Watcher(args.episodes)
    demo = DemoRunner(watcher)
    watcher.demo = demo
    Handler.watcher = watcher

    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        demo.start()
        print(f"Network watcher running at http://127.0.0.1:{server.server_port}")
        print(f"Watching {checkpoint_path()} (updated once per episode)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            demo.stopped.set()
            demo.join()


if __name__ == "__main__":
    main()
