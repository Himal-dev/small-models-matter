"""Smoke tests: the contract that makes the demo teachable."""

import os

import numpy as np
import torch

import neuron_watch
import snake_ai


def test_parameter_count():
    net = snake_ai.QNetwork()
    total = sum(p.numel() for p in net.parameters())
    assert total == 20_099


def test_feature_vector():
    game = snake_ai.SnakeGame()
    state = game.reset()

    assert state.shape == (24,)
    assert state.dtype == np.float32
    assert np.all(state >= 0.0)


def test_step_contract():
    game = snake_ai.SnakeGame()
    game.reset()

    next_state, reward, done, info = game.step(0)

    assert next_state.shape == (24,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "event" in info
    assert "action" in info


def test_filling_the_board_returns_a_valid_terminal_state():
    game = snake_ai.SnakeGame()
    game.food = (1, 0)
    game.snake = [(0, 0)] + [
        (x, y) for x in range(snake_ai.GRID) for y in range(snake_ai.GRID)
        if (x, y) not in ((0, 0), game.food)
    ]

    state, reward, done, info = game.step(0)

    assert done and info["event"] == "WIN"
    assert reward == 10 and game.food is None
    assert state.shape == (24,) and state.dtype == np.float32
    assert not state[[7, 8, 9, 10, 17, 18, 21, 22]].any()


def test_forward_shape():
    net = snake_ai.QNetwork()
    out = net(torch.zeros(1, 24))
    assert out.shape == (1, 3)


def test_replay_evicts_oldest_and_still_trains(monkeypatch):
    monkeypatch.setattr(snake_ai, "MEMORY_SIZE", 3)
    monkeypatch.setattr(snake_ai, "BATCH_SIZE", 2)
    agent = snake_ai.Agent()
    state = snake_ai.SnakeGame().get_state()
    for reward in range(4):
        agent.remember(state, 0, reward, state, True)

    assert sorted(transition[2] for transition in agent.memory) == [1, 2, 3]
    assert np.isfinite(agent.learn())
    assert agent.learn_steps == 1


def test_shipped_model_loads():
    path = snake_ai.resolve_model(snake_ai.LATEST_MODEL)
    assert os.path.exists(path), "models/snake_ai_lab_latest.pt should ship"

    state = torch.load(path, map_location="cpu", weights_only=True)
    net = snake_ai.QNetwork()
    net.load_state_dict(state)


def test_full_network_and_live_checkpoint_switch(tmp_path, monkeypatch):
    latest = str(tmp_path / "snake_ai_lab_latest.pt")
    monkeypatch.setattr(snake_ai, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(snake_ai, "LATEST_MODEL", latest)
    monkeypatch.setattr(snake_ai, "BEST_MODEL", str(tmp_path / "best.pt"))
    monkeypatch.setattr(snake_ai, "STATS_FILE", str(tmp_path / "training.csv"))

    watcher = neuron_watch.Watcher(total_episodes=1)
    shipped = watcher.payload(1, 0, "start", edge_version="initial")
    assert sum(shipped["sizes"]) == 283
    assert sum(len(row) for matrix in shipped["all_weights"] for row in matrix) == 19_840
    assert len(shipped["training_history"]) == 500
    assert shipped["selected"]["weight_deltas"] is None
    version = ",".join(map(str, shipped["version"]))
    assert watcher.payload(1, 0, "start", edge_version=version)["all_weights"] is None

    snake_ai.train(1, False)
    live = watcher.payload(1, 0, "start", edge_version=version)
    assert live["version"][2] == "training"
    assert live["episode"] == len(live["training_history"]) == 1
    assert live["all_weights"] is not None
    assert len(live["selected"]["weight_deltas"]) == 128
    assert live["selected"]["bias_delta"] is not None

    torch.save({}, latest)
    failed = watcher.payload(1, 0, "start")
    assert failed["ok"]
    assert failed["version"] == live["version"]
    assert "checkpoint read failed" in failed["error"]

    snake_ai.train(1, False)
    recovered = watcher.payload(1, 0, "start")
    assert recovered["version"] != live["version"]
    assert recovered["error"] is None


def test_live_demo_reports_one_consistent_transition():
    watcher = neuron_watch.Watcher(total_episodes=500)
    watcher.demo = neuron_watch.DemoRunner(watcher)
    watcher.demo.step(watcher.net, watcher.version)

    payload = watcher.payload(2, 0, "auto")
    transition = payload["learning"]["demo"]
    assert payload["board"]["steps"] == 0
    assert payload["board"]["action"] == payload["chosen_action"] == transition["action"]
    np.testing.assert_allclose(transition["q"], payload["activations"][-1], atol=1e-6)
