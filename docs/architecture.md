# Architecture

This document explains exactly what the code does, with the numbers from the
implementation. If you only read one section, read **The update**.

## 1. The problem as a Markov decision process

| Element | In this project |
|---|---|
| State `s` | 24 hand-designed features (below) |
| Action `a` | `0 = straight`, `1 = turn right`, `2 = turn left` |
| Reward `r` | `+10` food, `-10` death, `+0.05` closer, `-0.02` farther, `0` otherwise |
| Next state `s'` | the state after the action |
| Done | wall, body collision, win, or idle timeout |
| Discount `γ` | `0.90` |

The agent is trained with **Deep Q-Learning** (DQN). It learns `Q(s, a)`, the
expected discounted return from taking action `a` in state `s` and then acting
greedily.

## 2. State: 24 features

The network never sees the 20×20 grid directly. It sees a vector of 24
normalized numbers built by `SnakeGame.get_features()`.

| Index | Feature | Meaning |
|---:|---|---|
| 0–2 | danger straight / right / left | 1 if the next cell in that direction is fatal |
| 3–6 | heading right / down / left / up | one-hot current direction |
| 7–10 | food left / right / up / down | which side of the head the food is on |
| 11–13 | wall distance straight / right / left | normalized distance to the wall |
| 14–16 | body distance straight / right / left | normalized distance to the body |
| 17–18 | food dx / dy | normalized absolute distance to food |
| 19 | snake length | normalized by grid area |
| 20 | idle ratio | steps since food / idle limit |
| 21–22 | same column / same row | 1 if the head and food share an x / y coordinate |
| 23 | score | normalized by grid area |

This is a **feature-engineered** state, not pixels. That is deliberate: the
point of the demo is to make the network legible, not to solve perception.

## 3. Network

```
Linear(24, 128) -> ReLU
Linear(128, 128) -> ReLU
Linear(128, 3)
```

| Layer | Weights | Biases | Parameters |
|---|---:|---:|---:|
| 24 → 128 | 3,072 | 128 | 3,200 |
| 128 → 128 | 16,384 | 128 | 16,512 |
| 128 → 3 | 384 | 3 | 387 |
| **Total** | | | **20,099** |

At fp32 that is **78.5 KiB** of weights and biases. Adam keeps two additional
moments per parameter, roughly doubling the model size in optimizer state
alone.

The output is three Q-values. The policy is `argmax`. They are not
probabilities and can be any real number.

## 4. The update

For a transition `(s, a, r, s', done)`:

```
target = r + γ · max_a' Q_target(s', a') · (1 - done)
loss   = ( Q_online(s, a) - target )²
```

A batch of 128 transitions is sampled uniformly from a replay buffer of up to
100,000 transitions. The loss is averaged over the batch.

```
all_q   = Q_online(states)                 # (128, 3)
pred    = all_q.gather(1, actions)         # (128,) chosen action only
next_q  = Q_target(next_states)            # (128, 3), no gradient
target  = rewards + γ · next_q.max(1) · (1 - dones)
loss    = mse_loss(pred, target)
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(online.parameters(), 1.0)
optimizer.step()                           # Adam, lr = 0.001
```

Key details:

- Only the Q-value of the action actually taken is in the loss for that
  sample. The other two outputs get gradients from other samples where they
  were chosen.
- The target network is a frozen copy of the online network. It is refreshed
  every **200** optimizer steps.
- Gradients are clipped at norm **1.0**.
- The optimizer is Adam with learning rate **0.001**.
- One optimizer step happens after every game step, once the replay buffer
  has at least 128 transitions.

### Why each stabilizer exists

| Component | Without it |
|---|---|
| Replay buffer | Consecutive frames are highly correlated; the network overfits recent experience. |
| Target network | The target moves every step, so the network chases itself. |
| Gradient clipping | A single large TD error can dominate an update. |
| Discount factor | Without discounting, the agent treats far-future reward as equal to immediate reward. |

## 5. Exploration

```
epsilon = max(0.02, 1 - episode / (episodes * 0.4))
```

For 500 episodes, epsilon starts at 0.995 and reaches 0.02 at episode 196,
then stays at 0.02. With probability epsilon the agent picks a random action;
otherwise it picks `argmax Q`. At play time, epsilon is 0.

## 6. Training loop

```
for episode in 1..E:
    state = game.reset()
    epsilon = schedule(episode)

    while not done:
        action = epsilon_greedy(Q_online, state, epsilon)
        next_state, reward, done, info = game.step(action)

        replay.append((state, action, reward, next_state, done))
        loss = learn_from_random_batch(replay)   # one Adam step

        state = next_state

    append stats row
    publish latest checkpoint atomically
```

Checkpoints are written once per episode, which is why the dashboard shows
per-episode snapshots rather than individual gradient steps.

## 7. Dashboard

`src/neuron_watch.py` is a small HTTP server that:

1. Watches `data/run/snake_ai_lab_latest.pt` (or the shipped
   `models/snake_ai_lab_latest.pt`).
2. Runs a separate demo game with the latest checkpoint.
3. Serves the static dashboard from `web/`.
4. Exposes `GET /api/network` with activations, weights, biases, edges,
   Q-values, training history, and a live TD example.

The frontend draws all 283 neurons and, for readability, only the two
strongest incoming weights per neuron in the background by default. Enable
**All 19,840 connections** to see every line. Selecting a hidden or output
neuron highlights all its connections and shows its equation. Each weight can
be inspected on its destination neuron; input nodes represent features and
have no incoming weights or bias.

## 8. File map

| Path | Purpose |
|---|---|
| `src/snake_ai.py` | game, model, agent, trainer, pygame viewer, CLI |
| `src/neuron_watch.py` | live dashboard server and demo runner |
| `web/index.html` | dashboard markup |
| `web/dashboard.css` | dashboard styles |
| `web/network.js` | SVG network rendering and interaction |
| `web/dashboard.js` | polling, panels, charts, demo board |
| `web/replay.js` | browser inference over recorded states |
| `scripts/build_pages.py` | export the static GitHub Pages demo |
| `scripts/benchmark.py` | latency benchmark |
| `tests/test_smoke.py` | contract tests |
| `models/` | shipped demo checkpoints |
| `data/` | runtime checkpoints and training stats |

## 9. Known limitations

- The state is hand-engineered; a pixel-based agent would need a larger model
  and much more data.
- The reward shaping can bias behavior toward short-term food-chasing.
- Vanilla DQN with a `max` target overestimates values. Double DQN would be a
  straightforward improvement.
- Training is sample-inefficient compared with modern off-policy methods.
- The `best` checkpoint is saved on ties as well as new bests, so it is not
  strictly the single best episode.
- The demo game is separate from the trainer and is not used for learning.

These are teaching trade-offs, not hidden bugs. The point is to keep the
whole system small enough to understand and watch.
