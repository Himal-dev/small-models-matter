# Benchmarks

## Reproduce

```bash
python3 scripts/benchmark.py
python3 scripts/benchmark.py --json
```

The script warms up each stage, then times 20,000 batch-one calls per stage
on one CPU thread by default. The fixed-board stages use a seeded Snake state;
only the full loop advances the game. JSON output includes the seed, sample
count, model path, and runtime versions. P99 uses nearest rank.

## Reference machine

| | |
|---|---|
| CPU | Apple M3 Pro |
| OS | macOS 26.5.2 (arm64) |
| Python | 3.9.6 |
| PyTorch | 2.8.0 |
| NumPy | 2.0.2 |
| Threads | 1 |
| Model | `models/snake_ai_lab_latest.pt` |
| Parameters | 20,099 (78.5 KiB at fp32) |

## Reference results

| Stage | Mean | Median | p99 |
|---|---:|---:|---:|
| Network forward | 19.6 µs | 19.6 µs | 29.7 µs |
| Forward + argmax | 21.8 µs | 21.1 µs | 49.3 µs |
| Features + forward + argmax | 36.8 µs | 37.0 µs | 69.6 µs |
| Full loop incl. `game.step` | 71.9 µs | 61.2 µs | 279.0 µs |

In a later 20,000-run check on the same machine, decision latency was
41.9 µs mean, 37.8 µs median, and 165.5 µs p99. The figures above are one
reference run; tail latency varies. The reference runs predate the fixed
benchmark seed and the current food-placement implementation.

**Throughput:** ~27,000 decisions/second on one CPU thread for the
features + forward + argmax path.

The `full loop` row includes the environment's own logic (collision checks,
food placement, reward calculation), which is not part of the model. It is
included to show the cost of the complete control loop. The decision-only row
ends at the argmax tensor; the full loop also extracts its Python integer.

These measurements do not include a hosted baseline. The browser demo uses a
separate JavaScript implementation and is outside this benchmark.

## Policy quality

`python scripts/evaluate.py` compares the released checkpoint with a greedy
rule that picks the non-colliding move nearest to food by Manhattan distance.
Each policy starts 100 games with the same seed sequence (`10000`–`10099`).
The cap is 5,000 moves per game; neither policy hit it.

| Policy | Mean food | Median food | Best |
|---|---:|---:|---:|
| Released DQN | 36.56 | 37 | 62 |
| Greedy Manhattan rule | 26.38 | 24 | 53 |

The seeds control the initial conditions. Food placement after a pickup
depends on the policy's occupied cells. This comparison covers one released
model and one heuristic; it does not measure training-to-training variance.
