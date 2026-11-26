"""
Snake AI Laboratory

A visual Deep Q-Network demonstration: 24 inputs -> 128 -> 128 -> 3 outputs.

Run from the repository root:
    python3 src/snake_ai.py train --episodes 500 --render
    python3 src/snake_ai.py play

Controls while the window is open:
    Space       Pause/resume
    Right       Execute one step while paused
    Up          Increase speed
    Down        Decrease speed
    I           Toggle the information panel
    Esc         Quit

Files created (under data/run/ by default):
    snake_ai_lab_best.pt       Best model
    snake_ai_lab_latest.pt     Latest model
    snake_training_stats.csv   Training statistics

A pretrained demo model ships in models/ and is used automatically when
there is no local checkpoint.
"""

import argparse
import csv
import os
import random
import sys
import tempfile
from collections import deque

import numpy as np
import pygame
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID = 20
CELL = 25
PANEL_WIDTH = 390
WINDOW_WIDTH = GRID * CELL + PANEL_WIDTH
WINDOW_HEIGHT = GRID * CELL

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(
    os.environ.get("SNAKE_AI_DATA", os.path.join(ROOT_DIR, "data", "run"))
)
BEST_MODEL = os.path.join(DATA_DIR, "snake_ai_lab_best.pt")
LATEST_MODEL = os.path.join(DATA_DIR, "snake_ai_lab_latest.pt")
STATS_FILE = os.path.join(DATA_DIR, "snake_training_stats.csv")
LAYER_SIZES = (24, 128, 128, 3)
GAMMA = 0.90
LEARNING_RATE = 0.001
BATCH_SIZE = 128
MEMORY_SIZE = 100_000
TARGET_SYNC = 200
GRAD_CLIP = 1.0


def resolve_model(path):
    """Return path, or the shipped demo model with the same filename."""
    if os.path.exists(path):
        return path
    fallback = os.path.join(ROOT_DIR, "models", os.path.basename(path))
    return fallback if os.path.exists(fallback) else path


# Clockwise: right, down, left, up
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

ACTION_NAMES = {
    0: "STRAIGHT",
    1: "TURN RIGHT",
    2: "TURN LEFT",
}


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class SnakeGame:
    def __init__(self, rng=None):
        self.rng = random if rng is None else rng
        self.reset()

    def reset(self):
        self.direction = 0
        center = GRID // 2
        self.snake = [
            (center, center),
            (center - 1, center),
            (center - 2, center),
        ]
        self.score = 0
        self.steps = 0
        self.idle_steps = 0
        self.food = None
        self.place_food()
        return self.get_state()

    def place_food(self):
        occupied = set(self.snake)
        free = [
            (x, y)
            for x in range(GRID)
            for y in range(GRID)
            if (x, y) not in occupied
        ]
        self.food = self.rng.choice(free) if free else None

    def danger(self, position):
        x, y = position
        return (
            x < 0
            or x >= GRID
            or y < 0
            or y >= GRID
            or position in self.snake
        )

    def distance_to_wall(self, position, direction):
        x, y = position
        dx, dy = direction

        distance = 0
        while True:
            x += dx
            y += dy

            if x < 0 or x >= GRID or y < 0 or y >= GRID:
                return distance

            distance += 1

    def distance_to_body(self, position, direction):
        x, y = position
        dx, dy = direction

        distance = 0
        while True:
            x += dx
            y += dy

            if x < 0 or x >= GRID or y < 0 or y >= GRID:
                return distance

            distance += 1

            if (x, y) in self.snake:
                return distance

    def get_features(self):
        """
        Build the input vector sent to the neural network.

        The features are deliberately named and displayed so that you can
        understand exactly what the model receives.

        Total: 24 inputs.

        0-2: immediate danger: straight, right, left
        3-6: current direction: right, down, left, up
        7-10: food direction: left, right, up, down
        11-13: normalized wall distances: straight, right, left
        14-16: normalized body distances: straight, right, left
        17: normalized horizontal food distance
        18: normalized vertical food distance
        19: normalized snake length
        20: normalized idle steps
        21: food in the same column
        22: food in the same row
        23: normalized score
        """
        hx, hy = self.snake[0]
        d = self.direction

        straight = DIRS[d]
        right = DIRS[(d + 1) % 4]
        left = DIRS[(d - 1) % 4]

        directions = [straight, right, left]

        danger_features = [
            float(self.danger((hx + dx, hy + dy)))
            for dx, dy in directions
        ]

        direction_features = [
            float(d == i)
            for i in range(4)
        ]

        if self.food is None:
            food_features = [0.0] * 4
            horizontal_distance = vertical_distance = 0.0
            horizontal_alignment = vertical_alignment = 0.0
        else:
            fx, fy = self.food
            food_features = [
                float(fx < hx), float(fx > hx), float(fy < hy), float(fy > hy),
            ]
            horizontal_distance = abs(fx - hx) / GRID
            vertical_distance = abs(fy - hy) / GRID
            horizontal_alignment = float(fx == hx)
            vertical_alignment = float(fy == hy)

        wall_distances = [
            self.distance_to_wall((hx, hy), direction) / GRID
            for direction in directions
        ]

        body_distances = [
            self.distance_to_body((hx, hy), direction) / GRID
            for direction in directions
        ]

        normalized_length = len(self.snake) / (GRID * GRID)
        normalized_idle = self.idle_steps / max(1, 100 * len(self.snake))
        normalized_score = self.score / (GRID * GRID)

        values = (
            danger_features
            + direction_features
            + food_features
            + wall_distances
            + body_distances
            + [
                horizontal_distance,
                vertical_distance,
                normalized_length,
                normalized_idle,
                horizontal_alignment,
                vertical_alignment,
                normalized_score,
            ]
        )

        return np.array(values, dtype=np.float32)

    def get_state(self):
        return self.get_features()

    def step(self, action):
        """
        Execute one action and return:

            next_state, reward, done, info
        """
        old_distance = abs(self.food[0] - self.snake[0][0]) + \
                       abs(self.food[1] - self.snake[0][1])

        if action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4

        dx, dy = DIRS[self.direction]
        hx, hy = self.snake[0]
        new_head = (hx + dx, hy + dy)

        self.steps += 1
        self.idle_steps += 1

        death = (
            self.danger(new_head)
            or self.idle_steps > 100 * len(self.snake)
        )

        if death:
            return (
                self.get_state(),
                -10.0,
                True,
                {
                    "event": "DEATH",
                    "action": ACTION_NAMES[action],
                },
            )

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.idle_steps = 0
            reward = 10.0
            event = "ATE FOOD"
            self.place_food()

            if self.food is None:
                return (
                    self.get_state(),
                    reward,
                    True,
                    {
                        "event": "WIN",
                        "action": ACTION_NAMES[action],
                    },
                )
        else:
            self.snake.pop()

            new_distance = abs(self.food[0] - new_head[0]) + \
                           abs(self.food[1] - new_head[1])

            # Small shaping reward: reward movement toward food.
            if new_distance < old_distance:
                reward = 0.05
                event = "CLOSER"
            elif new_distance > old_distance:
                reward = -0.02
                event = "FARTHER"
            else:
                reward = 0.0
                event = "MOVED"

        info = {
            "event": event,
            "action": ACTION_NAMES[action],
        }

        return self.get_state(), reward, False, info


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    The neural network estimates the value of each possible action.

    Input:
        24 descriptive features

    Hidden layers:
        128 neurons
        128 neurons

    Output:
        3 Q-values:
            straight, turn right, turn left
    """

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(LAYER_SIZES[0], LAYER_SIZES[1]),
            nn.ReLU(),
            nn.Linear(LAYER_SIZES[1], LAYER_SIZES[2]),
            nn.ReLU(),
            nn.Linear(LAYER_SIZES[2], LAYER_SIZES[3]),
        )

    def forward(self, x):
        return self.layers(x)


class Agent:
    def __init__(self):
        self.online = QNetwork()
        self.target = QNetwork()

        # Initially both networks contain identical weights.
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=LEARNING_RATE,
        )

        self.memory = []
        self.memory_index = 0
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.learn_steps = 0

    def q_values(self, state):
        """Return the three Q-values without training."""
        tensor = torch.from_numpy(state).float().unsqueeze(0)

        with torch.no_grad():
            values = self.online(tensor)[0].numpy()

        return values

    def choose_action(self, state, epsilon=0.0):
        """
        Epsilon-greedy action selection.

        With probability epsilon:
            choose randomly.

        Otherwise:
            choose the action with the highest Q-value.
        """
        if random.random() < epsilon:
            return random.randrange(3)

        values = self.q_values(state)
        return int(np.argmax(values))

    def remember(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        # A ring backed by a list keeps random replay indexing constant-time.
        if len(self.memory) < MEMORY_SIZE:
            self.memory.append(transition)
        else:
            self.memory[self.memory_index] = transition
        self.memory_index = (self.memory_index + 1) % MEMORY_SIZE

    def learn(self):
        """
        Train on a random batch of previous experiences.

        The target is:

            reward + gamma * future_value

        If the game ended, there is no future value.
        """
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.stack(states)).float()
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.from_numpy(np.stack(next_states)).float()
        dones = torch.tensor(dones, dtype=torch.float32)

        # Q-values for every action, for every state in the batch.
        all_q_values = self.online(states)

        # Select only the Q-value for the action actually taken.
        predicted = all_q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        # The target network estimates the best future action.
        with torch.no_grad():
            next_q_values = self.target(next_states)
            best_future_values = next_q_values.max(dim=1).values

            targets = rewards + self.gamma * \
                best_future_values * (1.0 - dones)

        # Mean squared error between prediction and target.
        loss = nn.functional.mse_loss(predicted, targets)

        self.optimizer.zero_grad()
        loss.backward()

        # Prevent unusually large updates.
        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(),
            max_norm=GRAD_CLIP,
        )

        self.optimizer.step()

        self.learn_steps += 1

        # Periodically copy the online network into the target network.
        if self.learn_steps % TARGET_SYNC == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def save(self, path):
        with tempfile.NamedTemporaryFile(
            dir=os.path.dirname(os.path.abspath(path)), delete=False
        ) as handle:
            temporary_path = handle.name
        try:
            torch.save(self.online.state_dict(), temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def load(self, path):
        self.online.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        self.target.load_state_dict(self.online.state_dict())
        self.online.eval()


# ---------------------------------------------------------------------------
# Visual interface
# ---------------------------------------------------------------------------

class Viewer:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Snake AI Laboratory")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.small_font = pygame.font.SysFont("monospace", 13)
        self.big_font = pygame.font.SysFont("monospace", 20)

        self.paused = False
        self.show_panel = True
        self.fps = 12

    def handle_events(self):
        step_requested = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT:
                    step_requested = True
                elif event.key == pygame.K_UP:
                    self.fps = min(60, self.fps + 2)
                elif event.key == pygame.K_DOWN:
                    self.fps = max(1, self.fps - 2)
                elif event.key == pygame.K_i:
                    self.show_panel = not self.show_panel

        return step_requested

    def draw(self, game, agent, state, action, reward, loss,
             episode, epsilon, event_text):
        self.screen.fill((20, 20, 25))

        # Draw grid.
        for x in range(GRID):
            for y in range(GRID):
                pygame.draw.rect(
                    self.screen,
                    (35, 35, 42),
                    (x * CELL, y * CELL, CELL, CELL),
                    1,
                )

        # Draw snake.
        for index, (x, y) in enumerate(game.snake):
            color = (80, 230, 120) if index == 0 else (35, 160, 85)

            pygame.draw.rect(
                self.screen,
                color,
                (x * CELL + 2, y * CELL + 2, CELL - 4, CELL - 4),
            )

        # Draw food.
        if game.food is not None:
            fx, fy = game.food
            pygame.draw.rect(
                self.screen,
                (235, 70, 70),
                (fx * CELL + 4, fy * CELL + 4, CELL - 8, CELL - 8),
            )

        if self.show_panel:
            self.draw_panel(
                game,
                agent,
                state,
                action,
                reward,
                loss,
                episode,
                epsilon,
                event_text,
            )

        pygame.display.flip()

    def text(self, message, x, y, color=(225, 225, 225), font=None):
        if font is None:
            font = self.font

        surface = font.render(str(message), True, color)
        self.screen.blit(surface, (x, y))

    def draw_panel(self, game, agent, state, action, reward,
                   loss, episode, epsilon, event_text):
        x = GRID * CELL + 12
        y = 12

        pygame.draw.rect(
            self.screen,
            (29, 29, 37),
            (GRID * CELL, 0, PANEL_WIDTH, WINDOW_HEIGHT),
        )

        self.text("SNAKE AI LABORATORY", x, y,
                  (255, 220, 90), self.big_font)
        y += 32

        self.text(
            f"Episode: {episode}  Score: {game.score}",
            x, y,
        )
        y += 20

        self.text(
            f"FPS: {self.fps}  "
            f"{'PAUSED' if self.paused else 'RUNNING'}",
            x, y,
            (255, 180, 100) if self.paused else (120, 230, 150),
        )
        y += 27

        self.text("INPUT FEATURES", x, y, (100, 200, 255))
        y += 20

        names = [
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
            "horizontal align",
            "vertical align",
            "score",
        ]

        for name, value in zip(names, state):
            self.text(
                f"{name:17s} {value: .3f}",
                x, y,
                (210, 210, 215),
                self.small_font,
            )
            y += 15

        y += 8
        q = agent.q_values(state)

        self.text("NETWORK OUTPUT / Q-VALUES", x, y, (100, 255, 180))
        y += 20

        for index, value in enumerate(q):
            color = (255, 240, 100) if index == action else (210, 210, 215)
            self.text(
                f"{ACTION_NAMES[index]:12s} {value: .5f}",
                x, y,
                color,
                self.font,
            )
            y += 19

        y += 5
        self.text(f"ACTION: {ACTION_NAMES[action]}", x, y,
                  (255, 220, 90))
        y += 20
        self.text(f"REWARD: {reward: .4f}", x, y)
        y += 20
        self.text(
            f"LOSS: {loss if loss is not None else 0.0: .5f}",
            x, y,
        )
        y += 20
        self.text(f"EPSILON: {epsilon: .3f}", x, y)
        y += 20
        self.text(f"EVENT: {event_text}", x, y, (255, 150, 150))

        self.text(
            "Space pause | Right step | Up/Down speed",
            x, WINDOW_HEIGHT - 38,
            (170, 170, 180),
            self.small_font,
        )
        self.text(
            "I panel | Esc quit",
            x, WINDOW_HEIGHT - 20,
            (170, 170, 180),
            self.small_font,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def write_stats_header():
    with open(STATS_FILE, "w", newline="") as file:
        csv.writer(file).writerow([
            "episode",
            "score",
            "average_score",
            "best_score",
            "epsilon",
            "steps",
            "average_loss",
        ])


def train(episodes, render):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    os.makedirs(DATA_DIR, exist_ok=True)
    game = SnakeGame()
    agent = Agent()
    viewer = Viewer() if render else None

    write_stats_header()

    best_score = 0
    scores = deque(maxlen=20)

    for episode in range(1, episodes + 1):
        state = game.reset()
        done = False
        epsilon = max(
            0.02,
            1.0 - episode / max(1, episodes * 0.4),
        )

        losses = []
        last_action = 0
        last_reward = 0.0
        last_event = "START"

        while not done:
            if viewer:
                step_requested = viewer.handle_events()

                if viewer.paused and not step_requested:
                    viewer.draw(
                        game, agent, state, last_action,
                        last_reward,
                        np.mean(losses) if losses else None,
                        episode, epsilon, last_event,
                    )
                    viewer.clock.tick(30)
                    continue

            action = agent.choose_action(state, epsilon)
            next_state, reward, done, info = game.step(action)

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            loss = agent.learn()

            if loss is not None:
                losses.append(loss)

            state = next_state
            last_action = action
            last_reward = reward
            last_event = info["event"]

            if viewer:
                viewer.draw(
                    game,
                    agent,
                    state,
                    action,
                    reward,
                    np.mean(losses) if losses else None,
                    episode,
                    epsilon,
                    last_event,
                )
                viewer.clock.tick(viewer.fps)

        scores.append(game.score)
        best_score = max(best_score, game.score)

        if game.score >= best_score:
            agent.save(BEST_MODEL)

        recent_average = np.mean(scores)
        average_loss = np.mean(losses) if losses else 0.0

        with open(STATS_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                episode,
                game.score,
                round(float(recent_average), 4),
                best_score,
                round(epsilon, 4),
                game.steps,
                round(float(average_loss), 6),
            ])
        agent.save(LATEST_MODEL)

        if episode % 10 == 0:
            print(
                f"Episode {episode:4d}/{episodes} | "
                f"score {game.score:3d} | "
                f"best {best_score:3d} | "
                f"average {recent_average:6.2f} | "
                f"epsilon {epsilon:.3f}"
            )

    print()
    print(f"Training complete. Best score: {best_score}")
    print(f"Best model: {BEST_MODEL}")
    print(f"Statistics: {STATS_FILE}")


# ---------------------------------------------------------------------------
# Play a saved model
# ---------------------------------------------------------------------------

def play():
    model_path = resolve_model(BEST_MODEL)

    if not os.path.exists(model_path):
        print(f"{BEST_MODEL} was not found.")
        print("Train the model first, or keep the shipped model in models/.")
        return

    game = SnakeGame()
    agent = Agent()
    agent.load(model_path)
    viewer = Viewer()

    episode = 1
    state = game.reset()
    last_action = 0
    last_reward = 0.0
    last_loss = None
    last_event = "START"

    while True:
        step_requested = viewer.handle_events()

        if viewer.paused and not step_requested:
            viewer.draw(
                game,
                agent,
                state,
                last_action,
                last_reward,
                last_loss,
                episode,
                0.0,
                last_event,
            )
            viewer.clock.tick(30)
            continue

        action = agent.choose_action(state, epsilon=0.0)
        next_state, reward, done, info = game.step(action)

        state = next_state
        last_action = action
        last_reward = reward
        last_event = info["event"]

        viewer.draw(
            game,
            agent,
            state,
            action,
            reward,
            None,
            episode,
            0.0,
            last_event,
        )
        viewer.clock.tick(viewer.fps)

        if done:
            pygame.time.wait(700)
            episode += 1
            game.reset()
            state = game.get_state()


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visual Snake Deep Q-Network laboratory"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--episodes",
        type=int,
        default=500,
    )
    train_parser.add_argument(
        "--render",
        action="store_true",
    )

    subparsers.add_parser("play")

    args = parser.parse_args()

    if args.command == "train":
        if args.episodes < 1:
            train_parser.error("--episodes must be positive")
        train(args.episodes, args.render)
    elif args.command == "play":
        play()


if __name__ == "__main__":
    main()
