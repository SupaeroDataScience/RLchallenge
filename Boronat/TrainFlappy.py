"""Deep Q-Network training for Flappy Bird.

Trains a CNN-based DQN agent using experience replay on the
PyGame Learning Environment (PLE) Flappy Bird game.
"""

import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
from ple import PLE
from ple.games.flappybird import FlappyBird
from skimage.color import rgb2gray
from skimage.transform import resize

# --- Hyperparameters ---

FRAME_SIZE = (80, 80)
STACK_SIZE = 4
TOTAL_STEPS = 1_000_000
REPLAY_MEMORY_SIZE = 1_000_000
MINI_BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4
EVAL_PERIOD = 10_000
SAVE_PERIOD = 50_000
ACTIONS = None  # Set after PLE init


# --- Model ---


def build_dqn() -> Sequential:
    """Build the CNN architecture for Q-value estimation."""
    model = Sequential([
        Conv2D(16, (8, 8), strides=4, activation="relu", input_shape=(80, 80, 4)),
        Conv2D(32, (4, 4), strides=2, activation="relu"),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(2, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model


# --- Experience Replay ---


class ReplayBuffer:
    """Fixed-size circular buffer storing (state, action, reward, next_state, done) transitions."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.screens_x = np.zeros((capacity, *FRAME_SIZE), dtype=np.uint8)
        self.screens_y = np.zeros((capacity, *FRAME_SIZE), dtype=np.uint8)
        self.actions = np.zeros((capacity, 1), dtype=np.uint8)
        self.rewards = np.zeros((capacity, 1), dtype=np.uint8)
        self.terminals = np.zeros((capacity, 1), dtype=bool)
        self.terminals[-1] = True
        self.index = 0
        self.size = 0

    def append(self, screen_x, action, reward, screen_y, done):
        self.screens_x[self.index] = screen_x
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.screens_y[self.index] = screen_y
        self.terminals[self.index] = done
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_stacked_frames(self, screens, index):
        """Build a 4-frame stack walking backwards from index, stopping at terminal states."""
        frames = deque(maxlen=STACK_SIZE)
        pos = index % self.capacity
        for _ in range(STACK_SIZE):
            frames.appendleft(screens[pos])
            prev = (pos - 1) % self.capacity
            if not self.terminals[prev]:
                pos = prev
        return np.stack(frames, axis=-1)

    def sample(self, batch_size: int):
        """Sample a random minibatch of transitions with stacked frames."""
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        x = np.zeros((batch_size, *FRAME_SIZE, STACK_SIZE))
        y = np.zeros((batch_size, *FRAME_SIZE, STACK_SIZE))
        for i, idx in enumerate(indices):
            x[i] = self._get_stacked_frames(self.screens_x, idx)
            y[i] = self._get_stacked_frames(self.screens_y, idx)
        return x, self.actions[indices], self.rewards[indices], y, self.terminals[indices]


# --- Utilities ---


def preprocess_frame(screen: np.ndarray) -> np.ndarray:
    """Convert raw RGB screen to 80x80 grayscale."""
    return 255 * resize(rgb2gray(screen[60:, 25:310, :]), FRAME_SIZE)


def epsilon_schedule(step: int) -> float:
    """Linearly decay epsilon from 0.1 to 0.001 over 1M steps."""
    if step < 1_100_000:
        return (1.0 - step * 9e-7) * 0.1
    return 0.001


def clip_reward(reward: float) -> float:
    """Clip reward: 1.0 for scoring, 0.1 otherwise (survival bonus)."""
    return reward if reward == 1 else 0.1


def greedy_action(model, state: np.ndarray) -> int:
    """Select the action with highest Q-value."""
    q_values = model.predict(np.array([state]))
    return np.argmax(q_values)


def evaluate(env: PLE, model, num_games: int = 10):
    """Evaluate the current model over multiple games."""
    scores = np.zeros(num_games)
    for i in range(num_games):
        frame_stack = deque([np.zeros(FRAME_SIZE)] * STACK_SIZE, maxlen=STACK_SIZE)
        env.reset_game()
        while not env.game_over():
            screen = preprocess_frame(env.getScreenRGB())
            frame_stack.append(screen)
            stacked = np.stack(frame_stack, axis=-1)
            action = ACTIONS[np.argmax(model.predict(np.expand_dims(stacked, axis=0)))]
            scores[i] += env.act(action)
    return np.mean(scores), np.max(scores)


# --- Training Loop ---


def train():
    global ACTIONS

    game = FlappyBird(graphics="fixed")
    env = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
    env.init()
    ACTIONS = env.getActionSet()

    model = build_dqn()
    print("Model created")

    # Initialize replay buffer and first state
    replay = ReplayBuffer(REPLAY_MEMORY_SIZE)
    env.reset_game()
    screen_x = preprocess_frame(env.getScreenRGB())
    frame_stack = deque([screen_x] * STACK_SIZE, maxlen=STACK_SIZE)
    state = np.stack(frame_stack, axis=-1)

    for step in range(TOTAL_STEPS):
        # Epsilon-greedy action selection (biased toward no-flap)
        if np.random.rand() < epsilon_schedule(step):
            action_idx = 1 if np.random.randint(0, 5) != 1 else 0
        else:
            action_idx = greedy_action(model, state)

        reward = clip_reward(env.act(ACTIONS[action_idx]))
        screen_y = preprocess_frame(env.getScreenRGB())
        done = env.game_over()

        replay.append(screen_x, action_idx, reward, screen_y, done)

        # Train after warmup period
        if step > EVAL_PERIOD:
            x_batch, a_batch, r_batch, y_batch, d_batch = replay.sample(MINI_BATCH_SIZE)
            q_next = model.predict(y_batch)
            q_max = q_next.max(axis=1).reshape((MINI_BATCH_SIZE, 1))
            targets = r_batch + GAMMA * (1 - d_batch) * q_max
            q_current = model.predict(x_batch)
            q_current[np.arange(MINI_BATCH_SIZE), a_batch.ravel()] = targets.ravel()
            model.train_on_batch(x=x_batch, y=q_current)

        # Reset on game over, otherwise advance state
        if done:
            env.reset_game()
            screen_x = preprocess_frame(env.getScreenRGB())
            frame_stack = deque([screen_x] * STACK_SIZE, maxlen=STACK_SIZE)
        else:
            screen_x = screen_y
            frame_stack.append(screen_x)
        state = np.stack(frame_stack, axis=-1)

        # Periodic evaluation and model save
        if step % SAVE_PERIOD == 0:
            mean_score, max_score = evaluate(env, model)
            print(f"Step {step}: mean={mean_score:.1f}, max={max_score:.1f}")
            model.save(f"FlappyModel_{step}")


if __name__ == "__main__":
    train()
