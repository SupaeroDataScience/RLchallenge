"""Flappy Bird agent using a pre-trained DQN model.

Loads a Keras CNN model trained with Deep Q-Learning and uses it
to select actions based on stacked grayscale frames.
"""

import numpy as np
from collections import deque
from keras.models import load_model
from skimage.color import rgb2gray
from skimage.transform import resize

# --- Constants ---

FRAME_SIZE = (80, 80)
STACK_SIZE = 4
MODEL_PATH = "FlappyModel_900000"
ACTIONS = [119, None]  # 119 = flap, None = do nothing

# --- Frame Processing ---

model = load_model(MODEL_PATH)
frame_stack = deque([np.zeros(FRAME_SIZE) for _ in range(STACK_SIZE)], maxlen=STACK_SIZE)


def preprocess_frame(screen: np.ndarray) -> np.ndarray:
    """Convert raw RGB screen to 80x80 grayscale, cropped to play area."""
    cropped = screen[60:, 25:310, :]
    return 255 * resize(rgb2gray(cropped), FRAME_SIZE)


def FlappyPolicy(state, screen) -> int | None:
    """Select action using the DQN model on stacked frames."""
    processed = preprocess_frame(screen)
    frame_stack.append(processed)
    stacked = np.stack(frame_stack, axis=-1)
    q_values = model.predict(np.expand_dims(stacked, axis=0))
    return ACTIONS[np.argmax(q_values)]
