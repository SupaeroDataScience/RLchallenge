from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from FlappyAgent import FlappyPolicy
import sys

from CNN import CNN


def select_action(model, screen):
    neural_value = model.predict(screen)
    if round(neural_value) == 1:
        return 119
    else:
        return None

def epsilon(step):
    if step<1e6:
        return 1.-step*9e-7
    return .1


if __name__ == "__main__":
    continue_training = False
    path_CNN = "model_dql_flappy3_dense.dqf"

    total_steps = 10#000000


    game = FlappyBird()
    p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
    p.init()
    reward = 0.0

    # Generate CNN
    cnn = CNN()
    if continue_training:
        try:
            print("Loading an existing CNN...")
            cnn.load(path_CNN)
        except IOError:
            print("File not found : ", path_CNN)
            sys.exit()
    else:
        print("Creating a new CNN...")
        cnn.init()

    #Train
    for step in range(total_steps):
        p.reset_game()

        while (not p.game_over()):
            state = game.getGameState()
            screen = p.getScreenRGB()

            #action
            if np.random.rand() < epsilon(step):
                action = 119*np.random.randint(0,2)
            else:
                action = select_action(cnn.model, screen)

            reward = p.act(action)
            print(reward)


    # Save CNN
    print("Saving the new CNN...")
    cnn.save(path_CNN)