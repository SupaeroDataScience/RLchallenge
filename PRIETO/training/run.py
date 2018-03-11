# You're not allowed to change this file
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from FlappyPolicy import FlappyPolicy
import math

game = FlappyBird()

p = PLE(game, fps=30, frame_skip=1, num_steps=1,force_fps=True, display_screen=True)


p.init()
reward = 0.0

nb_games = 10_000
cumulated = np.zeros((nb_games))

agent = FlappyPolicy(model='DQN',
                     nb_games=nb_games)

step = -1

for i in range(nb_games):

    p.reset_game()
    agent.reset_game()
    keep_going = True
    while(keep_going):
        step += 1
        keep_going &= (not p.game_over())
        state = game.getGameState()
        screen = p.getScreenRGB()
        # Your job is to define this function.
        action = agent.next_action(state, screen, reward, step)

        if keep_going:
            reward = p.act(action)
            if reward != 1.0:
                reward = 0.1

            cumulated[i] = cumulated[i] + (1 if reward == 1 else 0)
    '''
    print(f'state is: {state}')
    
    print(f'action taken: {action}')
    print()
    print()
    '''
    # print(f'{i} training achieved {cumulated[i]} score')
    # if i > 100 and not i % 20:
    #     average_score = np.mean(cumulated[i-100:i+1])
    #     max_score = np.max(cumulated[:i+1])
    #     agent.log(['mean_score','max_score'],[average_score,max_score],i)

    if cumulated[i] > 0:
        print(f"{int(cumulated[i])}", end='', flush=True)

    if not i % 100:
        if i >= 100:
            average_score = np.mean(cumulated[i-100:i+1])
            max_score = np.max(cumulated[i-100:i+1])
            agent.save()
            #agent.log(['mean_score','max_score'],[average_score,max_score],i)
            if average_score > 20:
                break
        else:
            average_score = 0.0    
            max_score = 0.0

        print('', flush=True)
        print(' ____________________________\n', flush=True)
        print(f' | {i:_} out of {nb_games:_} episodes completed ', flush=True)
        print(f' | ---------- ', flush=True)
        print(f' | average_score: \t {average_score:.4f} ', flush=True)
        print(f' | max_score: \t \t {max_score:.0f} ', flush=True)
        print(f' | ---------- ', flush=True)
        agent.print_settings()
        print(' ____________________________', flush=True)
        print('', flush=True)
    else:
        print('.', end='', flush=True)


# pygame.display.quit()
# pygame.quit()

average_score = np.mean(cumulated)
max_score = np.max(cumulated)
print()
print('___________TOTAL______________', flush=True)
print(f'average_score: {average_score:.4}')
print(f'max_score: {max_score:.0f}')


agent.save()
