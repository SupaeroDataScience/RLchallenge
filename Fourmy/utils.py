import os
import numpy as np


def myround(x, base):
    return int(base * round(float(x)/base))


def rounddown(x, base):
    return int(x - (x % base))


def roundup(x, base):
    return int(x - (x % base) + base)


def delete_files(folder_path):
    for the_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)


def init_train(fname, data_direc):
    if fname is None:
        delete_files(data_direc)
        f0 = 0
        curr_frame = 0
        nb_save = 0
        nb_games = 0
    else:
        nb_save, curr_frame, nb_games = fname.split('_')[1:]
        nb_save = ord(nb_save) - 97  # !!
        curr_frame, nb_games = int(curr_frame), int(nb_games)
        f0 = curr_frame
    return f0, curr_frame, nb_save, nb_games


def print_scores(scores, score_freq):
    print(''.join([(str(s) if s != 0 else '.') for s in scores]))
    print('Over the last', score_freq, 'games:')
    print('    MEAN', sum(scores)/len(scores))
    print('    TOTAL', sum(scores))
    print('############################################')


def update_epsilon(curr_frame, f0, eps0, eps_tau, nb_frames):
    epsilon = eps0*np.exp(-(curr_frame-f0)/eps_tau)
    # self.epsilon = eps0*(1 + (1/(f0 - self.NB_FRAMES))*(curr_frame - f0))
    print('FRAME:', curr_frame,
          100*curr_frame / nb_frames, '%', 'EPSILON: ', epsilon)
    return epsilon
