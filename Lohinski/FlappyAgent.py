from Train import Athlete
from DeepTrain import DeepAthlete

ACTIONS = [0, 119]

DEEP = False
MODEL_PATH = 'models/model_0.85_0.65_10_10_1.pkl'

athlete = DeepAthlete() if DEEP else Athlete()

athlete.load_model(file_path=MODEL_PATH)


def FlappyPolicy(state, _):
    return ACTIONS[athlete.act(state=state)]
