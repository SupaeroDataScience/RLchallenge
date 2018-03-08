from Train import Athlete

ACTIONS = [0, 119]

MODEL_PATH = 'models/model_0.85_0.70_15_10_2.pkl'

athlete = Athlete()

athlete.load_model(file_path=MODEL_PATH)


def FlappyPolicy(state, _):
    return ACTIONS[athlete.act(state=state)]
