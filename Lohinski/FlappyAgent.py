from StateEngineeringAthlete import Athlete
from DeepStateAthlete import DeepAthlete

ACTIONS = [0, 119]

deep = False
model = '.models/model_0.85_0.65_15_15_2.pkl'

athlete = DeepAthlete() if deep else Athlete()

athlete.load_model(file_path=model)


def FlappyPolicy(state, _):
    return ACTIONS[athlete.act(state=state)]
