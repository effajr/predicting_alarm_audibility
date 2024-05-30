import os
import torch


def load_weights(model, directory, model_name):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dict_path = os.path.join('./', directory, model_name+'.tar')
    state_dict = torch.load(state_dict_path, map_location=torch.device(device))
    model.load_state_dict(state_dict['state_dict'])

    return model

