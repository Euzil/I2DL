import os
import torch
import pickle

from exercise_code.models import Encoder, Classifier
from exercise_code.tests.base_tests import bcolors

PARAM_LIMIT = 5e6
SIZE_LIMIT_MB = 20
ACC_THRESHOLD = 0.5


def checkParams(model):
    n_params = sum(p.numel() for p in model.parameters())
    color = "green" if n_params < PARAM_LIMIT else "red"
    normalized = float("{:.3f}".format(n_params / 1e6))

    if n_params > PARAM_LIMIT:
        print(
            "# Paramters: Your model has {} mio. params but must have less than 5 mio. params. Simplify your model before submitting it. You won't need that many params :)".format(
            bcolors.colorize(color, normalized)))
        return False
    print("# Paramters: Your model has {} mio. params.".format(bcolors.colorize(color, normalized)))
    return True


def checkLayers(model):
    '''
        Important Note: convolutional layers are not allowed in this exercise, as they have not been covered yet in the lecture.
        Using these would be highly unfair towards student that haven't heard about them yet. 
    '''

    forbidden_layers = [torch.nn.modules.conv.Conv2d]

    for key, module in model.encoder._modules.items():
        for i in range(len(module)):
            if type(module[i]) == forbidden_layers:
                print(
                    "Please don't use convolutions! For now, only use layers that have been already covered in the lecture!")
                return False

    return True


def checkSize(path="./models/classifier_pytorch.torch"):
    size = os.path.getsize(path)
    sizeMB = size / 1e6
    color = "green" if sizeMB < SIZE_LIMIT_MB else "red"
    normalized = float("{:.1f}".format(sizeMB))
    if sizeMB > SIZE_LIMIT_MB:
        print(
            f"Size: {bcolors.colorize('red', 'WARNING - ')} \
                Your model is too large! The size is {bcolors.colorize('red', normalized)}\
                    MB, but it must be less than 20 MB. Please simplify your model before submitting.")
        return False
    print(f"Size: Great! Your model size is {bcolors.colorize('green', normalized)} MB and is less than 20 MB.")
    return True


def printModelInfo(model):
    accepted = checkParams(model) & checkLayers(model)
    print("Model accepted!") if accepted else print(
        "Model not accepted. Please follow the instructions.")
    return accepted


def load_model(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))["classifier_pt1"]

    encoder = Encoder(model_dict['encoder_hparam'], model_dict['encoder_inputsize'], model_dict['encoder_latent_dim'])
    model = Classifier(model_dict['hparams'], encoder)
    model.load_state_dict(model_dict["state_dict"])
    return model


def save_model(model, file_name, directory="models"):
    model = model.cpu()
    model_dict = {"classifier_pt1": {
        "state_dict": model.state_dict(),
        "hparams": model.hparams,
        'encoder_hparam': model.encoder.hparams,
        'encoder_inputsize': model.encoder.input_size,
        'encoder_latent_dim': model.encoder.latent_dim,
        'encoder_state_dict': model.encoder.state_dict()
    }}
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(model_dict, open(os.path.join(directory, file_name), 'wb', 4))


def test_and_save(model, val_dataloader, test_dataloader):
    _, val_acc = model.getAcc(val_dataloader)
    val_acc *= 100
    color = "green" if val_acc >= ACC_THRESHOLD * 100 else "red"
    print("Validation Accuracy: {}%".format(bcolors.colorize(color, val_acc)))
    if val_acc < ACC_THRESHOLD:
        print(f"{bcolors.colorize('red', 'That is too low!')} Please tune your model in order to reach at least {bcolors.colorize('yellow', ACC_THRESHOLD * 100)}% before running on the test set and submitting!")
        return

    if not (checkParams(model) & checkLayers(model)):
        return

    save_model(model, "classifier_pytorch.p")
    if not checkSize("./models/classifier_pytorch.p"):
        return

    print("Your model has been saved and is ready to be submitted.")
    _, test_acc = model.getAcc(test_dataloader)
