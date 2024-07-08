import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.backends.cudnn
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms
import torchvision.transforms.functional
from tqdm import tqdm

from src.train_src.dataloader import get_classes, get_data_loaders


# from ..train_src.dataloader import get_classes, get_data_loaders, count_parameters


def apply_test_transforms(inp):
    out = transforms.functional.resize(inp, [224, 224])
    out = transforms.functional.to_tensor(out)
    out = transforms.functional.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return out


def predict(model, filepath, classes, show_img=False, url=False):
    if url:
        response = requests.get(filepath)
        im = Image.open(BytesIO(response.content))
    else:
        im = Image.open(filepath)
    if show_img:
        plt.imshow(im)
    im_as_tensor = apply_test_transforms(im)
    minibatch = torch.stack([im_as_tensor])
    if torch.cuda.is_available():
        minibatch = minibatch.cuda()
    pred = model(minibatch)
    _, classnum = torch.max(pred, 1)
    # print(classnum)
    return classes[classnum]


def formatText(string):
    string = string[4:]
    string = string.replace("-", " ")
    return string


if __name__ == "__main__":
    dataset_path = "../../data/CUB_200_2011/images/"
    # im_path = '../../data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
    # im_path = '../../data/CUB_200_2011/images/011.Rusty_Blackbird/Rusty_Blackbird_0001_6548.jpg'
    show_img = True

    # pt = torch.load('../../models/checkpoints/checkpoint_epoch_19.pt')
    pt = torch.load('../../models/checkpoints2/model.pt')

    print("Loading model with accuracy: ")
    print(pt['best_acc'])

    classes = get_classes(dataset_path)
    model = torchvision.models.efficientnet_b2()
    # EfficientNet Model, Uncomment these models if you want to use
    n_inputs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(n_inputs, 2048),  # Increase the size of the first fully connected layer
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(2048, 2048),  # Add another fully connected layer
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(2048, len(classes))  # Adjust the output size to match the number of classes
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(pt['model_state_dict'])
    model.eval()

    test_loss = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    criterion = criterion.to(device)

    test_loss = 0.0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    (val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(dataset_path, 64, train=False)
    for data, target in tqdm(test_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(
            correct_tensor.cpu().numpy())
        if len(target) == 64:
            for i in range(64):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
