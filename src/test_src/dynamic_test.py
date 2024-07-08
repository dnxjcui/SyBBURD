import os

import torch.backends.cudnn
import torchvision.transforms.functional
from torch import nn
from tqdm import tqdm
import glob

from src.train_src.dataloader import get_classes
from test import predict, formatText

if __name__ == "__main__":
    dataset_path = "../../data/CUB_200_2011/images/"
    show_img = True

    pt = torch.load('../../models/checkpoints/checkpoint_epoch_19.pt')

    print("Loading model with accuracy: ")
    print(pt['best_acc'].data.item())

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

    im_path = None
    input_text = "Enter the image path (enter q to end): "
    while im_path != 'q':
        im_path = input(input_text)
        if os.path.exists(im_path):
            print(f'{formatText(predict(model, im_path, classes=classes, show_img=show_img, url=False))}\n')
            input_text = "Enter the image path (enter q to end):"
        elif os.path.exists(glob.glob(os.path.join(dataset_path, '*', im_path))[0]):
            print(f'{formatText(predict(model, (glob.glob(os.path.join(dataset_path, "*", im_path))[0]), classes=classes, show_img=show_img, url=False))}\n')
            input_text = "Enter the image path (enter q to end):"
        else:
            input_text = 'Invalid path, try again (enter q to end): '
