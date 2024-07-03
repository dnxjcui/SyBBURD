from torch import nn, optim
from torch.nn import functional as F
from torchvision.models import EfficientNet_B2_Weights

from dataloader import get_classes, get_data_loaders, count_parameters
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
from torch.nn import CrossEntropyLoss
from trainer import Trainer

if __name__ == "__main__":
    dataset_path = "../../data/CUB_200_2011/images/"

    if not os.path.exists(dataset_path):
        print("Dataset does not exist")
        exit()

    (train_loader, train_data_len) = get_data_loaders(dataset_path, 256, train=True)
    (val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(dataset_path, 64, train=False)
    classes = get_classes(dataset_path)

    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }
    dataset_sizes = {
        "train": train_data_len,
        "val": valid_data_len
    }

    # print(len(train_loader))
    # print(len(val_loader))
    # print(len(test_loader))
    # print(train_data_len, test_data_len, valid_data_len)

    dataiter = iter(train_loader)
    images, labels = dataiter.__next__()
    images = images.numpy()  # convert images to numpy for display

    # plot the images in the batch, along with the corresponding labels
    # fig = plt.figure(figsize=(25, 4))
    # for idx in np.arange(20):
    #     ax = fig.add_subplot(2, int(20 / 2), idx + 1, xticks=[], yticks=[])
    #     plt.imshow((np.transpose(images[idx], (1, 2, 0))).astype(np.uint8))
    #     ax.set_title(classes[labels[idx]])
    #     plt.show()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    torch.backends.cudnn.benchmark = True
    model = torchvision.models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

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
    print(f"Trainable Params: {count_parameters(model)}")
    print("Model: ", model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    criterion = criterion.to(device)

    # optimizer = optim.AdamW(model.classifier.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=0.001)
    # optimizer = optim.Adam(model.classifier.parameters(), lr=0.001, betas=(0.9, 0.999))

    step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.96)

    training_history = {'accuracy': [], 'loss': []}
    validation_history = {'accuracy': [], 'loss': []}

    checkpoint_dir = '../../models/checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, scheduler=step_scheduler, num_epochs=20,
                      training_history=training_history, validation_history=validation_history,
                      dataloaders=dataloaders, device=device, dataset_sizes=dataset_sizes,
                      checkpoint_dir=checkpoint_dir)

    trainer.train()

