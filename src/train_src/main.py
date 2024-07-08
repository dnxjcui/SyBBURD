import os

import torch.backends.cudnn
import torchvision
from torch import nn, optim
from torchvision.models import EfficientNet_B2_Weights

from dataloader import get_classes, get_data_loaders, count_parameters
from trainer import Trainer

if __name__ == "__main__":
    dataset_path = "../../data/CUB_200_2011/images/"

    save_dir = '../../models/checkpoints2/'

    if os.path.exists(save_dir):
        print("Checkpoint Directory already exists")
        kill_switch = input("Do you want to delete the directory? (y/n): ")
        if kill_switch == 'n':
            quit()
    else:
        os.makedirs(save_dir, exist_ok=True)

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

    print(dataset_sizes)
    quit()

    dataiter = iter(train_loader)
    images, labels = dataiter.__next__()
    images = images.numpy()  # convert images to numpy for display

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
    # print("Model: ", model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    criterion = criterion.to(device)

    optimizer = optim.AdamW(model.classifier.parameters(), lr=0.001)

    step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.96)

    training_history = {'accuracy': [], 'loss': []}
    validation_history = {'accuracy': [], 'loss': []}

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, scheduler=step_scheduler, num_epochs=20,
                      training_history=training_history, validation_history=validation_history,
                      dataloaders=dataloaders, device=device, dataset_sizes=dataset_sizes,
                      save_dir=save_dir)

    trainer.train()

