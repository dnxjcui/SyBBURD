import copy
import os
import time

import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model,
                 criterion,
                 optimizer,
                 scheduler,
                 device,
                 num_epochs,
                 dataset_sizes,
                 dataloaders,
                 save_dir,
                 training_history,
                 validation_history):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.dataset_sizes = dataset_sizes
        self.dataloaders = dataloaders
        self.save_dir = save_dir
        self.training_history = training_history
        self.validation_history = validation_history

    def train_model(self):
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler
        device = self.device
        num_epochs = self.num_epochs
        dataset_sizes = self.dataset_sizes
        dataloaders = self.dataloaders
        save_dir = self.save_dir
        training_history = self.training_history
        validation_history = self.validation_history

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    matches = torch.eq(preds, labels.data)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += matches.sum().item()
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                if phase == 'train':
                    training_history['accuracy'].append(
                        torch.tensor(epoch_acc).cpu())  # Convert to tensor and move to CPU
                    training_history['loss'].append(epoch_loss)
                elif phase == 'val':
                    validation_history['accuracy'].append(
                        torch.tensor(epoch_acc).cpu())  # Convert to tensor and move to CPU
                    validation_history['loss'].append(epoch_loss)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save checkpoint
                    save_path = os.path.join(save_dir, 'model.pt'.format(epoch))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': best_acc,
                        'training_history': training_history,
                        'validation_history': validation_history
                    }, save_path)

                    time_elapsed = time.time() - since
                    print('\nSaving best model: {:.4f}% validation accuracy at epoch {:.0f}, training time '
                          '{:.0f}m {:.0f}s'.format(epoch_acc, epoch, time_elapsed // 60, time_elapsed % 60))

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def train(self):
        self.model = self.train_model()
        return self.model
