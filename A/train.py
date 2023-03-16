import math
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch import nn, optim, tensor
from torch.utils.data import DataLoader

from A.explore import MODALITIES
from models import PlainCNN, ResNet, ResidualBlock
from A.preprocess import make_dataset
from constants import TRAIN_BATCH_SIZE, TRAIN_EPOCH_NUM


class Trainer:
    def __init__(
            self,
            model,
            device,
            optimizer,
            criterion
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.best_valid_score = np.inf
        self.n_patience = 0
        self.last_model = None

    def fit(self, epochs, train_dataloader, val_dataloader, save_path, patience):
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_time = self.train_epoch(train_dataloader)
            valid_loss, valid_auc, valid_time = self.valid_epoch(val_dataloader)

            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s            ",
                n_epoch, train_loss, train_time
            )

            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s",
                n_epoch, valid_loss, valid_auc, valid_time
            )

            # if True:
            # if self.best_valid_score < valid_auc:
            if self.best_valid_score > valid_loss:
                self.save_model(n_epoch, save_path, valid_loss, valid_auc)
                self.info_message(
                    "auc improved from {:.4f} to {:.4f}. Saved model to '{}'",
                    self.best_valid_score, valid_loss, self.last_model
                )
                self.best_valid_score = valid_loss
                self.n_patience = 0
            else:
                self.n_patience += 1

            if self.n_patience >= patience:
                self.info_message("\nValid auc didn't improve last {} epochs.", patience)
                break

    def train_epoch(self, train_dataloader):
        self.model.train()
        t = time.time()
        total_train_loss = 0.0
        for step, (inputs, labels) in enumerate(train_dataloader, 1):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero gradients
            self.optimizer.zero_grad()
            # forward, backward, optimize
            outputs = self.model(inputs)
            # TODO: dimension problem,
            loss = self.criterion(torch.squeeze(outputs), labels)
            # loss.requires_grad = True
            loss.backward()
            self.optimizer.step()
            loss_per_batch = loss.item()
            total_train_loss += loss_per_batch

        return total_train_loss / len(train_dataloader), int(time.time() - t)

    def valid_epoch(self, val_dataloader):
        self.model.eval()
        t = time.time()
        val_labels_all = list()
        val_outputs_all = list()

        total_val_loss = 0.0
        for step, (val_inputs, val_labels) in enumerate(val_dataloader, 1):
            with torch.no_grad():
                val_inputs = val_inputs.to(self.device)
                val_labels = val_labels.to(self.device)

                val_outputs = self.model(val_inputs)
                val_loss = self.criterion(torch.squeeze(val_outputs), val_labels)
                # print(val_outputs.shape, val_labels.shape)
                _, val_outputs = torch.max(val_outputs.data, 1)
                total_val_loss += val_loss.detach().item()
                val_labels_all.extend(val_labels.tolist())
                val_outputs_all.extend(val_outputs.tolist())

            message = 'Valid Step {}/{}, valid_loss: {:.4f}'
            self.info_message(message, step, len(val_dataloader), total_val_loss / step, end="\r")

        auc = roc_auc_score(val_labels_all, val_outputs_all)

        return total_val_loss / len(val_dataloader), auc, int(time.time() - t)

    def save_model(self, n_epoch, save_path, loss, auc):
        self.last_model = f"{save_path}-e{n_epoch}-loss{loss:.3f}-auc{auc:.3f}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            self.last_model,
        )

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)


def train_new(train_set, val_set, test_set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_dataloader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_set, shuffle=False)


    # load model
    model = PlainCNN()
    # model = ResNet(ResidualBlock, [3, 4, 6, 3])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # can add momentum
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    trainer = Trainer(model, device, optimizer, criterion)
    trainer.fit(TRAIN_EPOCH_NUM, train_dataloader, val_dataloader, MODALITIES[1], 10)

    # model.load_state_dict(torch.load(PATH))

    model.eval()
    t = time.time()
    val_labels_all = list()
    val_outputs_all = list()

    total_val_loss = 0.0
    for step, (val_inputs, val_labels) in enumerate(test_dataloader, 1):
        with torch.no_grad():
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_inputs)
            _, val_outputs = torch.max(val_outputs.data, 1)
            val_labels_all.extend(val_labels.tolist())
            val_outputs_all.extend(val_outputs.tolist())

    auc = roc_auc_score(val_labels_all, val_outputs_all)
    print(auc)

# def train(train_set, val_set, test_set):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     np.random.seed(42)  # numpy random generator
#
#     torch.manual_seed(42)
#     # TODO: validation step.
#     # training loop
#     train_dataloader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
#     val_dataloader = DataLoader(val_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
#
#     total_train_samples = len(train_set)
#     total_val_samples = len(val_set)
#
#     train_iteration_num = math.ceil(total_train_samples / TRAIN_BATCH_SIZE)
#     val_iteration_num = math.ceil(total_val_samples / TRAIN_BATCH_SIZE)
#
#     print(total_train_samples, train_iteration_num)
#     print(f'The number of training samples is {total_train_samples}.\n'
#           f'The number of training epoch is {TRAIN_EPOCH_NUM}.\n'
#           f'The training batch size is {TRAIN_BATCH_SIZE}.\n'
#           f'The number of iterations per training epoch is {train_iteration_num}.')
#     plain_cnn = PlainCNN()
#     criterion = nn.BCELoss()
#     # can add momentum
#     optimizer = optim.SGD(plain_cnn.parameters(), lr=0.01)
#
#     total_train_loss_list = list()
#     total_val_loss_list = list()
#     train_acc_list = list()
#     val_acc_list = list()
#
#     for epoch in range(TRAIN_EPOCH_NUM):
#         total_train_loss = 0.0
#         train_true_label_num = 0
#         for i, (inputs, labels) in enumerate(train_dataloader):
#             # zero gradients
#             optimizer.zero_grad()
#             # forward, backward, optimize
#             outputs = plain_cnn(inputs)
#             # _, predictions = torch.max(outputs, 1)
#             #
#             # for j in range(len(labels)):
#             #     if predictions[j] == labels[j]:
#             #         train_true_label_num = train_true_label_num + 1
#
#             # TODO: dimension problem,
#             loss = criterion(torch.squeeze(outputs), labels.float())
#             # loss.requires_grad = True
#             loss.backward()
#             optimizer.step()
#             loss_per_batch = loss.item()
#             # if (i + 1) % 1 == 0:
#             #     print(
#             #         f'epoch {epoch + 1} / {TRAIN_EPOCH_NUM}, '
#             #         f'step {i + 1}  / {train_iteration_num}, '
#             #         f'inputs {inputs.shape}, '
#             #         f'loss {loss_per_batch:.3f}'
#             #     )
#             total_train_loss += loss_per_batch
#
#         avg_train_loss = total_train_loss / train_iteration_num
#         total_train_loss_list.append(avg_train_loss)
#
#         train_acc = train_true_label_num / total_train_samples
#         train_acc_list.append(train_acc)
#
#         print(f'Average training loss for epoch {epoch + 1} is {avg_train_loss}')
#         # print(f'Training accuracy for epoch {epoch + 1} is {train_acc}')
#
#         # validation step
#         with torch.no_grad():
#             total_val_loss = 0.0
#             val_true_label_num = 0
#             for i, (val_inputs, val_labels) in enumerate(val_dataloader):
#                 val_outputs = plain_cnn(val_inputs)
#                 # _, predictions = torch.max(val_outputs, 1)
#                 #
#                 # for j in range(len(val_labels)):
#                 #     if predictions[j] == val_labels[j]:
#                 #         val_true_label_num = val_true_label_num + 1
#
#                 val_loss = criterion(torch.squeeze(val_outputs), val_labels.float())
#                 total_val_loss += val_loss
#
#         avg_val_loss = total_val_loss / val_iteration_num
#         total_val_loss_list.append(avg_val_loss)
#
#         val_acc = val_true_label_num / total_val_samples
#         val_acc_list.append(val_acc)
#
#         print(f'Average validation loss for epoch {epoch + 1} is {avg_val_loss}')
#         # print(f'Validation accuracy for epoch {epoch + 1} is {val_acc}')
#
#     print('Finished Training')
#
#     # Generate a sequence of integers to represent the epoch numbers
#     epochs = range(1, TRAIN_EPOCH_NUM + 1)
#
#     print(val_acc_list)
#     # Plot and label the training and validation loss values
#     plt.plot(epochs, total_train_loss_list, label='Training Loss')
#     plt.plot(epochs, total_val_loss_list, label='Validation Loss')
#
#     # Add in a title and axes labels
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#
#     # Set the tick locations
#     # plt.xticks(arange(0, 21, 2))
#
#     # Display the plot
#     plt.legend(loc='best')
#     plt.savefig('first.png')
#
#     # TODO: separate test section from train.py
#     # test section:
#     test_dataloader = DataLoader(test_set, shuffle=False)
#     total_test_samples = len(test_set)
#     print(f'There are {total_test_samples} test samples.')
#
#     # plain_cnn.eval()
#     test_outputs = list()
#     test_labels = list()
#
#     test_true_label_num = 0.0
#     with torch.no_grad():
#         # Test out inference with 5 samples
#         for i, (test_input, test_label) in enumerate(test_dataloader):
#             test_output = plain_cnn(test_input)
#             if test_output > 0.5:
#                 test_output = tensor([1])
#             else:
#                 test_output = tensor([0])
#
#             test_outputs.append(torch.squeeze(test_output).numpy())
#             test_labels.append(torch.squeeze(test_label).numpy())
#             if torch.equal(test_output, test_label):
#                 test_true_label_num = test_true_label_num + 1
#
#         test_acc = test_true_label_num / total_test_samples
#         print(f'The testing accuracy is {test_acc}')
#     test_result_df = pd.DataFrame(list(zip(test_outputs, test_labels)), columns=['Output', 'Label'])
#     print(test_result_df)


if __name__ == '__main__':
    training_set, validation_set, test_set = make_dataset(0, MODALITIES[1])
    train_new(training_set, validation_set, test_set)
