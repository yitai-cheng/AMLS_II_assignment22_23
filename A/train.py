"""This file defines the training process of the model.


"""
import os
import time
import torch
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader
from A.constants import TRAIN_BATCH_SIZE, TRAIN_EPOCH_NUM
from A.preprocess import data_preprocess


class Trainer:
    """Trainer class for training the model.

    It contains the functions for training, validation and saving the models with the best validation score.

    """
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

        self.best_valid_score = 0
        self.n_patience = 0
        self.last_model = None
        self.train_acc_list = list()
        self.valid_acc_list = list()

    def fit(self, epochs, train_dataloader, val_dataloader, save_path, patience):
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_acc, train_time = self.train_epoch(train_dataloader)
            valid_loss, valid_acc, valid_time = self.valid_epoch(val_dataloader)
            self.train_acc_list.append(train_acc)
            self.valid_acc_list.append(valid_acc)
            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, train acc: {:.4f} time: {:.2f} s",
                n_epoch, train_loss, train_acc, train_time
            )

            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, validation acc: {:.4f}, time: {:.2f} s",
                n_epoch, valid_loss, valid_acc, valid_time
            )

            # if True:
            # if self.best_valid_score < valid_auc:
            if self.best_valid_score <= valid_acc:
                self.save_model(n_epoch, save_path, valid_loss, valid_acc)
                self.info_message(
                    "validation accuracy improved from {:.4f} to {:.4f}. Saved model to '{}'",
                    self.best_valid_score, valid_acc, self.last_model
                )
                self.best_valid_score = valid_acc
                self.n_patience = 0
            else:
                self.n_patience += 1
            # early stopping with patience = 10
            if self.n_patience >= patience:
                self.info_message("\nValid acc didn't improve last {} epochs.", patience)
                break

    def train_epoch(self, train_dataloader):
        self.model.train()
        t = time.time()
        correct_train = 0
        total_train = 0
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
            _, outputs = torch.max(outputs.data, 1)

            total_train += labels.size(0)
            correct_train += (torch.squeeze(outputs) == labels).sum().item()

            loss_per_batch = loss.item()
            total_train_loss += loss_per_batch
        # Calculate training accuracy
        train_accuracy = correct_train / total_train

        return total_train_loss / len(train_dataloader), train_accuracy, int(time.time() - t)

    def valid_epoch(self, val_dataloader):
        self.model.eval()
        t = time.time()
        val_labels_all = list()
        val_outputs_all = list()
        correct_val = 0
        total_val = 0
        total_val_loss = 0.0
        with torch.no_grad():
            for step, (val_inputs, val_labels) in enumerate(val_dataloader, 1):
                val_inputs = val_inputs.to(self.device)
                val_labels = val_labels.to(self.device)

                val_outputs = self.model(val_inputs)
                val_loss = self.criterion(torch.squeeze(val_outputs), val_labels)
                _, val_outputs = torch.max(val_outputs.data, 1)

                total_val += val_labels.size(0)
                correct_val += (torch.squeeze(val_outputs) == val_labels).sum().item()

                total_val_loss += val_loss.detach().item()
                val_labels_all.extend(val_labels.tolist())
                val_outputs_all.extend(val_outputs.tolist())

                message = 'Valid Step {}/{}, valid_loss: {:.4f}'
                self.info_message(message, step, len(val_dataloader), total_val_loss / step, end="\r")
            val_accuracy = correct_val / total_val
            auc = roc_auc_score(val_labels_all, val_outputs_all)

        return total_val_loss / len(val_dataloader), val_accuracy, int(time.time() - t)

    def save_model(self, n_epoch, save_path, loss, acc):

        self.last_model = f"{save_path}-e{n_epoch}-loss{loss:.3f}-acc{acc:.3f}.pth"
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


def train_with_cross_validation(cross_validation_list, ml_model, ml_model_name, mode):
    """Train and validation.

    * Optimizer: Adam
    * Loss function: CrossEntropyLoss

    :param cross_validation_list: The list containing the train and validation set.
    :param ml_model: The machine learning model to be fitted.
    :param ml_model_name: The name of the machine learning model.
    :param mode: Data selection mode.
    :return: training accuracy and validation accuracy.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_dir = os.path.join(parent_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cv_step = 1
    cv_total = len(cross_validation_list)
    # TODO: for cross_validation_set in cross_validation_list:

    # without k fold cross validation
    cross_validation_set = cross_validation_list[0]
    print('-------------------------------------------------------------------------------------------------------')
    # print(f'Fold {cv_step} / {cv_total} cross validation')
    train_set = cross_validation_set['training_set']
    val_set = cross_validation_set['validation_set']
    # load data
    train_dataloader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    # test_dataloader = DataLoader(test_set, shuffle=False)

    # load model
    # model = PlainCNN()
    # model = MultiInputCNN()
    # model = ResNet(ResidualBlock, [3, 4, 6, 3])
    model = ml_model
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, device, optimizer, criterion)
    path_prefix = model_dir + '/mode' + str(mode) + '-' + ml_model_name
    trainer.fit(TRAIN_EPOCH_NUM, train_dataloader, val_dataloader, path_prefix, 10)
    print('-------------------------------------------------------------------------------------------------------')

    return trainer.train_acc_list, trainer.valid_acc_list
    # cv_step += 1

# model.load_state_dict(torch.load(PATH))

# model.eval()
# t = time.time()
# val_labels_all = list()
# val_outputs_all = list()
#
# total_val_loss = 0.0
# for step, (val_inputs, val_labels) in enumerate(test_dataloader, 1):
#     with torch.no_grad():
#         val_inputs = val_inputs.to(device)
#         val_labels = val_labels.to(device)
#         val_outputs = model(val_inputs)
#         _, val_outputs = torch.max(val_outputs.data, 1)
#         val_labels_all.extend(val_labels.tolist())
#         val_outputs_all.extend(val_outputs.tolist())
#
# auc = roc_auc_score(val_labels_all, val_outputs_all)
# print(auc)


if __name__ == '__main__':
    k_fold_cv_list, test_set = data_preprocess(3)
    train_with_cross_validation(k_fold_cv_list, test_set)
