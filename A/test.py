import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from A.explore import MODALITIES
from A.preprocess import data_preprocess


def test_model(test_set, model, model_para_path):
    """testing process

    :param test_set: test set
    :param model: the trained model
    :param model_para_path: saved model parameters path
    :return: confusion matrix, accuracy, precision, recall, f1 score as evaluation metrics for testing results.
    """
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: design a way to define the path correctly
    model.load_state_dict(torch.load(model_para_path)['model_state_dict'])

    model.eval()
    true_label_list = list()
    predicted_label_list = list()

    total_val_loss = 0.0
    with torch.no_grad():
        for step, (test_inputs, test_labels) in enumerate(test_dataloader, 1):
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            predicted_labels = model(test_inputs)
            _, predicted_labels = torch.max(predicted_labels.data, 1)
            predicted_label_list.extend(predicted_labels.tolist())
            true_label_list.extend(test_labels.tolist())

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_label_list, predicted_label_list)

    # Compute the evaluation metrics
    accuracy = accuracy_score(true_label_list, predicted_label_list)
    precision = precision_score(true_label_list, predicted_label_list, average='weighted')
    recall = recall_score(true_label_list, predicted_label_list, average='weighted')
    f1 = f1_score(true_label_list, predicted_label_list, average='weighted')
    return conf_matrix, accuracy, precision, recall, f1


if __name__ == '__main__':
    training_set, validation_set, test_set = data_preprocess(1, MODALITIES[1])
    test_model(validation_set)
