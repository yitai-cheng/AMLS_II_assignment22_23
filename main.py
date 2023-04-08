import gc
import os
from matplotlib import pyplot as plt
import seaborn as sns
import A

print("The program is starting...")
print("Start downloading the needed data...")
# ======================================================================================================================
# explore the dataset
A.explore_dataset()
# ======================================================================================================================
# Data preprocessing
# preprocessed data for single modality model with data_make_mode = 0
k_fold_cv_list0, test_set0 = A.data_preprocess(0, A.MODALITIES[0])
# preprocessed data for single modality model with data_make_mode = 1
k_fold_cv_list1, test_set1 = A.data_preprocess(1, A.MODALITIES[0])
# preprocessed data for multi-modality model with data_make_mode = 2
k_fold_cv_list2, test_set2 = A.data_preprocess(2)
# preprocessed data for multi-modality model with data_make_mode = 3
k_fold_cv_list3, test_set3 = A.data_preprocess(3)
# # ======================================================================================================================
# Training and validation the model
# Build model object.
# For single modality model, use PlainCNN() first.
single_modality_model = A.PlainCNN()
# Train PlainCNN model with data_make_mode = 0
train_acc_list0, val_acc_list0 = A.train_with_cross_validation(k_fold_cv_list0, single_modality_model, 'PlainCNN', mode=0)

single_modality_model = A.PlainCNN()
# Train PlainCNN model with data_make_mode = 1
train_acc_list1, val_acc_list1 = A.train_with_cross_validation(k_fold_cv_list1, single_modality_model, 'PlainCNN', mode=1)

# For multi-modality model, use MultiModalCNN() first.
multi_modality_model = A.MultiModalCNN()
# Train MultiModalCNN model with data_make_mode = 2
train_acc_list2, val_acc_list2 = A.train_with_cross_validation(k_fold_cv_list2, multi_modality_model, 'MultiModalCNN', mode=2)

multi_modality_model = A.MultiModalCNN()
# Train MultiModalCNN model with data_make_mode = 3
train_acc_list3, val_acc_list3 = A.train_with_cross_validation(k_fold_cv_list3, multi_modality_model, 'MultiModalCNN', mode=3)

current_dir = os.path.dirname(os.path.realpath(__file__))
result_dir = os.path.join(current_dir, 'results')
model_dir = os.path.join(current_dir, 'models')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
# Create subplots with 2 rows and 2 columns
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot the learning curve for Model A on the first subplot
axes[0, 0].plot(train_acc_list0, label='Training', linestyle='-', color='blue')
axes[0, 0].plot(val_acc_list0, label='Validation', linestyle='--', color='green')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Model 0 - Learning Curve')
axes[0, 0].legend()

# Plot the learning curve for Model B on the second subplot
axes[0, 1].plot(train_acc_list1, label='Training', linestyle='-', color='blue')
axes[0, 1].plot(val_acc_list1, label='Validation', linestyle='--', color='green')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Model 1 - Learning Curve')
axes[0, 1].legend()

# Plot the learning curve for Model C on the third subplot
axes[1, 0].plot(train_acc_list2, label='Training', linestyle='-', color='blue')
axes[1, 0].plot(val_acc_list2, label='Validation', linestyle='--', color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_title('Model 2 - Learning Curve')
axes[1, 0].legend()

# Plot the learning curve for Model D on the fourth subplot
axes[1, 1].plot(train_acc_list3, label='Training', linestyle='-', color='blue')
axes[1, 1].plot(val_acc_list3, label='Validation', linestyle='--', color='green')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Model 3 - Learning Curve')
axes[1, 1].legend()

# Adjust the layout and display the figure
plt.tight_layout()

plt.savefig(os.path.join(result_dir, 'learning_curve_PlainCNN.png'))

# # make comparison with data_make_mode = 3
single_modality_model_PlainCNN = A.PlainCNN()
single_modality_model_ResNet34 = A.ResNet([3, 4, 6, 3])
single_modality_model_EfficientNetB0 = A.EfficientNet()

multi_modality_model_PlainCNN_attention = A.MultiModalCNN()
multi_modality_model_ResNet34_attention = A.MultiModalResNet()
multi_modality_model_EfficientNetB0_attention = A.MultiModalEfficientNet()
#
train_acc_list_SSM_PlainCNN, val_acc_list_SSM_PlainCNN = A.train_with_cross_validation(k_fold_cv_list0, single_modality_model_PlainCNN, 'SSM_PlainCNN', mode=2)
train_acc_list_SSM_ResNet34, val_acc_list_SSM_ResNet34 = A.train_with_cross_validation(k_fold_cv_list0, single_modality_model_ResNet34, 'SSM_ResNet34', mode=2)
train_acc_list_SSM_EfficientNetB0, val_acc_list_SSM_EfficientNetB0 = A.train_with_cross_validation(k_fold_cv_list0, single_modality_model_EfficientNetB0, 'SSM_EfficientNetB0', mode=2)
train_acc_list_MMM_PlainCNN_attention, val_acc_list_MMM_PlainCNN_attention = A.train_with_cross_validation(k_fold_cv_list2, multi_modality_model_PlainCNN_attention, 'MMM_PlainCNN', mode=2)
train_acc_list_MMM_ResNet34_attention, val_acc_list_MMM_ResNet34_attention = A.train_with_cross_validation(k_fold_cv_list2, multi_modality_model_ResNet34_attention, 'MMM_ResNet34', mode=2)
train_acc_list_MMM_EfficientNetB0_attention, val_acc_list_MMM_EfficientNetB0_attention = A.train_with_cross_validation(k_fold_cv_list2, multi_modality_model_EfficientNetB0_attention, 'MMM_EfficientNetB0', mode=2)

_, ax = plt.subplots()

# Plot the learning curve for Model 0
ax.plot(val_acc_list_SSM_PlainCNN, label='Model 0', linestyle='-', color='blue')

# Plot the learning curve for Model 1
ax.plot(val_acc_list_SSM_ResNet34, label='Model 1', linestyle='-', color='red')

# Plot the learning curve for Model 2
ax.plot(val_acc_list_SSM_EfficientNetB0, label='Model 2', linestyle='-', color='green')

# Plot the learning curve for Model 3
ax.plot(val_acc_list_MMM_PlainCNN_attention, label='Model 3', linestyle='-', color='orange')

# Plot the learning curve for Model 4
ax.plot(val_acc_list_MMM_ResNet34_attention, label='Model 4', linestyle='-', color='purple')

# Plot the learning curve for Model 5
ax.plot(val_acc_list_MMM_EfficientNetB0_attention, label='Model 5', linestyle='-', color='brown')

ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Learning Curves')
ax.legend()
plt.savefig(os.path.join(result_dir, 'ablation_study.png'))

# ======================================================================================================================
# Test the model
conf_matrix0, accuracy0, precision0, recall0, f1_score0 = A.test_model(test_set0, single_modality_model_PlainCNN,
                                                                       os.path.join(model_dir,
                                                                                    'mode2-SSM_PlainCNN-e17-loss0.686-acc0.590.pth'))
print('SMM_PlainCNN performance: accuracy: {}, precision: {}, recall: {}, f1: {}'.format(accuracy0, precision0, recall0,
                                                                                         f1_score0))

conf_matrix1, accuracy1, precision1, recall1, f1_score1 = A.test_model(test_set0, single_modality_model_ResNet34,
                                                                       os.path.join(model_dir,
                                                                                    'mode2-SSM_ResNet34-e8-loss0.687-acc0.581.pth'))
print('SMM_ResNet34 performance: accuracy: {}, precision: {}, recall: {}, f1: {}'.format(accuracy1, precision1, recall1,
                                                                                         f1_score1))

conf_matrix2, accuracy2, precision2, recall2, f1_score2 = A.test_model(test_set0, single_modality_model_EfficientNetB0,
                                                                       os.path.join(model_dir,
                                                                                    'mode2-SSM_EfficientNetB0-e7-loss0.695-acc0.581.pth'))
print('SMM_EfficientNetB0 performance: accuracy: {}, precision: {}, recall: {}, f1: {}'.format(accuracy2, precision2,
                                                                                               recall2, f1_score2))

conf_matrix3, accuracy3, precision3, recall3, f1_score3 = A.test_model(test_set2,
                                                                       multi_modality_model_PlainCNN_attention,
                                                                       os.path.join(model_dir,
                                                                                    'mode2-MMM_PlainCNN-e20-loss0.700-acc0.556.pth'))
print('MMM_PlainCNN performance: accuracy: {}, precision: {}, recall: {}, f1: {}'.format(accuracy3, precision3, recall3,
                                                                                         f1_score3))

conf_matrix4, accuracy4, precision4, recall4, f1_score4 = A.test_model(test_set2,
                                                                       multi_modality_model_ResNet34_attention,
                                                                       os.path.join(model_dir,
                                                                                    'mode2-MMM_ResNet34-e20-loss0.695-acc0.521.pth'))
print('MMM_ResNet34 performance: accuracy: {}, precision: {}, recall: {}, f1: {}'.format(accuracy4, precision4, recall4,
                                                                                         f1_score4))

conf_matrix5, accuracy5, precision5, recall5, f1_score5 = A.test_model(test_set2,
                                                                       multi_modality_model_EfficientNetB0_attention,
                                                                       os.path.join(model_dir,
                                                                                    'mode2-MMM_EfficientNetB0-e13-loss0.679-acc0.590.pth'))
print('MMM_EfficientNetB0 performance: accuracy: {}, precision: {}, recall: {}, f1: {}'.format(accuracy5, precision5,
                                                                                               recall5, f1_score5))

conf_matrix_list = [conf_matrix0, conf_matrix1, conf_matrix2, conf_matrix3, conf_matrix4, conf_matrix5]
# Define class labels
class_names = ['MGMT_pos', 'MGMT_neg']

# Define colormap
cmap = "Blues"

# Create figure with 6 subplots
fig, axs = plt.subplots(2, 3, figsize=(10, 8))
axs = axs.flatten()

# Plot the confusion matrices
for i in range(6):
    sns.heatmap(conf_matrix_list[i], annot=True, fmt='d', cmap=cmap, ax=axs[i],
                xticklabels=class_names, yticklabels=class_names)
    axs[i].set_xlabel('Predicted')
    axs[i].set_ylabel('True')
    axs[i].set_title('Confusion Matrix ' + str(i + 1))

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
# ======================================================================================================================

# Clean up memory/GPU etc...             # Some code to free memory if necessary.
gc.collect()

# ======================================================================================================================
