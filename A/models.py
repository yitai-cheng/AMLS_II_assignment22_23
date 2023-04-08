"""This file contains the model architectures for the project.

PlainCNN, ResNet-34, EfficientNet-B0, MultiModalCNN, MultiModalResNet-34, MultiModalEfficientNet-B0 are defined in this file respectively.

"""
import timm
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


# class PlainCNN(nn.Module):
#     def __init__(self):
#         super(PlainCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9) # 4 incoming channels
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.conv3 = nn.Conv2d(32, 16, kernel_size=3)
#         self.conv3_drop = nn.Dropout2d()
#         self.conv4 = nn.Conv2d(16, 8, kernel_size=3)
#         self.conv4_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(32, 128)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 1)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 4))
#         x = F.relu(F.max_pool2d(self.conv2(x), 4))
#         x = F.relu(F.max_pool2d(self.conv3(x), 2))
#         x = F.relu(F.max_pool2d(self.conv4(x), 2))
#         x = x.view(x.shape[0], -1)
#         x = F.relu(self.fc1(x))
#         # x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         x = self.fc4(x)
#         x = torch.sigmoid(x)
#         return x


class PlainCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(self.conv_layer(in_chan=1, out_chan=4),
                                  self.conv_layer(in_chan=4, out_chan=8),
                                  self.conv_layer(in_chan=8, out_chan=16))

        self.fc = nn.Sequential(nn.Linear(10816, 512),
                                nn.Dropout(p=0.15),
                                nn.Linear(512, num_classes))

    def conv_layer(self, in_chan, out_chan):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(out_chan))

        return conv_layer

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# TODO: one layer before the attention
class MultiInputCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(self.conv_layer(in_chan=1, out_chan=4),
                                  self.conv_layer(in_chan=4, out_chan=8),
                                  self.conv_layer(in_chan=8, out_chan=16))

        self.fc = nn.Sequential(nn.Linear(32, 16),
                                nn.Dropout(p=0.15),
                                nn.Linear(16, num_classes))

    def conv_layer(self, in_chan, out_chan):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(out_chan))

        return conv_layer

    def forward(self, input):
        x1 = input[:, 0:1, :, :]
        x2 = input[:, 1:2, :, :]
        x3 = input[:, 2:3, :, :]
        x4 = input[:, 3:, :, :]

        out1 = self.conv(x1)
        out2 = self.conv(x2)
        out3 = self.conv(x3)
        out4 = self.conv(x4)
        #
        # out1 = out1.view(out1.size(0), -1)
        # out2 = out2.view(out2.size(0), -1)
        # out3 = out3.view(out3.size(0), -1)
        # out4 = out4.view(out4.size(0), -1)

        # simple concatenation as plain attention mechanism
        out = torch.cat((out1, out2, out3, out4), dim=1)
        # TODO: here the architecture is simple and subject to change.
        out = self.conv_layer(in_chan=64, out_chan=32)(out)
        out = self.conv_layer(in_chan=32, out_chan=16)(out)
        out = self.conv_layer(in_chan=16, out_chan=8)(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """Model Template for ResNet-34

    """

    def __init__(self, layer_nums, block=ResidualBlock, num_classes=2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layer_nums[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layer_nums[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layer_nums[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layer_nums[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512, num_classes)
        self.fc = nn.Sequential(nn.Linear(512, 1000),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(1000, num_classes))

    def _make_layer(self, block, planes, block_nums, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layer_nums = list()
        layer_nums.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, block_nums):
            layer_nums.append(block(self.inplanes, planes))

        return nn.Sequential(*layer_nums)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class MultiInputCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(self.conv_layer(in_chan=1, out_chan=4),
                                  self.conv_layer(in_chan=4, out_chan=8),
                                  self.conv_layer(in_chan=8, out_chan=16))

        self.fc = nn.Sequential(nn.Linear(32, 16),
                                nn.Dropout(p=0.15),
                                nn.Linear(16, num_classes))

    def conv_layer(self, in_chan, out_chan):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(out_chan))

        return conv_layer

    def forward(self, input):
        x1 = input[:, 0:1, :, :]
        x2 = input[:, 1:2, :, :]
        x3 = input[:, 2:3, :, :]
        x4 = input[:, 3:, :, :]

        out1 = self.conv(x1)
        out2 = self.conv(x2)
        out3 = self.conv(x3)
        out4 = self.conv(x4)
        #
        # out1 = out1.view(out1.size(0), -1)
        # out2 = out2.view(out2.size(0), -1)
        # out3 = out3.view(out3.size(0), -1)
        # out4 = out4.view(out4.size(0), -1)

        # simple concatenation as plain attention mechanism
        out = torch.cat((out1, out2, out3, out4), dim=1)
        # TODO: here the architecture is simple and subject to change.
        out = self.conv_layer(in_chan=64, out_chan=32)(out)
        out = self.conv_layer(in_chan=32, out_chan=16)(out)
        out = self.conv_layer(in_chan=16, out_chan=8)(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class MultiModalCNN(nn.Module):
    def __init__(self, num_modalities=4, num_classes=2, cnn_output_size=128, attention_hidden_size=64, num_heads=4):
        super(MultiModalCNN, self).__init__()

        # Define a CNN for each modality
        self.feature_extractors = nn.ModuleList([nn.Sequential(
            PlainCNN(num_classes=cnn_output_size)
        ) for _ in range(num_modalities)])
        # Define the multi-head attention mechanism
        self.multihead_attention = MultiheadAttention(embed_dim=attention_hidden_size, num_heads=num_heads)

        # Define the attention mechanism
        self.attention_query = nn.Linear(cnn_output_size, attention_hidden_size)
        self.attention_key = nn.Linear(cnn_output_size, attention_hidden_size)
        self.attention_value = nn.Linear(cnn_output_size, attention_hidden_size)

        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Linear(attention_hidden_size, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Split the input tensor into separate tensors for each modality
        modality_tensors = torch.chunk(x, x.size(1), dim=1)

        # Pass each modality tensor through its corresponding CNN
        modality_features = [feature_extractor(modality) for feature_extractor, modality in
                             zip(self.feature_extractors, modality_tensors)]

        # Calculate attention weights
        queries = [self.attention_query(feature) for feature in modality_features]
        keys = [self.attention_key(feature) for feature in modality_features]
        values = [self.attention_value(feature) for feature in modality_features]

        query = torch.stack(queries).transpose(0, 1)  # Shape: (batch_size, num_modalities, attention_hidden_size)
        key = torch.stack(keys).transpose(0, 1)  # Shape: (batch_size, num_modalities, attention_hidden_size)
        value = torch.stack(values).transpose(0, 1)  # Shape: (batch_size, num_modalities, attention_hidden_size)

        attention_output, _ = self.multihead_attention(query, key, value)
        attention_output = torch.mean(attention_output, dim=1)  # Average attention output across all heads

        # Pass the attention output through the classifier
        class_probabilities = self.classifier(attention_output)

        return class_probabilities

class MultiModalResNet(nn.Module):
    def __init__(self, num_modalities=4, num_classes=2, resnet_output_size=128, attention_hidden_size=64, num_heads=4):
        super(MultiModalResNet, self).__init__()
        # Define a CNN for each modality
        self.feature_extractors = nn.ModuleList([nn.Sequential(
            ResNet([3, 4, 6, 3], num_classes=resnet_output_size)
        ) for _ in range(num_modalities)])
        # Define the multi-head attention mechanism
        self.multihead_attention = MultiheadAttention(embed_dim=attention_hidden_size, num_heads=num_heads)

        # Define the attention mechanism
        self.attention_query = nn.Linear(resnet_output_size, attention_hidden_size)
        self.attention_key = nn.Linear(resnet_output_size, attention_hidden_size)
        self.attention_value = nn.Linear(resnet_output_size, attention_hidden_size)

        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Linear(attention_hidden_size, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Split the input tensor into separate tensors for each modality
        modality_tensors = torch.chunk(x, x.size(1), dim=1)

        # Pass each modality tensor through its corresponding CNN
        modality_features = [feature_extractor(modality) for feature_extractor, modality in
                             zip(self.feature_extractors, modality_tensors)]

        # Calculate attention weights
        queries = [self.attention_query(feature) for feature in modality_features]
        keys = [self.attention_key(feature) for feature in modality_features]
        values = [self.attention_value(feature) for feature in modality_features]

        query = torch.stack(queries).transpose(0, 1)  # Shape: (batch_size, num_modalities, attention_hidden_size)
        key = torch.stack(keys).transpose(0, 1)  # Shape: (batch_size, num_modalities, attention_hidden_size)
        value = torch.stack(values).transpose(0, 1)  # Shape: (batch_size, num_modalities, attention_hidden_size)

        attention_output, _ = self.multihead_attention(query, key, value)
        attention_output = torch.mean(attention_output, dim=1)  # Average attention output across all heads

        # Pass the attention output through the classifier
        class_probabilities = self.classifier(attention_output)

        return class_probabilities

class EfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNet, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
        # Set the number of input channels to 1 (grayscale)
        self.model_conv_stem_kernel_size = self.model.conv_stem.kernel_size
        self.model.conv_stem = nn.Conv2d(1, self.model.conv_stem.out_channels, kernel_size=(self.model_conv_stem_kernel_size[0], self.model_conv_stem_kernel_size[1]),
                                         stride=self.model.conv_stem.stride, padding=self.model.conv_stem.padding, bias=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class MultiModalEfficientNet(nn.Module):
    def __init__(self, num_modalities=4, num_classes=2, resnet_output_size=128, attention_hidden_size=64, num_heads=4):
        super(MultiModalEfficientNet, self).__init__()
        # Define a CNN for each modality
        self.feature_extractors = nn.ModuleList([nn.Sequential(
            EfficientNet(num_classes=resnet_output_size)
        ) for _ in range(num_modalities)])
        # Define the multi-head attention mechanism
        self.multihead_attention = MultiheadAttention(embed_dim=attention_hidden_size, num_heads=num_heads)

        # Define the attention mechanism
        self.attention_query = nn.Linear(resnet_output_size, attention_hidden_size)
        self.attention_key = nn.Linear(resnet_output_size, attention_hidden_size)
        self.attention_value = nn.Linear(resnet_output_size, attention_hidden_size)

        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Linear(attention_hidden_size, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Split the input tensor into separate tensors for each modality
        modality_tensors = torch.chunk(x, x.size(1), dim=1)

        # Pass each modality tensor through its corresponding CNN
        modality_features = [feature_extractor(modality) for feature_extractor, modality in
                             zip(self.feature_extractors, modality_tensors)]

        # Calculate attention weights
        queries = [self.attention_query(feature) for feature in modality_features]
        keys = [self.attention_key(feature) for feature in modality_features]
        values = [self.attention_value(feature) for feature in modality_features]

        query = torch.stack(queries).transpose(0, 1)  # Shape: (batch_size, num_modalities, attention_hidden_size)
        key = torch.stack(keys).transpose(0, 1)  # Shape: (batch_size, num_modalities, attention_hidden_size)
        value = torch.stack(values).transpose(0, 1)  # Shape: (batch_size, num_modalities, attention_hidden_size)

        attention_output, _ = self.multihead_attention(query, key, value)
        attention_output = torch.mean(attention_output, dim=1)  # Average attention output across all heads

        # Pass the attention output through the classifier
        class_probabilities = self.classifier(attention_output)

        return class_probabilities
