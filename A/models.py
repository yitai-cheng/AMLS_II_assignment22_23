"""This file contains the model architectures for the project.

PlainCNN, ResNet-34, EfficientNet-B0, MultiModalCNN, MultiModalResNet-34, MultiModalEfficientNet-B0 are defined in this file respectively.

"""
import timm
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


class PlainCNN(nn.Module):
    """Plain Convolutional Neural Network (CNN) model.

    Attributes:
        num_classes (int): Number of classes for the output layer.
    """

    def __init__(self, num_classes=2):
        """
        Args:
            num_classes (int, optional): Number of classes for the output layer. Defaults to 2.
        """
        super().__init__()
        self.conv = nn.Sequential(self.conv_layer(in_chan=1, out_chan=4),
                                  self.conv_layer(in_chan=4, out_chan=8),
                                  self.conv_layer(in_chan=8, out_chan=16))

        self.fc = nn.Sequential(nn.Linear(10816, 512),
                                nn.Dropout(p=0.15),
                                nn.Linear(512, num_classes))

    def conv_layer(self, in_chan, out_chan):
        """Creates a convolutional layer with given input and output channels.

        Args:
            in_chan (int): Number of input channels.
            out_chan (int): Number of output channels.

        Returns:
            nn.Sequential: A sequential container for the convolutional layer.
        """
        conv_layer = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(out_chan))

        return conv_layer

    def forward(self, x):
        """Forward pass of the PlainCNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResidualBlock(nn.Module):
    """A residual block module for the ResNet architecture.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride for the convolutional layer. Default is 1.
        downsample (nn.Sequential, optional): Downsample layer for skip connection. Default is None.
    """

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
        """Forward pass of the ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the residual block.
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet-34 model, a deep convolutional neural network for image classification.

    Args:
        layer_nums (list): A list of integers representing the number of residual blocks in each layer.
        block (nn.Module, optional): The residual block class to use for constructing the layers. Default is ResidualBlock.
        num_classes (int, optional): Number of output classes. Default is 2.

    Attributes:
        inplanes (int): The number of input planes for the current layer.
        conv1 (nn.Sequential): Initial convolutional layer with batch normalization and ReLU activation.
        maxpool (nn.MaxPool2d): Max pooling layer.
        layer0 (nn.Sequential): First layer of residual blocks.
        layer1 (nn.Sequential): Second layer of residual blocks.
        layer2 (nn.Sequential): Third layer of residual blocks.
        layer3 (nn.Sequential): Fourth layer of residual blocks.
        avgpool (nn.AvgPool2d): Global average pooling layer.
        fc (nn.Sequential): Fully connected layers for classification, including ReLU activation and dropout.
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
        self.fc = nn.Sequential(nn.Linear(512, 1000),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(1000, num_classes))

    def _make_layer(self, block, planes, block_nums, stride=1):
        """
        Create a layer with the given number of residual blocks.

        Args:
            block (nn.Module): The residual block class to use for constructing the layer.
            planes (int): The number of output planes for the layer.
            block_nums (int): The number of residual blocks in the layer.
            stride (int, optional): The stride for the first block in the layer. Default is 1.

        Returns:
            layer (nn.Sequential): A sequential layer containing the specified number of residual blocks.
        """
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
        """
        Perform a forward pass of the model on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, num_classes) representing
            the class logits.
        """
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


class MultiModalCNN(nn.Module):
    """
    A multi-modal CNN model that processes multiple modalities of data and combines
    them using a multi-head attention mechanism before passing the output to a classifier.

    Args:
        num_modalities (int, optional): Number of input modalities. Default is 4.
        num_classes (int, optional): Number of output classes. Default is 2.
        cnn_output_size (int, optional): Output size of the PlainCNN feature extractor. Default is 128.
        attention_hidden_size (int, optional): Hidden size of the attention mechanism. Default is 64.
        num_heads (int, optional): Number of attention heads in the multi-head attention mechanism. Default is 4.

    Attributes:
        feature_extractors (nn.ModuleList): A list of CNNs for each modality.
        multihead_attention (MultiheadAttention): Multi-head attention mechanism.
        attention_query (nn.Linear): Linear layer for attention query calculation.
        attention_key (nn.Linear): Linear layer for attention key calculation.
        attention_value (nn.Linear): Linear layer for attention value calculation.
        classifier (nn.Sequential): Classifier to produce class probabilities.
    """

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
        """
        Perform a forward pass of the model on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_modalities, height, width).

        Returns:
            class_probabilities (torch.Tensor): Output tensor of shape (batch_size, num_classes) representing
                the class probabilities.
        """
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
    """
    Multi-modal deep learning model using ResNet as feature extractors for each modality, followed by a multi-head
    attention mechanism and a classifier.

    Args:
        num_modalities (int, optional): Number of input modalities. Default is 4.
        num_classes (int, optional): Number of output classes. Default is 2.
        resnet_output_size (int, optional): Output size of the ResNet feature extractors. Default is 128.
        attention_hidden_size (int, optional): Hidden size of the attention mechanism. Default is 64.
        num_heads (int, optional): Number of heads in the multi-head attention mechanism. Default is 4.

    Attributes:
        feature_extractors (nn.ModuleList): List of ResNet feature extractors for each modality.
        multihead_attention (MultiheadAttention): Multi-head attention mechanism.
        attention_query (nn.Linear): Linear layer for creating the query vector for the attention mechanism.
        attention_key (nn.Linear): Linear layer for creating the key vector for the attention mechanism.
        attention_value (nn.Linear): Linear layer for creating the value vector for the attention mechanism.
        classifier (nn.Sequential): Classifier consisting of a linear layer and a softmax activation.
    """

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
        """
        Perform a forward pass of the model on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            class_probabilities (torch.Tensor): Output tensor of shape (batch_size, num_classes) representing
                the class probabilities.
        """
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
    """
    A wrapper class for the EfficientNet model from the timm library.

    Args:
        num_classes (int, optional): Number of output classes. Default is 2.

    Attributes:
        model (timm.models.efficientnet.EfficientNet): The EfficientNet model.
    """

    def __init__(self, num_classes=2):
        super(EfficientNet, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
        # Set the number of input channels to 1 (grayscale)
        self.model_conv_stem_kernel_size = self.model.conv_stem.kernel_size
        self.model.conv_stem = nn.Conv2d(1, self.model.conv_stem.out_channels, kernel_size=(
            self.model_conv_stem_kernel_size[0], self.model_conv_stem_kernel_size[1]),
                                         stride=self.model.conv_stem.stride, padding=self.model.conv_stem.padding,
                                         bias=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        """
        Perform a forward pass of the model on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, num_classes) representing
                the class logits.
        """
        return self.model(x)



class MultiModalEfficientNet(nn.Module):
    """
    A multi-modal deep learning model that uses EfficientNet as the feature extractor for each modality.

    Args:
        num_modalities (int, optional): Number of input modalities. Default is 4.
        num_classes (int, optional): Number of output classes. Default is 2.
        resnet_output_size (int, optional): Output size of the feature extractor. Default is 128.
        attention_hidden_size (int, optional): Hidden size for the attention mechanism. Default is 64.
        num_heads (int, optional): Number of attention heads. Default is 4.

    Attributes:
        feature_extractors (nn.ModuleList): A list of EfficientNet feature extractors for each modality.
        multihead_attention (nn.MultiheadAttention): Multi-head attention mechanism.
        attention_query (nn.Linear): Linear layer for calculating attention queries.
        attention_key (nn.Linear): Linear layer for calculating attention keys.
        attention_value (nn.Linear): Linear layer for calculating attention values.
        classifier (nn.Sequential): Classifier for producing class probabilities.
    """

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
        """
        Perform a forward pass of the model on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_modalities, num_channels, height, width).

        Returns:
            class_probabilities (torch.Tensor): Output tensor of shape (batch_size, num_classes) representing
                the class probabilities.
        """
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
