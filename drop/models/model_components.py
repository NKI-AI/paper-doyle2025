import math
import warnings
from typing import Literal, List, Any, Dict, Optional, Callable, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision
import collections


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        act_layer: Any = nn.ReLU(),
        norm: bool = False,
        num_groups: int = 32,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.out_dim = out_features
        self.fc1 = nn.Linear(in_features,  self.out_dim)
        self.act = act_layer # changed from self.act = act_layer()    act_layer: str = nn.ReLU,
        self.norm = None
        if norm:
            if num_groups:
                self.norm = nn.GroupNorm(
                    num_groups, num_channels=self.out_dim
                )  # not BatchNorm, suitable for smaller batches
            else:
                self.norm = nn.BatchNorm1d(num_features=self.out_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x) # lin
        if self.norm:
            x = self.norm(x)
        x = self.drop(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        act_layer: Any = nn.ReLU(),
        norm: bool = False,
        num_groups: int = 32,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.out_dim = out_features
        self.conv1 = nn.Conv2d(in_features, self.out_dim, kernel_size=3, stride=2, padding=1)
        self.act = act_layer
        self.norm = None
        if norm:
            if num_groups:
                self.norm = nn.GroupNorm(
                    num_groups, num_channels=self.out_dim
                )  # not BatchNorm, suitable for smaller batches
            else:
                self.norm = nn.BatchNorm1d(num_features=self.out_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x) # conv
        if self.norm:
            x = self.norm(x)
        x = self.drop(x)
        return x

def append_dropout(model, rate=0.0):
    """Adds dropout to model in-place"""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module)
        if isinstance(module, nn.ReLU):
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
            setattr(model, name, new)


def freeze_layers(model, freeze_layers):
    """Allows freezing of model weights in certain layers."""
    for name, child in model.named_children():
        for layer_n in freeze_layers:
            if layer_n == name:
                for param in child.parameters():
                    param.requires_grad = False
    return model


class ResNetEncoder(torch.nn.Module):
    def __init__(
        self, resnet_arch: str, conv_blocks: int, keep_fc: bool, freeze_layer_names: List[int], dropout_rate: float
    ):
        super().__init__()
        self.conv_blocks = conv_blocks
        self.keep_fc = keep_fc
        self.dropout_rate = dropout_rate
        self.freeze_layer_names = freeze_layer_names
        self.resnet_arch = resnet_arch

        self.feature_extractor = self.create_model()

    def create_model(self, pretrain_weights_path=None):
        model = self.customise_encoder_arch(self.get_base_model(), self.conv_blocks, self.keep_fc)
        if pretrain_weights_path != None:
            weights = torch.load(pretrain_weights_path)
            if type(weights) != collections.OrderedDict:
                raise Exception("Model weights need to be converted to correct format.")
            model.load_state_dict(self.adjust_weights_to_arch(self.get_base_model(), weights), strict=True)

        freeze_layer_names = [f"{l}" for l in self.freeze_layer_names]  # f"layer{l}" before customising
        model = freeze_layers(model, freeze_layer_names)
        append_dropout(model, self.dropout_rate)
        return model

    def add_pretrain_weights(self, pretrain_weights_path):
        self.feature_extractor = self.create_model(pretrain_weights_path)

    def forward(self, x):
        # x should be [B, C, W, H]
        y = self.feature_extractor(x).squeeze()  # outputs [B, F]

        return y

    def adjust_weights_to_arch(self, model, weights):
        """
        1.Loads pretrain weights to a complete Resnet model
        2. Customises model to desired architecture
        3. Save state_dict and return adjusted weights.
        """
        model_tmp = model
        model_tmp.load_state_dict(weights, strict=False)
        model_tmp = self.customise_encoder_arch(model_tmp, self.conv_blocks, self.keep_fc)
        adjusted_weights = model_tmp.state_dict()

        return adjusted_weights

    def customise_encoder_arch(self, model, conv_blocks, keep_fc):
        """
        This function adjusts the model architecture. A specified number of convolutional blocks are kept.
         Additionally, it can be specified to keep the original FC layer. According to the new model arch, the output dim is set.
        """
        stem_layers = ["Conv2d", "BatchNorm", "Relu", "MaxPool2D"]
        out_layers = ["avg_pool", "fc"]

        if keep_fc:
            assert conv_blocks == 4, "The last FC layer can only be kept if all the conv blocks are kept too."
            keep_layers = list(model.children())[: len(stem_layers) + conv_blocks + len(out_layers)]
            self.out_dim = model.state_dict()["fc.weight"].size()[0]
        else:
            keep_layers = list(model.children())[: len(stem_layers) + conv_blocks]
            keep_layers.append(model.avgpool)
            self.out_dim = model.state_dict()[f"layer{conv_blocks}.1.conv2.weight"].size()[0]

        return torch.nn.Sequential(*keep_layers)

    def get_base_model(self):
        if self.resnet_arch == "resnet18":
            return torchvision.models.resnet18(weights=False)
        else:
            raise Exception("Select valid resnet encoder.")


class MLP(torch.nn.Module):
    """Classification Head"""

    def __init__(
        self,
        in_features: int,
        # out_features: int,
        dropout_rate: float,
        hidden_unit_sizes: Optional[List[int]] = [],
        norm: bool = False,
        num_classes: Optional[int] = None,
        num_groups: Optional[int] = 32,
        act_layer: Any = nn.ReLU(),
    ):
        super().__init__()
        #  [(i,dim) for i, dim in reversed(list(enumerate(sizes)))]
        print(type(hidden_unit_sizes))
        sizes = [in_features] + hidden_unit_sizes # + [out_features]  #todo add out_features and adjsut all configs and logging
        head_layers: List[nn.Module] = []
        for i, dim in list(enumerate(sizes)):
            if i == len(sizes) - 1:
                if num_classes:
                    head_layers.append(nn.Linear(in_features=sizes[-1], out_features=num_classes, bias=True))
                else:
                    self.out_dim = sizes[-1]
            else:
                head_layers.append(
                    LinearBlock(
                        in_features=sizes[i],
                        out_features=sizes[i + 1],
                        norm=norm,
                        num_groups=num_groups,
                        dropout_rate=dropout_rate,
                        act_layer=act_layer
                    )
                )
        self.hidden_head = nn.Sequential(*head_layers)

    def forward(self, x):
        y = self.hidden_head(x)

        return y

class AttentionClassificationHead(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        attention_heads: int,
        num_classes: Optional[int] = None,
        attention_bias: bool = True,
    ):
        """Attention-based Deep Multiple Instance Learning class initialization
        Parameters
        ----------
        in_features : int
            Input dimension for attention weight computation.   #2048
        hidden_features : int
            Hidden dimension for attention weight computation.  #128
        num_classes : int
            Number of classes to give an output score for.
        attention_heads: int
            Number of attention heads.
        attention_bias : bool
            Whether or not to include a bias in the attention computation. Default is set to True. False may
            be preferred when padding a bag with zero-vectors for mini-batch training, since it
            ensures the attention weight for zero-vectors is 0. However, experiments on genomic feature prediction
            did not show a significant effect when removing the bias.
        Returns
        -------
        nn.Module
            Initialized model
        """
        super().__init__()

        # DeepMIL specific initialization
        self.num_classes = num_classes
        self.L = in_features  # in_features  #hidden dim / input for weights
        self.D = hidden_features  # 128
        self.K = attention_heads  # number of attention heads

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D, bias=attention_bias),
            nn.Tanh(),
            nn.Linear(self.D, self.K, bias=attention_bias),
        )  # DeepMil has attentionV and attentionU from which they calculate the attention
        if self.num_classes:
            self.classifier = nn.Sequential(
                nn.Linear((self.L * self.K), self.num_classes),
            )  # Max Ilse's group have a sigmoid at the end.
        else:
            self.out_dim = self.L * self.K

    def forward(self, x):
        H = x
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N  [1, batch_size (changes)]
        M = torch.mm(A, H)  # KxL
        if self.num_classes:
            Y_hat = self.classifier(M)
        else:
            Y_hat = M

        return Y_hat, A



# Removed mask option, adapted from https://github.com/owkin/HistoSSLscaling/blob/main/rl_benchmarks/models/slide_models/utils/attention.py#L293
class GatedAttention(torch.nn.Module):
    """Gated Attention, as defined in https://arxiv.org/abs/1802.04712.
    Permutation invariant Layer on dim 1.
    Parameters
    ----------
    d_model: int = 128
    temperature: float = 1.0
        Attention Softmax temperature
    """

    def __init__(
        self,
        d_model: int = 128,
        temperature: float = 1.0,
    ):
        super(GatedAttention, self).__init__()
        self.out_dim = d_model

        self.att = torch.nn.Linear(d_model, d_model)
        self.gate = torch.nn.Linear(d_model, d_model)

        self.w = torch.nn.Linear(d_model, 1)  # should it be 1 in our case?

        self.temperature = temperature

    def attention(
        self,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Gets attention logits.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """

        h_v = self.att(v)
        h_v = torch.tanh(h_v)

        u_v = self.gate(v)
        u_v = torch.sigmoid(u_v)

        attention_logits = self.w(h_v * u_v) / self.temperature
        return attention_logits

    def forward(
        self, v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        scaled_attention, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, IN_FEATURES), (B, N_TILES, 1)
        """
        attention_logits = self.attention(v=v)

        attention_weights = torch.softmax(attention_logits, 1)
        scaled_attention = torch.matmul(attention_weights.transpose(1, 2), v)

        return scaled_attention.squeeze(1), attention_weights