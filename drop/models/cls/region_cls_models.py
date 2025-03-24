from typing import Generator, List, Optional, Tuple, Dict, Any
from omegaconf import DictConfig
import torch
from torch import Tensor


class RegionMIL(torch.nn.Module):
    """Region MIL network"""

    def __init__(
        self,
        encoder: Any,
        decoder: Any,
        conv_block: Optional[Any] = None,
        extra_features: Optional[bool] = False,

    ) -> None:
        """
        Setup Region MIL network
        Parameters
        ----------
        encoder: Any
        decoder: Any
        conv_block: Any (optional) To add another conv block after encoder
        """
        super().__init__()
        self.encoder = encoder
        self.extra_features = extra_features
        encoder_out_dim = (
            self.encoder.out_dim if hasattr(self.encoder, "out_dim") else self.encoder.feature_extractor.embed_dim
        )
        if conv_block is not None:
            self.conv_block = conv_block(in_features=encoder_out_dim)
            encoder_out_dim = self.conv_block.out_dim
        if self.extra_features:
            encoder_out_dim += len(self.extra_features)
        self.decoder = decoder(in_features=encoder_out_dim)

    def forward(self, batch: Tensor, extra_features: Tensor=None):
        """
        Forward for RegionMIL model
        Parameters
        ----------
        batch : torch Tensor of shape [batch_size, channels, height, width]
        extra_features: torch Tensor of shape [batch_size, n_features]

        Returns
        -------
        Logits prediction for each class
        """
        if type(self.encoder) == DictConfig:
            z = batch
        else:
            z = self.encoder.forward(batch)
        if z.dim() == 1:  # if batch size is 1
            z = z.unsqueeze(0)
        if self.extra_features:
            z = torch.cat([z, extra_features], dim=1)
        else:
            pass
        logits = self.decoder.forward(z)

        return logits, None


class RegionDeepMIL(torch.nn.Module):
    """Implementation of the wsi label prediction network"""

    def __init__(
        self,
        encoder: Any,
        fc1: Any,
        fc2: Any,
        decoder: Any,
        extra_features: Optional[bool] = False,

    ) -> None:
        """
        Setup Region DeepMIL network
        Parameters
        ----------
        """
        super().__init__()
        self.extra_features = extra_features
        self.encoder = encoder
        encoder_out_dim = (
            self.encoder.out_dim if hasattr(self.encoder, "out_dim") else self.encoder.feature_extractor.embed_dim
        )
        self.fc1 = fc1(in_features=encoder_out_dim)
        self.fc2 = fc2(in_features=self.fc1.out_dim)
        encoder_out_dim = self.fc2.out_dim
        if self.extra_features:
            encoder_out_dim += len(self.extra_features)
        self.attention_decoder = decoder(in_features=encoder_out_dim)

    def forward(self, batch: Tensor, extra_features: Tensor=None):
        """
        Forward for RegionMIL model
        Parameters
        ----------
        batch : torch Tensor of shape [batch_size, channels, height, width]
        extra_features: torch Tensor of shape [batch_size, n_features]

        Returns
        -------
        Logits prediction for each class
        """
        if type(self.encoder) == DictConfig:
            z1 = batch
        else:
            z1 = self.encoder.forward(batch)

        if z1.dim() == 1:
            z1 = z1.unsqueeze(0)
        z2 = self.fc1(z1)
        z3 = self.fc2(z2)
        if self.extra_features:
            z3 = torch.cat([z3, extra_features], dim=1)
        logits, A = self.attention_decoder(z3)

        return logits, A


class RegionDeepMILPhikon(torch.nn.Module):
    """Implementation of the wsi label prediction network"""

    def __init__(
        self,
        encoder: Any,
        fc1: Any,
        attention: Any,
        decoder: Any,
        extra_features: Optional[bool] = False,

    ) -> None:
        """
        Setup Region DeepMIL network
        Parameters
        ----------
        """
        super().__init__()
        self.extra_features = extra_features

        self.encoder = encoder
        encoder_out_dim = (
            self.encoder.out_dim if hasattr(self.encoder, "out_dim") else self.encoder.feature_extractor.embed_dim
        )
        if self.extra_features:
            encoder_out_dim += len(self.extra_features)
        self.fc1 = fc1(in_features=encoder_out_dim)
        self.attention = attention(d_model=self.fc1.out_dim)
        self.decoder = decoder(in_features=self.attention.out_dim)


    def forward(self, batch: Tensor, extra_features: Tensor=None):
        """
        Forward for RegionMIL model
        Parameters
        ----------
        batch : torch Tensor of shape [batch_size, channels, height, width]
        extra_features: torch Tensor of shape [batch_size, n_features]
        Returns
        -------
        Logits prediction for each class
        """
        if type(self.encoder) == DictConfig:  # if we are using embeddigngs directly
            z1 = batch
        else:
            z1 = self.encoder.forward(batch)

        if z1.dim() == 1:
            z1 = z1.unsqueeze(0)
        if self.extra_features:
            z1 = torch.cat([z1, extra_features], dim=1)
        z2 = self.fc1(z1)
        z3, A = self.attention(z2.unsqueeze(0))
        logits = self.decoder(z3)

        return logits, A

