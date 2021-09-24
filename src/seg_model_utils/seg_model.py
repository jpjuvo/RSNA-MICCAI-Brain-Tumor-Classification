from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torchsummary import summary
from collections import OrderedDict
import os

class BaseModel(nn.Module, ABC):
    r"""
    BaseModel with basic functionalities for checkpointing and restoration.
    """

    def __init__(self):
        super().__init__()
        self.best_loss = 1000000

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def test(self):
        """
        To be implemented by the subclass so that
        models can perform a forward propagation
        :return:
        """
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def restore_checkpoint(self, ckpt_file, optimizer=None):
        r"""
        Restores checkpoint from a pth file and restores optimizer state.

        Args:
            ckpt_file (str): A PyTorch pth file containing model weights.
            optimizer (Optimizer): A vanilla optimizer to have its state restored from.

        Returns:
            int: Global step variable where the model was last checkpointed.
        """
        if not ckpt_file:
            raise ValueError("No checkpoint file to be restored.")

        try:
            ckpt_dict = torch.load(ckpt_file)
        except RuntimeError:
            ckpt_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        # Restore model weights
        self.load_state_dict(ckpt_dict['model_state_dict'])

        # Restore optimizer status if existing. Evaluation doesn't need this
        # TODO return optimizer?????
        if optimizer:
            optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])

        # Return global step
        return ckpt_dict['epoch']

    def save_checkpoint(self,
                        directory,
                        epoch, loss,
                        optimizer=None,
                        name=None):
        r"""
        Saves checkpoint at a certain global step during training. Optimizer state
        is also saved together.

        Args:
            directory (str): Path to save checkpoint to.
            epoch (int): The training. epoch
            optimizer (Optimizer): Optimizer state to be saved concurrently.
            name (str): The name to save the checkpoint file as.

        Returns:
            None
        """
        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':
                self.state_dict(),
            'optimizer_state_dict':
                optimizer.state_dict() if optimizer is not None else None,
            'epoch':
                epoch
        }

        # Save the file with specific name
        if name is None:
            name = "{}_{}_epoch.pth".format(
                os.path.basename(directory),  # netD or netG
                'last')

        torch.save(ckpt_dict, os.path.join(directory, name))
        if self.best_loss > loss:
            self.best_loss = loss
            name = "{}_BEST.pth".format(
                os.path.basename(directory))
            torch.save(ckpt_dict, os.path.join(directory, name))

    def count_params(self):
        r"""
        Computes the number of parameters in this model.

        Args: None

        Returns:
            int: Total number of weight parameters for this model.
            int: Total number of trainable parameters for this model.

        """
        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)

        return num_total_params, num_trainable_params

    def inference(self, input_tensor):
        self.eval()
        with torch.no_grad():
            output = self.forward(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            return output.cpu().detach()

class AdaptiveConcatPool3d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool3d(sz)
        self.mp = nn.AdaptiveMaxPool3d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class UNet3D_v2(BaseModel):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet3D_v2, self).__init__()

        features = init_features
        self.encoder1 = UNet3D_v2._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D_v2._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D_v2._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D_v2._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D_v2._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3D_v2._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D_v2._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D_v2._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D_v2._block(features * 2, features, name="dec1")

        
        
        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        
        n_fn = features
        self.clf_head = nn.Sequential(
            AdaptiveConcatPool3d(sz=1),
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.ReLU(inplace=True),
            nn.Linear(n_fn * 2, 1)
        )
        

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        seg = self.conv(dec1)
        clf = self.clf_head(dec1)
        
        return torch.sigmoid(seg), clf

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    #(name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    #(name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "norm2", nn.InstanceNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    
    def test(self,device='cpu'):

        input_tensor = torch.rand(1, 2, 32, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (2, 32, 32, 32),device='cpu')
        print("Unet3D test is complete")