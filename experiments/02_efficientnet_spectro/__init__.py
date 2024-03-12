import os
from collections import OrderedDict
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchvision.transforms.v2 import Compose
from torchvision.ops.stochastic_depth import StochasticDepth

from hms_brain_activity.module import TrainModule, PredictModule
from hms_brain_activity.datasets import HmsClassificationDataset
from hms_brain_activity import transforms as t
from hms_brain_activity.utils import split_annotations_across_patients


class BasicBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()  # Could equivalently use F.relu()
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        if in_channels != out_channels or stride != 1:
            self.projection_shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm1d(num_features=out_channels),
            )
        else:
            self.projection_shortcut = lambda x: x

    def forward(self, x):
        identity = self.projection_shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = self.relu(x)
        return x


class ResNet1d34Backbone(nn.Sequential):
    channels = (64, 128, 256, 512)

    def __init__(self, in_channels: int):
        super().__init__()

        # Layer 1
        conv1 = OrderedDict(
            conv1=nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.channels[0],
                kernel_size=7,
                stride=2,
                padding=0,
                bias=False,
            ),
            bn=nn.BatchNorm1d(num_features=self.channels[0]),
            relu=nn.ReLU(),
        )
        self.conv1 = nn.Sequential(conv1)

        # Layer 2
        in_channels2 = self.channels[0]
        conv2 = OrderedDict(
            mp=nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
            conv2_1=BasicBlock1d(self.channels[0], self.channels[0], stride=1),
            conv2_2=BasicBlock1d(in_channels2, self.channels[0], stride=1),
            conv2_3=BasicBlock1d(in_channels2, self.channels[0], stride=1),
        )
        self.conv2 = nn.Sequential(conv2)

        # Layer 3
        in_channels3 = self.channels[1]
        conv3 = OrderedDict(
            conv3_1=BasicBlock1d(in_channels2, self.channels[1], stride=2),
            conv3_2=BasicBlock1d(in_channels3, self.channels[1], stride=1),
            conv3_3=BasicBlock1d(in_channels3, self.channels[1], stride=1),
            conv3_4=BasicBlock1d(in_channels3, self.channels[1], stride=1),
        )
        self.conv3 = nn.Sequential(conv3)

        # Layer 4
        in_channels4 = self.channels[2]
        conv4 = OrderedDict(
            conv4_1=BasicBlock1d(in_channels3, self.channels[2], stride=2),
            conv4_2=BasicBlock1d(in_channels4, self.channels[2], stride=1),
            conv4_3=BasicBlock1d(in_channels4, self.channels[2], stride=1),
            conv4_4=BasicBlock1d(in_channels4, self.channels[2], stride=1),
            conv4_5=BasicBlock1d(in_channels4, self.channels[2], stride=1),
            conv4_6=BasicBlock1d(in_channels4, self.channels[2], stride=1),
        )
        self.conv4 = nn.Sequential(conv4)

        # Layer 5
        in_channels5 = self.channels[3]
        conv5 = OrderedDict(
            conv5_1=BasicBlock1d(in_channels4, self.channels[3], stride=2),
            conv5_2=BasicBlock1d(in_channels5, self.channels[3], stride=1),
            conv5_3=BasicBlock1d(in_channels5, self.channels[3], stride=1),
        )
        self.conv5 = nn.Sequential(conv5)

        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weight_bias)

    def _init_weight_bias(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)


class ClassificationHead1d(nn.Sequential):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(num_channels, num_classes, 1)


# %%

# inverted_residual_setting = [
#     FusedMBConvConfig(1, 3, 1, 24, 24, 2),
#     FusedMBConvConfig(4, 3, 2, 24, 48, 4),
#     FusedMBConvConfig(4, 3, 2, 48, 64, 4),
#     MBConvConfig(4, 3, 2, 64, 128, 6),
#     MBConvConfig(6, 3, 1, 128, 160, 9),
#     MBConvConfig(6, 3, 2, 160, 256, 15),
# ]
# last_channel = 1280

# EfficientNet(
#     inverted_residual_setting,
#     dropout,
#     last_channel=last_channel,
#     norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
#     **kwargs,
# )

# Conv2dNormActivation(
#     3,
#     firstconv_output_channels,
#     kernel_size=3,
#     stride=2,
#     norm_layer=norm_layer,
#     activation_layer=nn.SiLU,
# )

class MBConv(nn.Module):
    def adjust_channels()
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class Conv2dNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, norm_cls=None, act_cls=None, **conv_kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)
        if not conv_kwargs.get("bias") is None:
            conv_kwargs["bias"] = norm_cls is None

        if norm_cls is not None:
            self.bn = norm_cls(out_channels)
        if act_cls is not None:
            self.act = act_cls()



class EfficientNetV2sBackbone(nn.Sequential):
    channels = (24, 24, 48, 64, 128, 160, 256)

    def __init__(self, in_channels: int):
        super().__init__(self)
        self.in_channels = in_channels

        # Layer 0: conv 3x3 s2
        self.layer0 = Conv2dNormAct(
            in_channels,
            out_channels=self.channels[0],
            kernel_size=3,
            stride=2,
            norm_cls=nn.BatchNorm2d,
            act_cls=nn.SiLU,
            padding=0,
            bias=False,
        )

        # Layer 1: fused-mbconv1 3x3 s1
        layer1 =
        # Layer 2: fused-mbconv1 3x3 s2
        # Layer 3: fused-mbconv4 3x3 s2

        # Layer 4: mbconv4 3x3 s2
        # Layer 5: mbconv4 3x3 s1
        # Layer 6: mbconv4 3x3 s2

        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weight_bias)

    def _init_weight_bias(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init_range = 1.0 / math.sqrt(m.out_features)
            nn.init.uniform_(m.weight, -init_range, init_range)
            nn.init.zeros_(m.bias)

    ...


# %%


def model_config(hparams):
    num_channels = 19
    num_classes = 6
    return nn.Sequential(
        ResNet1d34Backbone(num_channels),
        ClassificationHead1d(ResNet1d34Backbone.channels[-1], num_classes),
        nn.LogSoftmax(dim=1),
    )


def transforms(hparams):
    return [
        t.FillNanNpArray(0),
        t.PadNpArray(
            t.BandPassNpArray(
                hparams["config"]["bandpass_low"],
                hparams["config"]["bandpass_high"],
                hparams["config"]["sample_rate"],
            ),
            padlen=hparams["config"]["sample_rate"],
        ),
        t.ScaleEEG(1 / (35 * 1.5)),
        t.ScaleECG(1 / 1e4),
        t.TanhClipNpArray(4),
        t.DoubleBananaMontageNpArray(),
        t.ToTensor(),
    ]


def train_config(hparams):
    module = TrainModule(
        model_config(hparams),
        loss_function=nn.KLDivLoss(reduction="batchmean", log_target=True),
        metrics={
            "mse": MeanSquaredError(),
        },
        optimizer_factory=partial(
            optim.Adam,
            lr=hparams["config"]["learning_rate"],
        ),
        scheduler_factory=lambda opt: {
            "scheduler": optim.lr_scheduler.MultiStepLR(
                opt,
                milestones=hparams["config"]["milestones"],
                gamma=hparams["config"]["gamma"],
            ),
            "monitor": hparams["config"]["monitor"],
        },
    )

    annotations = pd.read_csv("./data/hms/train.csv")

    train_annotations, val_annotations = split_annotations_across_patients(
        annotations,
        test_size=0.2,
        random_state=0,
    )

    data_dir = "./data/hms/train_eegs"

    train_dataset = HmsClassificationDataset(
        data_dir=data_dir,
        annotations=train_annotations,
        transform=Compose(
            [
                t.RandomSaggitalFlipNpArray(),
                t.RandomScale(),
                *transforms(hparams),
                t.VotesToProbabilities(),
            ]
        ),
    )

    val_dataset = HmsClassificationDataset(
        data_dir=data_dir,
        annotations=val_annotations,
        transform=Compose(
            [
                *transforms(hparams),
                t.VotesToProbabilities(),
            ],
        ),
    )

    return dict(
        model=module,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=hparams["config"]["batch_size"],
            num_workers=num_workers(hparams),
            shuffle=True,
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=hparams["config"]["batch_size"],
            num_workers=num_workers(hparams),
            shuffle=False,
        ),
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor=hparams["config"]["monitor"],
                min_delta=0.0001,
                patience=hparams["config"]["patience"],
                verbose=True,
                mode="min",
            ),
        ],
    )


def num_workers(hparams) -> int:
    return min(
        hparams["config"].get("num_workers", os.cpu_count() or 0),
        os.cpu_count() or 0,
    )


def predict_config(hparams):
    module = PredictModule(
        model_config(hparams),
        transform=Compose(
            [
                lambda y_pred, md: (y_pred.squeeze(-1), md),
                lambda y_pred, md: (torch.exp(y_pred), md),
                lambda y_pred, md: (y_pred.to(torch.double), md),
                lambda y_pred, md: (torch.softmax(y_pred, axis=1), md),
                lambda y_pred, md: (y_pred.cpu().numpy(), md),
            ]
        ),
    )

    weights_path = Path(hparams["predict"]["weights_path"])
    ckpt = torch.load(weights_path, map_location="cpu")
    module.load_state_dict(ckpt["state_dict"], strict=False)

    data_dir = Path(hparams["predict"]["data_dir"])
    annotations = pd.DataFrame(
        {"eeg_id": [fp.stem for fp in data_dir.glob("*.parquet")]}
    )
    predict_dataset = HmsClassificationDataset(
        data_dir=data_dir,
        annotations=annotations,
        transform=Compose(transforms(hparams)),
    )

    return dict(
        model=module,
        predict_dataloaders=DataLoader(
            predict_dataset,
            batch_size=hparams["config"]["batch_size"],
            num_workers=num_workers(hparams),
            shuffle=False,
        ),
    )
