from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from multiseg_ml_args import BreathingAnalysisArgs


class BreathingTAE(nn.Module):
    """TAE = Transformer Auto Encoder"""

    def __init__(self, args: BreathingAnalysisArgs):
        super(BreathingTAE, self).__init__()
        self.embed = nn.Linear(args.time_enc_size + 2, args.embedding_size)
        self.trf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                args.embedding_size,
                args.n_head,
                args.ff_size,
                args.dropout,
                batch_first=True,
            ),
            args.n_layers,
        )
        
        self.winsize = args.winsize
        self.targ_channels = 2
        self.bottleneck = nn.Linear(args.embedding_size, args.bottleneck)
        layers = []
        for idx in range(len(args.ffn_layers) - 1):
            layers.append(nn.Linear(args.ffn_layers[idx], args.ffn_layers[idx + 1]))
            layers.append(nn.ReLU())
        self.ffn = nn.Sequential(*layers[:-1])

    def forward(self, input, output_bottleneck=False):
        emb = self.embed(input)
        trf = self.trf(emb)
        bottleneck = self.bottleneck(torch.sum(trf, dim=1))
        if output_bottleneck:
            return bottleneck
        return self.ffn(bottleneck).view(-1, self.winsize, self.targ_channels)


class BreathingConvAutoEnc(nn.Module):
    """TFA = Transformer Auto Encoder"""

    def __init__(self, args: BreathingAnalysisArgs):
        super(BreathingConvAutoEnc, self).__init__()
        conv_layers = []
        for idx in range(len(args.channels) - 1):
            conv_layers.append(
                nn.Conv1d(
                    args.channels[idx],
                    args.channels[idx + 1],
                    args.padding * 2 + 1,
                    padding=args.padding,
                )
            )
            conv_layers.append(nn.ReLU())
        self.conv_net = nn.Sequential(*conv_layers)
        self.pre_neck_size = (
            args.channels[-1] * (args.peek_intervals * 2 + 1) * args.winsize
        )
        self.winsize = args.winsize
        self.targ_channels = 2
        self.pre_neck = nn.Linear(self.pre_neck_size, args.conv_bottle)
        self.relu = nn.ReLU()
        self.bottleneck = nn.Linear(args.conv_bottle, args.bottleneck)
        layers = []
        for idx in range(len(args.ffn_layers) - 1):
            layers.append(nn.Linear(args.ffn_layers[idx], args.ffn_layers[idx + 1]))
            layers.append(nn.ReLU())
        self.ffn = nn.Sequential(*layers[:-1])

    def forward(self, input, output_bottleneck=False, only_bottleneck=False):
        if only_bottleneck:
            bottleneck = input
        else:
            conv = self.conv_net(input)
            pre_neck = self.relu(self.pre_neck(conv.view(-1, self.pre_neck_size)))
            bottleneck = self.bottleneck(pre_neck)
        if output_bottleneck:
            return bottleneck
        return self.ffn(bottleneck).view(-1, self.targ_channels, self.winsize)

class BreathingClassifier:
    def __init__(self, args:BreathingAnalysisArgs, n_classes:int, model_class:nn.Module) -> None:
        self.compressor = model_class()
        self.hidden = nn.Linear(args.bottleneck, args.ffn_layers[1])
        self.relu = nn.ReLU()
        self.ouput = nn.Linear(args.ffn_layers[1], )