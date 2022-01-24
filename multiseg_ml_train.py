from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm, trange
from datetime import datetime

from multiseg_ml_args import BreathingAnalysisArgs
from multiseg_ml_models import BreathingConvAutoEnc, BreathingTAE

from glob import glob
import gzip
import pickle

import matplotlib.pyplot as plt


class SleepDataset(Dataset):
    def __init__(self, args: BreathingAnalysisArgs, traces: List[np.ndarray]) -> None:
        super().__init__()
        self.ret_size = (args.peek_intervals * 2 + 1) * args.winsize
        # why stepsize? So train and validate sets don't include data that's too close
        self.lens = np.array(
            [(t.shape[1] - self.ret_size) // args.stepsize - 1 for t in traces]
        )
        self.traces = [torch.from_numpy(t).float() for t in traces]
        self.args = args
        self._len = np.sum(self.lens)
        self._start = args.peek_intervals * args.winsize
        self._end = (args.peek_intervals + 1) * args.winsize

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index):
        trace_idx = np.min(np.argwhere(self.lens > index).squeeze())
        jump_index = index
        rand_index = np.random.randint(self.args.stepsize)
        if trace_idx > 0:
            jump_index -= self.lens[trace_idx]
        final_index = jump_index * self.args.stepsize + rand_index
        data = self.traces[trace_idx][:, final_index : final_index + self.ret_size]
        data /= torch.sqrt(torch.mean(torch.square(data)))
        return data, data[:, self._start : self._end]


class SleepDatasetTimeEmb(SleepDataset):
    def __init__(self, args: BreathingAnalysisArgs, traces: List[np.ndarray]) -> None:
        super().__init__(args, traces)
        self._cat = torch.zeros(
            (self.ret_size, args.time_enc_size)
        ).float()
        for idx in range(args.time_enc_size // 2):
            self._cat[:, 2 * idx] = torch.sin(
                torch.arange(self.ret_size) / (10000 ** (2 * idx / args.time_enc_size))
            )
            self._cat[:, 2 * idx + 1] = torch.cos(
                torch.arange(self.ret_size) / (10000 ** (2 * idx / args.time_enc_size))
            )

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return torch.cat((data.T, self._cat), dim=1), target.T


def train(args: BreathingAnalysisArgs, traces: List[np.ndarray], conv: bool):
    dataset = SleepDataset(args, traces) if conv else SleepDatasetTimeEmb(args, traces)
    lgs = [int(len(dataset) * 0.85), int(len(dataset) * 0.1)]
    lgs.append(len(dataset) - sum(lgs))
    train, valid, test = random_split(dataset, lgs)
    trainloader = DataLoader(train, args.batch_size, shuffle=True)
    validloader = DataLoader(valid, args.batch_size, shuffle=True)

    model = (
        BreathingConvAutoEnc(args).to(args.device)
        if conv
        else BreathingTAE(args).to(args.device)
    )

    criterion = nn.MSELoss()
    optim = torch.optim.SGD(
        model.parameters(),
        lr=args.learn_rate,
        momentum=0.9,
        dampening=0.0,
        nesterov=True,
    )
    sched = torch.optim.lr_scheduler.CyclicLR(
        optim,
        args.learn_rate * 0.5,
        args.learn_rate * 6.0,
        20,
        mode="triangular2",
    )

    for edx in range(args.epochs):
        print(f"Epoch {edx + 1} started.")
        train_loss = 0.0
        for data, targets in tqdm(trainloader):
            data, targets = data.to(args.device), targets.to(args.device)

            optim.zero_grad()
            pred = model(data)
            loss: torch.Tensor = criterion(targets, pred)
            loss.backward()
            optim.step()
            train_loss += loss.item()
        sched.step()
        print(
            f"Loss for epoch {edx + 1} of {args.epochs} was",
            f"{1e3 * train_loss / len(train):.3f}",
        )

    valid_loss = 0.0
    print("starting validation")
    model.eval()
    for data, targets in tqdm(validloader):
        data, targets = data.to(args.device), targets.to(args.device)

        pred = model(data)
        loss = criterion(targets, pred)
        valid_loss += loss.item()
    print(f"Loss for validation set was {1e3 * valid_loss / len(valid):.3f}")
    model_type = "convnet" if conv else "transformer"
    fn = f"models/{model_type}-" + datetime.now().strftime("%y%m%d_%H%M%S")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": optim.state_dict(),
            "args": args,
            "sched": sched,
            "type": model_type,
        },
        fn + ".pt",
    )

    data = data.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()

    if not conv:
        data = data.transpose(0, 2, 1)
        pred = pred.transpose(0, 2, 1)

    _, ax = plt.subplots(1, 1, figsize=(16, 14))
    for idx in range(8):
        ax.plot(
            np.arange(args.winsize * (args.peek_intervals * 2 + 1)),
            data[idx, 0] + idx * 2,
            label=f"{idx}",
        )
        ax.plot(
            np.arange(args.winsize) + args.peek_intervals * args.winsize,
            pred[idx, 0] + idx * 2,
            linestyle=":",
            label=f"{idx} pred",
        )
    plt.savefig(fn + ".png")
    plt.show()


if __name__ == "__main__":
    args = BreathingAnalysisArgs()
    # 211107_015105 -- this is a good one
    # 211102_003909 -- another goodie. Lots of sleep, some time not in bed at the beginning.
    #               -- No wake up though.
    # 211101_002730 -- Excellent. 5 Sleep cycles visible. One spot not flipped right.
    dt = "220103_232249"

    gl = sorted(glob(f"sleepypi/run{dt}/*.pkl.gz"))

    streams = []

    # get timezone offset for local START of night, so that DST is handled appropriately
    uctdiff = datetime.strptime(dt, "%y%m%d_%H%M%S").astimezone().utcoffset()
    tzoffset = (
        uctdiff.days * 86400 + uctdiff.seconds
    ) * 1000  # timezone offset from utc

    for idx in trange(len(gl)):
        with gzip.open(gl[idx], "rb") as f:
            p = pickle.load(f)
            data_stream, *_ = p
            streams.append(data_stream)

    n = np.concatenate(streams, axis=0)

    norm = np.sqrt(np.mean(np.square(n[:, :2])))

    train(args, [n[:, :2].T / norm], conv=True)
