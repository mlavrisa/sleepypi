import gzip
import pickle
import numpy as np
import torch
from tqdm import tqdm, trange
from multiseg_ml_args import BreathingAnalysisArgs

from torch.utils.data import DataLoader

from multiseg_ml_models import BreathingConvAutoEnc, BreathingTAE
from multiseg_ml_train import SleepDataset, SleepDatasetTimeEmb

from glob import glob

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne(obj, traces):
    conv: bool = obj["type"] == "convnet"
    args: BreathingAnalysisArgs = obj["args"]
    model = BreathingConvAutoEnc(args) if conv else BreathingTAE(args)
    model.load_state_dict(obj["model"])
    model.eval()

    plot_conv(model.named_parameters())
    return

    model = model.to(args.device)

    dataset = SleepDataset(args, traces) if conv else SleepDatasetTimeEmb(args, traces)
    evec = torch.zeros(
        (len(dataset) // args.batch_size * args.batch_size, args.bottleneck)
    )
    shape = torch.zeros(
        (len(dataset) // args.batch_size * args.batch_size, args.winsize, 2)
    )
    loader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True)

    loop = tqdm(enumerate(loader), total=len(loader), leave=True)
    for idx, (data, _) in loop:
        data = data.to(args.device)
        res = model(data, output_bottleneck=True)
        pred = model(data)
        evec[idx * args.batch_size : (idx + 1) * args.batch_size] = res.detach().cpu()
        shape[idx * args.batch_size : (idx + 1) * args.batch_size] = pred.detach().cpu()

    X = evec.numpy()
    Y = shape.numpy()
    print("starting tsne...")
    x_emb = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(X[::10])
    print("finished tsne. Plotting...")

    while True:
        plot_points(x_emb)
        of_interest = input("comma separated list of indices >>> ").split(",")
        if all([val.isdigit() for val in of_interest]):
            _, ax = plt.subplots(1, 1)
            for val in of_interest:
                ax.plot(Y[int(val), :, 0], label=val + "x")
                ax.plot(Y[int(val), :, 1], label=val + "y")
            ax.legend()
            plt.show()


def plot_conv(named_params):
    for name, param in named_params:
        print(name, param.shape)
        if name == "conv_net.0.weight":
            cout = param.shape[0]
            cin = param.shape[1]
            img = np.zeros((cout * cin, param.shape[2]))
            _, ax = plt.subplots(1, 1)
            for idx in range(cout):
                for jdx in range(cin):
                    vals = param[idx, jdx].detach().cpu().numpy()
                    ax.plot(vals + (idx * cin + jdx) * 0.2, label=f"{idx}-{jdx}")
                    ax.axhline((idx * cin + jdx) * 0.2, alpha=0.2)
                    img[idx * cin + jdx] = vals
            ax.legend()
            plt.show()
            _, ax = plt.subplots(1, 1)
            ax.imshow(img)
            plt.show()
        if name == "pre_neck.weight":
            _, ax = plt.subplots(1, 1)
            ax.imshow(np.abs(param.detach().cpu().numpy().T))
            plt.show()



def plot_points(x_emb):
    fig, ax = plt.subplots(1, 1)
    sc = ax.scatter(x_emb[:, 0], x_emb[:, 1])
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = " ".join(list(map(str,ind["ind"])))
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


if __name__ == "__main__":
    dt = "220103_232249"

    gl = sorted(glob(f"sleepypi/run{dt}/*.pkl.gz"))

    streams = []

    for idx in trange(len(gl)):
        with gzip.open(gl[idx], "rb") as f:
            p = pickle.load(f)
            data_stream, *_ = p
            streams.append(data_stream)

    n = np.concatenate(streams, axis=0)

    norm = np.sqrt(np.mean(np.square(n[:, :2])))
    obj = torch.load("models/convnet-220124_141033.pt")
    tsne(obj, [n[:, :2].T])
