from ugnn import tasks
from ugnn import architectures
from ugnn.utils import training
import torch
import numpy as np
import random
from datetime import datetime
import sys, math

# TODO: SMP architecture here: https://github.com/cvignac/SMP/blob/master/models/smp_layers.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:".ljust(10) + str(device))


def universalNRI(*args, **kwargs):
    return architectures.Universal(*args, **kwargs, nri=0)


for setting in [
    "cora",
    "citeseer",
    "pubmed",
    "triangle",
    "square",
    "longest",
    "scoorediffusion",
    "scorefixeddiffusion",
    "propagation",
    "fixedpropagation",
    "degree",
]:
    starting_time = datetime.now()
    setting = setting+" overtrain"
    #setting = "degree"  # (cora | citeseer | pubmed | scoreentropy | scorediffusion | propagation | degree | triangle | square)  [overtrain]
    compare = [
        architectures.MLP,
        architectures.GCN,
        architectures.APPNP,
        architectures.GAT,
        architectures.GCNII,
        architectures.GCNNRI,
        architectures.Universal,
    ]

    def run(Model, task, splits, verbose=True, hidden=64, **kwargs):
        from ugnn.utils import GraphConv

        GraphConv._cached_edge_index = None
        GraphConv._cached_adj_t = None
        if hidden is None:
            hidden = int(math.log2(task.feats))
            hidden = hidden ** int((hidden - 1) // 2)
            print(f"Automatically detecting hidden dimensions: {hidden}")
        bestacc = None
        bestvaliation = float("inf")
        for retry in range(1):
            #task.l1 = 1 if architectures.Universal.__name__ is Model.__name__ else 0
            #splits["train"].l1 = task.l1
            model = Model(task.feats, task.classes, hidden=hidden).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            tracker_epoch = None  # list()
            tracker_train = None  # list()
            tracker_valid = None  # list()
            tracker_test = None  # list()
            acc, validation = training(
                model=model,
                optimizer=optimizer,
                verbose=model.__class__.__name__,
                #patience=5000 if architectures.Universal.__name__ is Model.__name__ else 100,
                #clip=10 if architectures.Universal.__name__ is Model.__name__ else None,
                tracker_train=tracker_train,
                tracker_valid=tracker_valid,
                tracker_test=tracker_test,
                tracker_epoch=tracker_epoch,
                **splits,
                **kwargs,
            )
            if tracker_epoch is not None:
                from matplotlib import pyplot as plt
                plt.plot(tracker_epoch, tracker_train, label="Train")
                plt.plot(tracker_epoch, tracker_valid, label="Valid")
                plt.plot(tracker_epoch, tracker_test, label="Test")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.show()
            if bestvaliation > validation:
                bestvaliation = validation
                bestacc = acc
            if verbose:
                print()
        return bestacc

    # make comparisons
    results = [list() for _ in compare]
    print("Setting:".ljust(10) + setting)
    print(" ".join([architecture.__name__.ljust(8) for architecture in compare]))
    for _ in range(5):
        if "diffusion" in setting:
            task = tasks.DiffusionTask(
                nodes=100, max_density=0.1, graphs=500, alpha=0.1 if "fixed" in setting else random.uniform(0, 0.5)
            ).to(device)
        elif "propagation" in setting:
            task = tasks.PropagationTask(
                nodes=100, max_density=0.1, graphs=500, alpha=0.1 if "fixed" in setting else random.uniform(0, 0.5)
            ).to(device)
        elif "longest" in setting:
            task = tasks.DiameterTask(nodes=100, max_density=0.1, graphs=500, distribution_name="Longest shortest path").to(device)
        elif "degree" in setting:
            task = tasks.DegreeTask(nodes=100, max_density=0.1, graphs=500, distribution_name="Degree").to(device)
        elif "entropy" in setting:
            task = tasks.EntropyTask(nodes=100, graphs=500).to(device)
        elif "triangle" in setting:
            task = tasks.TrianglesTask(nodes=100, max_density=0.1, graphs=500, distribution_name="Count triangles").to(
                device
            )
        elif "square" in setting:
            task = tasks.SquareCliqueTask(nodes=20, max_density=0.5, graphs=500, distribution_name="Is in a 4-clique").to(
                device
            )
        elif "cora" in setting:
            task = tasks.PlanetoidTask("Cora", device)
        elif "citeseer" in setting:
            task = tasks.PlanetoidTask("Citeseer", device)
        elif "pubmed" in setting:
            task = tasks.PlanetoidTask("Pubmed", device)
        else:
            raise Exception("invalid setting")

        # from matplotlib import pyplot as plt
        # plt.hist(task.labels.cpu().numpy(), bins=task.classes)
        # plt.show()

        splits = task.overtrain() if "overtrain" in setting else task.split()
        for architecture, result in zip(compare, results):
            result.append(float(run(architecture, task, splits)))
        print("\r".ljust(80), end="")
        print("\r".ljust(80), end="")
        print("\r" + " ".join([f"{result[-1]:.5f}".ljust(8) for result in results]))

    def printall():
        print(" ".join([architecture.__name__.ljust(8) for architecture in compare]))
        print(" ".join([f"{np.mean(result):.3f}".ljust(8) for result in results]))
        print("Standard deviations")
        print(" ".join([f"{np.std(result):.3f}".ljust(8) for result in results]))
        from scipy.stats import rankdata

        ranks = rankdata(np.array(results), axis=0).T
        if "score" not in setting:
            ranks = len(compare) + 1 - ranks
        ranks = ranks.mean(axis=0)
        print("Nemenyi ranks")
        print(" ".join([f"{rank:.1f}".ljust(8) for rank in ranks]))
        print("Latex")
        print(" & ".join([f"{np.mean(result):.3f} ({rank:.1f})" for result, rank in zip(results, ranks)]))

    print("\n==== Summary ====")
    printall()
    prev_sysout = sys.stdout
    with open(
        f'results/{setting} [{str(starting_time).replace(":", "-")}].txt', "w"
    ) as sys.stdout:
        printall()
    sys.stdout = prev_sysout
