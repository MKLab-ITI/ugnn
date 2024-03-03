from ugnn import tasks
from ugnn import architectures
from ugnn.utils import training
import torch
import numpy as np
import random

setting = "diffusion"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device:\t {device}")

#from matplotlib import pyplot as plt
#plt.hist(task.labels.cpu().numpy(), bins=task.classes)
#plt.show()


def run(Model, task, splits, **kwargs):
    print()
    model = Model(task.feats, task.classes, hidden=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    return training(
        model=model,
        optimizer=optimizer,
        verbose=model.__class__.__name__,
        **splits,
        **kwargs
    )


# make comparisons
compare = [architectures.MLP, architectures.GCN, architectures.GCNII, architectures.APPNP, architectures.Universal]
results = [list() for _ in compare]

print(f"Setting:\t {setting}")
for _ in range(20):
    if "diffusion" in setting:
        task = tasks.DiffusionTask(nodes=100, max_density=0.01, graphs=1000, alpha=random.uniform(0, 0.5)).to(device)
    elif "degree" in setting:
        task = tasks.DegreeTask(nodes=100, max_density=0.5, graphs=1000).to(device)
    elif "triangle" in setting:
        task = tasks.TrianglesTask(nodes=100, max_density=0.5, graphs=1000).to(device)
    elif "cora" in setting:
        task = tasks.PlanetoidTask("Cora", device)
    else:
        raise Exception("invalid setting")
    splits = task.overtrain() if "overtrain" in setting else task.split()
    for architecture, result in zip(compare, results):
        result.append(float(run(architecture, task, splits)))

    # show results
    print("\r".ljust(80))
    print(" ".join([architecture.__name__.ljust(8) for architecture in compare]))
    print(" ".join([f'{np.mean(result):.3f}'.ljust(8) for result in results]))

print("Standard deviations")
print(" ".join([f'{np.std(result):.3f}'.ljust(8) for result in results]))

