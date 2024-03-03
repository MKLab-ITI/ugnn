from ugnn import tasks
from ugnn import architectures
from ugnn.utils import training
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Experiment device: {device}")

task = tasks.DegreeTask(nodes=20, max_density=0.5, graphs=1000).to(device)
#task = tasks.PlanetoidTask("Cora", device)
splits = task.overtrain()


def run(Model, **kwargs):
    model = Model(task.feats, task.classes, hidden=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    return training(
        model=model,
        optimizer=optimizer,
        verbose=model.__class__.__name__,
        **splits,
        **kwargs
    )

base = run(architectures.APPNP)
improved = run(architectures.Universal)
print(f"\r{base:.4f} -> {improved:.4f}")
