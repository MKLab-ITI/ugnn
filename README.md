# Universal Minimization on the Node Domain

This is an experimentation framework assessing
how well graph neural networks (GNN) can minimize
various functions on the node domain.


**Author:** Emmanouil (Manios) Krasanakis <br>
**Contact:** maniospas@hotmail.com <br>
**License:** Apache 2<br>

## :rocket: Quickstart
First declare a predictive task on which to assess 
a GNN architecture, and obtain its training-test-validation
data subtasks. If there are multiple graphs, 
they are packed together in one unconnected graph.
A quick way for doing this is like this:

```python
from ugnn import tasks
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
task = tasks.DegreeTask(nodes=20, max_density=0.5, graphs=1000).to(device)
splits = task.split() # default split 
```

Then declare an architecture and train it. Training returns
the test accuracy, but test nodes are never used internally.

```python
from ugnn import architectures
from ugnn.utils import training

model = architectures.APPNP(task.feats, task.classes, hidden=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
acc = training(
    model=model,
    optimizer=optimizer,
    **splits # the previously obtained splits
)
print(f'{model.__class__.__name__} accuracy {acc:.4f}')
```

