# graph-sdc
Graph Representation for Autonomous Driving

## Installation
Install pytorch following instructions in https://pytorch.org/get-started/locally/.
Install torch_geometric following instructions in https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html.
`python -m pip install .`

## Usage
Run GRAD: `python script/graph.py`
Run SA: `python script/transformer.py`
Record video: `python script/render.py --model [model_name]`

Default configurations for the models are in `config` folder.
