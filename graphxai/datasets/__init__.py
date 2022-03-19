from .dataset import get_dataset, GraphDataset, NodeDataset
from .load_synthetic import load_ShapeGraph
from .shape_graph import ShapeGraph

# Real-world datasets:
from .real_world.MUTAG import MUTAG
from .real_world.benzene.benzene import Benzene
from .real_world.fluoride_carbonyl.fluoride_carbonyl import FluorideCarbonyl
from .real_world.mutagenicity import Mutagenicity
from .real_world.alkane_carbonyl.alkane_carbonyl import AlkaneCarbonyl
