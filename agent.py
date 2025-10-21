from enum import Enum
from dataclasses import dataclass, field

class Role(Enum):
    EXPLORER = 1
    COMMUNICATOR = 2

@dataclass (frozen=False)
class Agent:
    role: Role
    id: int
    x: float
    y: float
    c: float
    communication_threshold: float
    centroids: list = field(default_factory=list)
    centroids_schedule: list = field(default_factory=list)
    v_max: float = 0
    nest_radius: float = 0
    kappa: float = 0
    sigma: float = 0
    sensing_radius: float = 0
    loading_state: bool = False
    exploration_centroid: tuple = (0,0)
    exploration_radius: float = 0
    t: int = 0
    seen_neighbors: list = field(default_factory=list)