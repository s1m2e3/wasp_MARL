from enum import Enum
from dataclasses import dataclass, replace

class Role(Enum):
    EXPLORER = 1
    COMMUNICATOR = 2

@dataclass (frozen=True)
class Agent:
    role: Role
    id: int
    x: float
    y: float
    c: float
    communication_threshold: float
    centroids: list[tuple(float,float)] = []
    v_max: float = 0
    nest_radius: float = 0
    kappa: float = 0
    sigma: float = 0
    sensing_radius: float = 0
    loading_state: bool = False