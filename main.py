import yaml
from agent import Agent, Role, LostAgent, DecoyAgent, Position
from simulator import Simulator
import numpy as np
import random
import json
import os

def _make_serializable(obj):
    if isinstance(obj, (DecoyAgent, Position)):
        if isinstance(obj, DecoyAgent):
            return {'id': obj.id, 'x': obj.x, 'y': obj.y, 'role': obj.role.value,'num_agents': obj.num_agents}
        return {'x': obj.x, 'y': obj.y}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
def main():
    """
    Main function to run the agent-based simulation.
    """
    # Load configuration from YAML file
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    agent_config = config['agent']
    sim_config = config['simulator']
    # Initialize agents based on the configuration
    agents = []
    for i in range(sim_config['num_agents']):
        agent = Agent(
            id=i,
            role=Role.EXPLORER,  # Default role, will be reassigned by PlanningService
            **agent_config
        )
        agents.append(agent)
    lost_agents = []
    for i in range(sim_config['num_lost_agents']):
        degree = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(sim_config['exploration_radius']+agent_config['exploration_buffer_radius']+agent_config['nest_radius'], sim_config['exploration_radius']+agent_config['nest_radius']+agent_config['exploration_buffer_radius'])
        random_x = np.cos(degree)*radius
        random_y = np.sin(degree)*radius
        num_required_agents = 3
        agent = LostAgent(id=i, x=random_x, y=random_y, num_agents=num_required_agents)
        lost_agents.append(agent)

    # Initialize and run the simulator
    simulator = Simulator()
    simulator.run_simulation(agents, lost_agents, **sim_config)
    
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    serializable_decoys = [_make_serializable(d) for d in simulator.decoys]
    serializable_centroids = [_make_serializable(c) for c in simulator.exploration_centroids]
    serializable_expansion = {k: [_make_serializable(p) for p in v] for k, v in simulator.exploration_centroids_expansion.items()}

    data_to_save = {
        "schedule": simulator.schedule,
        "decoys": serializable_decoys,
        "exploration_centroids": serializable_centroids,
        "exploration_centroids_expansion": serializable_expansion
    }

    with open(os.path.join(output_dir, 'data.json'), 'w') as f:
        json.dump(data_to_save, f, indent=4)



if __name__ == "__main__":
    main()