import yaml
from agent import Agent, Role, LostAgent
from simulator import Simulator
import numpy as np
import random

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
        radius = np.random.uniform(sim_config['exploration_radius']+agent_config['exploration_buffer_radius'], sim_config['exploration_radius']+2*agent_config['exploration_buffer_radius'])
        random_x = np.cos(degree)*radius
        random_y = np.sin(degree)*radius
        num_agents = random.randint(1,5)
        agent = LostAgent(id=i, x=random_x, y=random_y, num_agents=num_agents)
        lost_agents.append(agent)

    # Initialize and run the simulator
    simulator = Simulator()
    simulator.run_simulation(agents, lost_agents, **sim_config)

if __name__ == "__main__":
    main()