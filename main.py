import yaml
from agent import Agent, Role
from simulator import Simulator

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

    # Initialize and run the simulator
    simulator = Simulator()
    simulator.run_simulation(agents, **sim_config)

if __name__ == "__main__":
    main()