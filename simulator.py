from planning_services import PlanningService
from movement_services import MovementService
# from communication_services import CommunicationService
class Simulator:
    def run_simulation(self,agents,simulation_length,proportion_explorers,exploration_radius,exploration_period,num_agents):
        planner = PlanningService()
        planner.assign_roles(agents,proportion_explorers)
        planner.plan_routes(agents,exploration_radius,exploration_period)
        for t in range(simulation_length):
            for agent in agents:
                agent.t = t
                self.step(agent,agents)
    def step(self,agent,agents):
        mover = MovementService()
        # communicator = CommunicationService()
        mover.move(agent,agents)
        # communicator.communicate(agent,agents)
