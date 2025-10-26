from planning_services import PlanningService
from movement_services import MovementService
from communication_services import CommunicationService
from memory_services import MemoryService
from agent import Role
import matplotlib.pyplot as plt
import pandas as pd
class Simulator:
    def run_simulation(self,agents,simulation_length,proportion_explorers,exploration_radius,exploration_period,num_agents,forget_frequency):
        planner = PlanningService()
        planner.assign_roles(agents,proportion_explorers)
        planner.plan_routes(agents,exploration_radius,exploration_period)
        schedule = {}
        plt.plot()
        for t in range(simulation_length):
            schedule[t] = []
            for agent in agents:
                self.step(agent,agents,t,forget_frequency)
                schedule[t].extend([agent.id,agent.x,agent.y])
            if t % 200 == 0 and t != 0:
                import seaborn as sns
                df = pd.DataFrame.from_dict(schedule).T.to_numpy()
                df = df.reshape(num_agents*(t+1),3)
                df = pd.DataFrame(df)
                
                agent_ids = sorted(df.iloc[:,0].astype(int).unique())
                palette = sns.color_palette("viridis", len(agent_ids))
                color_map = dict(zip(agent_ids, palette))

                sns.lineplot(x=df.iloc[:,1], y=df.iloc[:,2], hue=df.iloc[:,0].astype(int), palette=palette)
                for agent in agents:
                    for centroid in agent.centroids:
                        plt.scatter(x=centroid[0], y=centroid[1], color=color_map[agent.id])
                plt.show()
            print(t)
    def step(self,agent,agents,t, forget_frequency=5):
        mover = MovementService()
        communicator = CommunicationService()
        memorizer = MemoryService()
        mover.move(agent,agents,t)
        communicator.sense(agent,agents,t)
        memorizer.forget_seen_neighbors(agent,t,forget_frequency)
