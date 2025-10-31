from planning_services import PlanningService
from movement_services import MovementService
from communication_services import CommunicationService
from memory_services import MemoryService
from behavior_service import BehaviorService
from agent import Role, DecoyAgent, Status
import matplotlib.pyplot as plt
import pandas as pd
import logging
import numpy as np
from math import floor
class Simulator:
    def __init__(self):
        self.decoys = []
        self.schedule = {}
        
    def run_simulation(self,agents,lost_agents,simulation_length,proportion_explorers,exploration_radius,exploration_period,num_agents,forget_frequency, num_lost_agents, radius_decoys):
        # --- Configure logging ---
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - T%(levelname)s - %(message)s')
        # --- Instantiate services once before the simulation loop ---
        planner = PlanningService()
        communicator = CommunicationService()
        memorizer = MemoryService()
        behavior_changer = BehaviorService()
        mover = MovementService()
        planner.assign_roles(agents,proportion_explorers)
        required_explorers = floor(proportion_explorers*len(agents))
        planned_routes = False
        for t in range(simulation_length):
            self.schedule[t] = []
            for i in range(len(agents)):
                agents = self.step(agents[i],agents,lost_agents,t, exploration_period, mover, communicator, memorizer, behavior_changer, forget_frequency)
                explorers= [explorers .role == Role.EXPLORER for explorers in agents]
                if sum(explorers) == required_explorers and not planned_routes:
                    planner.plan_routes(agents,exploration_radius,exploration_period)
                    planned_routes = True
                self.schedule[t].extend([agents[i].id,agents[i].x,agents[i].y,agents[i].role.value])
            if t == 10:
                import seaborn as sns
                explorers= [(explorers.role.value,explorers.id) for explorers in agents if explorers.role == Role.EXPLORER]
                df = pd.DataFrame.from_dict(self.schedule).T.to_numpy()
                df = df.reshape(num_agents*(t+1),4)
                df = pd.DataFrame(df)
                print(df.head())
                input("in this step")
                
                agent_ids = sorted(df.iloc[:,0].astype(int).unique())
                palette = sns.color_palette("viridis", len(agent_ids))
                color_map = dict(zip(agent_ids, palette))

                # Role 1: EXPLORER, Role 2: COMMUNICATOR, Role 3: RESCUE
                dashes_map = {1: (2, 2), 2: "", 3: (4, 1, 1, 1)}
                sns.lineplot(x=df.iloc[:,1], y=df.iloc[:,2], 
                             hue=df.iloc[:,0].astype(int), palette=color_map, 
                             style=df.iloc[:,3].astype(int), dashes=dashes_map)
                for agent in agents:
                    for centroid in agent.centroids:
                        plt.scatter(x=centroid.x, y=centroid.y, color=color_map[agent.id])
                plt.show()
            # print([(agent.id,agent.x,agent.y,agent.role) for agent in agents])
            print(t)
            input('hipox')
            
        if self.decoys:
            decoys_data = np.array([[decoy.x, decoy.y, decoy.role.value] for decoy in self.decoys])
            
            fig, ax = plt.subplots()
            ax.set_aspect('equal', adjustable='box')
            
            color_map = {
                Status.FOUND.value: 'blue',      # 2
                Status.EXPLORED.value: 'green',  # 3
                Status.ENEMY.value: 'red'        # 1
            }
            
            for x, y, status in decoys_data:
                color = color_map.get(status, 'gray')
                circle = plt.Circle((x, y), radius_decoys, color=color, alpha=0.5)
                ax.add_patch(circle)
            ax.autoscale_view()
            plt.show()
            print(self.decoys)
    def step(self,agent,agents,lost_agents,t, exploration_period, mover, communicator, memorizer, behavior_changer, forget_frequency=5):
        # --- Logging agent and simulation state at the beginning of the step ---
        # logging.info(f"--- Time: {t}, Agent: {agent.id}, Role: {agent.role.name} ---")
        # logging.info(f"Full Agent List State: {[(agent.id,agent.x,agent.y,agent.role.name) for agent in agents]}")
        # print("\n\n\n")
        mover.move(agent,agents,t)
        communicator.sense(agent,agents, lost_agents,t)
        behavior_changer.drop_decoy_found(agent,self.decoys)
        behavior_changer.drop_decoy_explored(agent,self.decoys)
        memorizer.forget_seen_neighbors(agent,t,forget_frequency)
        memorizer.update_roles(agent)
        return agents