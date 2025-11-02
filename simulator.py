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
        fig, ax = plt.subplots(figsize=(8, 8))   # one axes for everything
        for t in range(simulation_length):
            self.schedule[t] = []
            for i in range(len(agents)):
                agents = self.step(agents[i],agents,lost_agents,t, exploration_period, mover, communicator, memorizer, behavior_changer, forget_frequency)
                explorers= [explorers.role == Role.EXPLORER for explorers in agents]
                if sum(explorers) == required_explorers and not planned_routes:
                    planner.plan_routes(agents,exploration_radius,exploration_period)
                    planned_routes = True
                    explorers = [explorers for explorers in agents if explorers.role == Role.EXPLORER]
                    # Add planning visualization from plan_routes
                    for agent in explorers:
                        if agent.centroids:
                            positions = np.array([[p.x, p.y] for p in agent.centroids])
                            ax.scatter(positions[:, 0], positions[:, 1], color='purple', marker='*')
                            ax.scatter(agent.exploration_centroid.x, agent.exploration_centroid.y, color='red', marker='X', s=150)
                            for pos in positions:
                                circle = plt.Circle((pos[0], pos[1]), agent.sensing_radius, fill=False, color='blue', alpha=0.3)
                                ax.add_patch(circle)
                    nest_circle = plt.Circle((0, 0), agents[0].nest_radius, color='black', fill=True, alpha=0.2)
                    ax.add_patch(nest_circle)
                self.schedule[t].extend([agents[i].id,agents[i].x,agents[i].y,agents[i].role.value])
            if t == 200:
                import seaborn as sns
                explorers= [(explorers.role.value,explorers.id) for explorers in agents if explorers.role == Role.EXPLORER]
                df = pd.DataFrame.from_dict(self.schedule).T.to_numpy()
                df = df.reshape(num_agents*(t+1),4)
                df = pd.DataFrame(df)
                df.columns = ['id', 'x', 'y', 'role']
                df['time'] = np.repeat(np.arange(t + 1), num_agents)
                
                # Filter out the first 10 steps from the DataFrame
                # df = df[(df['role'] == 1) | (df['role'] == 3)]
                # df = df[(df['time'] >= 100)]
                # Use a categorical color palette for agents
                palette = sns.color_palette("tab20", n_colors=num_agents)
                
                # Role 1: EXPLORER, Role 2: COMMUNICATOR, Role 3: RESCUE
                # Define a different marker for each role
                marker_map = {1: "o", 2: "s", 3: "X"}
                
                # Create a scatter plot with color per agent and marker per role
                ax = sns.scatterplot(data=df, x='x', y='y',
                                hue='id', palette=palette,
                                style='role', markers=marker_map,
                                s=100,ax=ax) # s is marker size
                
                # Find the last position for each agent
                last_positions_df = df.loc[df.groupby('id')['time'].idxmax()]
                # Add a circular patch with radius 2 to the last step of each agent
                for _, row in last_positions_df.iterrows():
                    circle = plt.Circle((row['x'], row['y']), radius=2, color='gray', fill=False, linestyle='--', alpha=0.8)
                    ax.add_patch(circle)
                

                plt.xlabel("x-position")
                plt.ylabel("y-position")
                plt.show()
            # print([(agent.id,agent.x,agent.y,agent.role) for agent in agents])
            
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
        logging.info(f"--- Time: {t}, Agent: {agent.id}, Role: {agent.role.name}, Finished Exploring: {agent.finished_exploring} ---")
        # logging.info(f"Full Agent List State: {[(agent.id,agent.x,agent.y,agent.role.name) for agent in agents]}")
        print("\n")
        mover.move(agent,agents,t)
        communicator.sense(agent,agents, lost_agents,t)
        behavior_changer.drop_decoy_found(agent,self.decoys)
        behavior_changer.drop_decoy_explored(agent,self.decoys)
        if t > 5:
            memorizer.forget_seen_neighbors(agent)
        if t> 3:
            memorizer.forget_stored_positions(agent)
        memorizer.update_roles(agent)
        return agents