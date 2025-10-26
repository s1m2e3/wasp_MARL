from agent import Role
from math import floor
import random 
import numpy as np
class PlanningService:
    def assign_roles(self,agents, proportion):
        num_explorers = floor(proportion*len(agents))
        for i in range(len(agents)):
            if i < num_explorers:
                agents[i].role = Role.EXPLORER
            else:
                agents[i].role = Role.COMMUNICATOR
    def plan_routes(self,agents, exploration_radius,exploration_period):
        num_explorers = sum(agent.role == Role.EXPLORER for agent in agents)
        degree_partition = round(2*3.14/num_explorers,2)
        counter = 1
        explorer_agents = [explorers for explorers in agents if explorers.role == Role.EXPLORER] 
        for agent in explorer_agents:
            agent.exploration_centroid = (np.cos(counter*degree_partition)*(exploration_radius+agent.nest_radius),np.sin(counter*degree_partition)*(exploration_radius+agent.nest_radius))
            counter += 1
            self.estimate_centroids(agent,counter, exploration_radius,exploration_period)
        
    def estimate_centroids(self,agent,counter, exploration_radius, exploration_period,num_centroids=5,):
        degree_partition = floor((2*3.14)/num_centroids)
        centroid_period = floor(exploration_period/num_centroids)
        for i in range(num_centroids):
            agent.centroids.append((np.cos((i+1)*degree_partition)*(exploration_radius)+agent.exploration_centroid[0],
                                    np.sin((i+1)*degree_partition)*(exploration_radius)+agent.exploration_centroid[1]))
            agent.centroids_schedule.append((i+1)*centroid_period)
        