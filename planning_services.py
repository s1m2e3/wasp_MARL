from agent import Role
from math import floor
import random 
import numpy as np
class PlanningService:
    def assign_roles(self,agents, proportion):
        for agent in agents:
            if random.random() < proportion:
                agent.role = Role.EXPLORER
            else:
                agent.role = Role.COMMUNICATOR
    def plan_routes(self,agents, exploration_radius,exploration_period):
        num_explorers = sum(agent.role == Role.EXPLORER for agent in agents)
        degree_partition = floor(2*3.14/num_explorers)
        counter = 1
        for agent in [explorers for explorers in agents if explorers.role == Role.EXPLORER]:
            if agent.role == Role.EXPLORER:
                agent.exploration_centroid = (np.cos(counter*degree_partition)*exploration_radius,np.sin(counter*degree_partition)*exploration_radius)
                counter += 1
                self.estimate_centroids(agent,counter,degree_partition,exploration_period,exploration_radius)
        pass
    def estimate_centroids(self,agent,counter,degree_partition, exploration_radius, exploration_period,num_centroids=5,):
        sampled_radius = np.random.choice([1,2,3,-1,-2,-3],1)
        degree_partition = floor(2*3.14/num_centroids)
        centroid_period = floor(exploration_period/num_centroids)
        counter = 1
        for _ in range(num_centroids):
            agent.centroids.append((np.cos(counter*degree_partition)*(sampled_radius+exploration_radius),np.sin(counter*degree_partition)*(sampled_radius+exploration_radius)))
            agent.centroids_schedule.append(counter*centroid_period)
            counter += 1