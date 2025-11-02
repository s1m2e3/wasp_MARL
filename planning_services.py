from agent import Role, Position
from math import floor
import random 
import numpy as np
import logging

class PlanningService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def assign_roles(self,agents, proportion):
        num_explorers = floor(proportion*len(agents))
        for i in range(len(agents)):
            if i == 0:
                agents[i].role = Role.EXPLORER
                agents[i].communication_threshold = (num_explorers-1)*100
            else:
                agents[i].role = Role.COMMUNICATOR
                agents[i].communication_threshold = 0

    def plan_routes(self,agents, exploration_radius,exploration_period):
        num_explorers = sum(agent.role == Role.EXPLORER for agent in agents)       
        degree_partition = round(2*3.14/num_explorers,2)
        counter = 1
        explorer_agents = [explorers for explorers in agents if explorers.role == Role.EXPLORER]
        communicator_agents = [communicator for communicator in agents if communicator.role == Role.COMMUNICATOR]
        for agent in explorer_agents:
            agent.exploration_centroid = Position(np.cos(counter*degree_partition)*(exploration_radius+agent.nest_radius),np.sin(counter*degree_partition)*(exploration_radius+agent.nest_radius))
            counter += 1
            self.estimate_centroids(agent,exploration_radius,exploration_period)
        for agent in communicator_agents:
            agent.nest_radius += random.uniform(-2.0,0.5)
        
    def estimate_centroids(self,agent, exploration_radius, exploration_period,num_centroids=5,):
        degree_partition = np.round((2*3.14)/num_centroids,2)
        centroid_period = floor(exploration_period/num_centroids)
        for i in range(num_centroids):
            agent.centroids.append(Position(np.cos((i+1)*degree_partition)*(exploration_radius)+agent.exploration_centroid.x,
                                    np.sin((i+1)*degree_partition)*(exploration_radius)+agent.exploration_centroid.y))
            agent.centroids_schedule.append((i+1)*centroid_period)
        