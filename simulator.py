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
        self.exploration_centroids = []
        self.exploration_centroids_expansion ={}
    def run_simulation(self,agents,lost_agents,simulation_length,proportion_explorers,exploration_radius,exploration_period,num_agents,forget_frequency, num_lost_agents, radius_decoys):
        # --- Configure logging ---
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - T%(levelname)s - %(message)s')
        # --- Instantiate services once before the simulation loop ---
        self.radius_decoys = radius_decoys
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
                agents = self.step(agents[i],agents,lost_agents,self.decoys,t, exploration_period, mover, communicator, memorizer, behavior_changer, forget_frequency)
                explorers= [explorers.role == Role.EXPLORER for explorers in agents]
                if sum(explorers) == required_explorers and not planned_routes:
                    planner.plan_routes(agents,exploration_radius,exploration_period)
                    planned_routes = True
                    explorers = [explorers for explorers in agents if explorers.role == Role.EXPLORER]
                    # Add planning visualization from plan_routes
                    for agent in explorers:
                        if agent.centroids:
                            self.exploration_centroids.append(agent.exploration_centroid)
                            self.exploration_centroids_expansion[agent.id]= agent.centroids
                self.schedule[t].extend([agents[i].id,agents[i].x,agents[i].y,agents[i].role.value,agents[i].found_state])
            
    def step(self,agent,agents,lost_agents, decoys,t, exploration_period, mover, communicator, memorizer, behavior_changer, forget_frequency=5):
        logging.info(f"--- Time: {t}, Agent: {agent.id}, Role: {agent.role.name}, Finished Exploring: {agent.finished_exploring} ---")
        mover.move(agent,agents,decoys,t)
        communicator.sense(agent,agents, lost_agents,t)
        behavior_changer.drop_decoy_found(agent,self.decoys,self.radius_decoys)
        behavior_changer.drop_decoy_explored(agent,self.decoys,self.radius_decoys)
        behavior_changer.saturate_decoys(agents,decoys)
        if t > 3:
            memorizer.forget_seen_neighbors(agent)
        if t> 3:
            memorizer.forget_stored_positions(agent)
        memorizer.update_leader(agent)
        memorizer.update_roles(agent)
        return agents