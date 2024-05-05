import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import traci
import os

# Define constants
LOW_BATTERY_THRESHOLD = 30  # Example threshold for low battery
NUM_STATIONS = 4  # Number of charging stations

# Environment
class ChargingEnv:
    def __init__(self):
        self.battery_capacity = 100  # Maximum battery capacity
        self.battery = self.battery_capacity
        self.station_locations = np.random.randint(0, 100, size=(NUM_STATIONS, 2))
        self.max_distance = None

    def reset(self):
        traci.start(["sumo", "-c", "C:/Users/cl502_08/Sumo/trial.sumocfg", "--no-step-log", "true"])
        self.battery = self.battery_capacity
        self.max_distance = None
        return self._get_observation()

    def _get_observation(self):
        vehicle_location = traci.vehicle.getPosition("veh0")
        observation = np.concatenate([vehicle_location, [self.battery]])
        return observation

    def step(self, action):
        # Move the vehicle
        traci.vehicle.moveToXY("veh0", action[0], action[1])
        # Calculate distance to each charging station
        vehicle_location = traci.vehicle.getPosition("veh0")
        distances = np.linalg.norm(vehicle_location - self.station_locations, axis=1)
        # Find nearest charging station
        nearest_station_idx = np.argmin(distances)
        nearest_station_distance = distances[nearest_station_idx]
        # Calculate reward
        if self.battery > nearest_station_distance:
            reward = -nearest_station_distance  # Move towards charging station
        else:
            reward = -self.max_distance  # Penalty for low battery
        # Update battery level
        self.battery -= nearest_station_distance
        # Check if charging needed
        if self.battery <= 0:
            self.battery = self.battery_capacity
        # Check if episode is done
        done = False
        if np.array_equal(vehicle_location, self.station_locations[nearest_station_idx]):
            done = True
        # Update max_distance
        if self.max_distance is None or nearest_station_distance > self.max_distance:
            self.max_distance = nearest_station_distance
        return self._get_observation(), reward, done, {}

# Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Assuming continuous action space
        return x

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Soft Actor-Critic (SAC) Algorithm
class SAC:
    def __init__(self):
        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None

    def initialize_networks(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0003)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mean = self.actor(state)
        action_std = torch.tensor(1.0)  # Assuming fixed standard deviation for simplicity
        action_dist = Normal(action_mean, action_std)
        action = action_dist.sample()
        return action.squeeze().detach().cpu().numpy()  # Convert to numpy array

    def train(self, state, action, next_state, reward, done, discount=0.99, tau=0.005):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)  # Changed here
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        target_Q = reward + (1 - done) * discount * self.critic(next_state)
        current_Q = self.critic(state)
        critic_loss = F.mse_loss(current_Q, target_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


# Main simulation loop
def main():
    print("The simulation is running...", flush=True)
    env = ChargingEnv()
    sac_agent = SAC()
    sac_agent.initialize_networks(state_dim=3, action_dim=2)

    for episode in range(100):
        state = env.reset()
        done = False
        while not done:
            # Control vehicle
            battery_level = traci.vehicle.getFuelConsumption("veh0")
            if battery_level < LOW_BATTERY_THRESHOLD:
                navigate_to_charging_station(env, sac_agent, state)
            else:
                action = sac_agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                sac_agent.train(state, action, next_state, reward, done)
                state = next_state

    traci.close()
    print("Simulation ended.")


def navigate_to_charging_station(env, agent, state):
    nearest_station_idx = np.argmin(np.linalg.norm(traci.vehicle.getPosition("veh0") - env.station_locations, axis=1))
    charging_station_location = env.station_locations[nearest_station_idx]
    action = charging_station_location
    agent.train(state, action, state, 0, False)

if __name__ == "__main__":
    main()
