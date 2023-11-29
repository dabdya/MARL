import torch
from ..communication.core import Swarm
from .experience_buffer import Experience


def compute_swarm_loss(swarm: Swarm, experience: Experience, discount_rate: float = 0.99):

    gamma = discount_rate
    states, actions, rewards, next_states, is_done = experience

    # shape: [batch_size, n_swarm_agents, encoded_state_size]
    # print(states.shape)
    states = swarm.transform_states(states[:,swarm.indexes])
    next_states = swarm.transform_states(next_states[:,swarm.indexes])
    
    # shape: [batch_size, n_swarm_agents]
    actions = torch.tensor(actions[:,swarm.indexes], dtype = torch.int64)
    rewards = torch.tensor(rewards[:,swarm.indexes], dtype = torch.float32)
    
    # shape: [batch_size, 1]
    is_done = torch.tensor(is_done, dtype = torch.uint8)
    is_done = is_done.reshape(*is_done.shape, 1)
    
    # shape: [batch_size, n_swarm_agents, n_actions]
    # print(states.shape)
    predicted_qvalues = torch.stack([
        agent.policy(states[:,i]) for i, agent in enumerate(swarm.squad)], dim = 1)

    # shape: [batch_size, n_swarm_agents, n_actions]
    predicted_next_qvalues = torch.stack([
        agent.target(next_states[:,i]) for i, agent in enumerate(swarm.squad)], dim = 1)

    # select q-values for chosen actions 
    # shape: [batch_size, n_swarm_agents]
    predicted_qvalues_for_actions = torch.stack([
        q[range(len(a)),a] for q, a in zip(predicted_qvalues, actions)])

    # compute v(next_states) using predicted next q-values 
    # shape: [batch_size, n_swarm_agents]
    next_state_values = torch.max(predicted_next_qvalues, dim = -1)[0]

    # shape: [batch_size, n_swarm_agents]
    target_qvalues_for_actions = rewards + gamma * next_state_values

    target_qvalues_for_actions = torch.where(
        is_done, rewards, target_qvalues_for_actions)

    # shape: [n_swarm_agents]
    loss = torch.mean((
        predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2, dim = 0)

    return torch.mean(loss)
