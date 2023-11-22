import torch
from .agent import Agent
from .swarm import Swarm
from .experience_buffer import Experience


def compute_swarm_loss(swarm: Swarm, experience: Experience, discount_rate: float = 0.99):

    gamma = discount_rate
    states, actions, rewards, next_states, is_done = experience

    # shape: [batch_size, n_swarm_agents, encoded_state_size]
    # print(states.shape)
    states = swarm.transfrom_states(states[:,swarm.agent_indexes])
    next_states = swarm.transfrom_states(next_states[:,swarm.agent_indexes])
    
    # shape: [batch_size, n_swarm_agents]
    actions = torch.tensor(actions[:,swarm.agent_indexes], dtype = torch.int64)    
    rewards = torch.tensor(rewards[:,swarm.agent_indexes], dtype = torch.float32)
    
    # shape: [batch_size, 1]
    is_done = torch.tensor(is_done, dtype = torch.uint8)
    is_done = is_done.reshape(*is_done.shape, 1)
    
    # shape: [batch_size, n_swarm_agents, n_actions]
    # print(states.shape)
    predicted_qvalues = torch.stack([
        agent.policy(states[:,agent.index]) for agent in swarm.squad], dim = 1)

    # shape: [batch_size, n_swarm_agents, n_actions]
    predicted_next_qvalues = torch.stack([
        agent.target(next_states[:,agent.index]) for agent in swarm.squad], dim = 1)

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


def compute_loss(agent: Agent, e: Experience, discount_rate: float = 0.99):

    gamma = discount_rate

    states, actions, rewards, next_states, is_done = \
        (e.states, e.actions, e.rewards, e.next_states, e.is_done)
    
    batch_size, agent_size, v, v, f = states.shape
    states = states[:,agent.index,].reshape((batch_size, v*v*f))

    actions = actions[:,agent.index,]
    rewards = rewards[:,agent.index,]

    next_states = next_states[:,agent.index,].reshape((batch_size, v*v*f))

    states = torch.tensor(states, dtype=torch.float32)    # shape: [batch_size, *state_shape]
    actions = torch.tensor(actions, dtype=torch.int64)    # shape: [batch_size]
    rewards = torch.tensor(rewards, dtype=torch.float32)  # shape: [batch_size]

    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, dtype=torch.float32)

    is_done = torch.tensor(is_done, dtype=torch.uint8)

    # get q-values for all actions in current states
    predicted_qvalues = agent.policy(states)  # shape: [batch_size, n_actions]

    # compute q-values for all actions in next states
    predicted_next_qvalues = agent.target(next_states)  # shape: [batch_size, n_actions]

    # select q-values for chosen actions # shape: [batch_size]
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    # compute v(next_states) using predicted next q-values
    next_state_values = torch.max(predicted_next_qvalues, dim = -1)[0]

    target_qvalues_for_actions = rewards + gamma * next_state_values

    target_qvalues_for_actions = torch.where(
        is_done, rewards, target_qvalues_for_actions)

    loss = torch.mean((
        predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    return loss