import random
import numpy as np


class Experience:
    def __init__(self, states, actions, rewards, next_states, is_done):
        self._experience = [
            states, actions, rewards, next_states, is_done
        ]
        self.__pointer = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.__pointer >= len(self._experience):
            self.__pointer = 0
            raise StopIteration
        output = self._experience[self.__pointer]
        self.__pointer += 1
        return output


class ExperienceBuffer:
    def __init__(self, size: int):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)
    
    def is_valid(self, playground):
        #TODO
        return True

    def warm_up(self, pg):
        state = pg.new_game()
        while len(self) < self._maxsize:
            pg.play_game(state, self, n_steps = 100)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size: int) -> Experience:
        states, actions, rewards, next_states, dones = [
            [] for _ in range(5)
        ]
        indices = [
            random.randint(0, len(self._storage) - 1)
            for _ in range(batch_size)
        ]

        for idx in indices:
            data = self._storage[idx]
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return Experience(
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
