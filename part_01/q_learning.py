# Artificial Intelligence for business
# Optimizing Warehouse Flows with Q-Learning

# Import the libraries
import numpy as np

# Setting the parameters gamma and alpha for Q-Learning
gamma = 0.75  # discaout
alpha = 0.9  # learning rate

# PART 1 - DEFINING THE ENVRONMENT

# define the states
location_to_state = {'A': 0,
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}

# defining the actions
actions = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

# defining the rewards
R = np.array([
    # A  B  C  D  E  F  G  H  I  J  K  L
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # B
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # C
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # D
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # E
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # F
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # G
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],  # H
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # I
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # J
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # K
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]   # L
])

# PART 2 BUILD THE AI SOLUTIONS
# Initializaing the q values
Q = np.array(np.zeros([12, 12]))

# Implement the Q-Learning Process
for i in range(1000):
    current_state = np.random.randint(0, 12)
    playable_action = []
    for j in range(12):
        if(R[current_state, j] > 0):
            playable_action.append(j)
    next_state = np.random.choice(playable_action)
    TD = R[current_state, next_state] + gamma + \
        Q[next_state, np.argmax(Q[next_state, ])] - \
        Q[current_state, next_state]
    Q[current_state, next_state] += alpha*TD
