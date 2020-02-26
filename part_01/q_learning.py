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

state_to_location = {state: location for location,
                     state in location_to_state.items()}

# defining the actions
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

priority_actions = ['G', 'K', 'L', 'J', 'A', 'I', 'H', 'C', 'B', 'D', 'F', 'E']

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


def training(R, interations):
    Q = np.array(np.zeros([12, 12]))
    # Implement the Q-Learning Process
    for i in range(interations):
        current_state = np.random.randint(0, 12)
        playable_action = []
        for j in range(12):
            if(R[current_state, j] > 0):
                playable_action.append(j)
        next_state = np.random.choice(playable_action)
        TD = R[current_state, next_state]
        TD += gamma * Q[next_state, np.argmax(Q[next_state, ])]
        TD -= Q[current_state, next_state]

        next_location = state_to_location[next_state]
        Q[current_state, next_state] += alpha*TD
        Q[current_state, next_state] -= priority_actions.index(next_location)
    return Q

# PART 3 - GOING INTO PRODUCTITION
# print(Q.astype(int))


def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    Q = training(R_new, 1000)
    # print(Q.astype(int))
    route = [starting_location]
    next_location = starting_location
    while(next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state, ])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route


# Printing the final route
print('Route:')
print(route('D', 'J'))
