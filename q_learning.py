"""
The original code is from https://github.com/dennybritz/reinforcement-learning/tree/master/TD
"""

import sys
import numpy as np
import itertools
import pickle
from collections import defaultdict
from game import Game

# In our case, we have 3 action (stay, go-left, go-right)
def get_action_num():
    return 3


## this function return policy function to choose the action based on Q value.
def make_policy(Q, epsilon, nA):
    """
    This is the epsilon-greedy policy, which select random actions for some chance (epsilon).
    (Check dennybritz's repository for detail)

    You may change the policy function for the given task.
    """
    def policy_fn(observation):        
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

###############################################################
def reachable(basketPos, item):
    basketPos *= 64
    itemX = 32 + 48*item[1]
    itemY = 48*item[2]

    dist = -1
    if itemX < basketPos:
        dist = basketPos - itemX
    elif basketPos+128 < itemX:
        dist = itemX - (basketPos+128)
    else:
        dist = 0

    leftTime = (480-itemY) // 48
    return dist <= 64*leftTime

def get_state(game_info):
    basketPos, items = game_info
    for item in items:
        if item[0] == 2 and reachable(basketPos, item):
            return (basketPos, item[1])

    for item in items:
        if reachable(basketPos, item):
            return (basketPos, item[1]) 
    return (4, 5)

def get_reward(state, action):
    # action = 0: stay, 1:left, 2:right
    basketPos, itemX = state
    basketPos = basketPos*64
    itemX = 32 + 48*itemX

    if itemX < basketPos:
        if action == 1:
            return 10
        else:
            return -100
    elif basketPos+128 < itemX:
        if action == 2:
            return 10
        else:
            return -100
    else:
        if action == 0:
            return 10
        else:
            return -100

###############################################################
def save_q(Q, num_episode, params, filename="model_q.pkl"):
    data = {"num_episode": num_episode, "params": params, "q_table": dict(Q)}
    with open(filename, "wb") as w:
        w.write(pickle.dumps(data))

        
def load_q(filename="model_q.pkl"):
    with open(filename, "rb") as f:
        data = pickle.loads(f.read())
        return defaultdict(lambda: np.zeros(3), data["q_table"]), data["num_episode"], data["params"]


def q_learning(game, num_episodes, params):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy.
    You can edit those parameters, please speficy your changes in the report.
    
    Args:
        game: Coin drop game environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        Q: the optimal action-value function, a dictionary mapping state -> action values.
    """
    
    epsilon, alpha, discount_factor = params
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(get_action_num()))  
    
    # The policy we're following
    policy = make_policy(Q, epsilon, get_action_num())
    
    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        _, counter, score, game_info = game.reset()
        state = get_state(game_info)
        action = 0
        
        # One step in the environment
        for t in itertools.count():
            # Take a step
            action_probs = policy(get_state(game_info))
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            done, next_counter, next_score, game_info = game.step(action)

            next_state = get_state(game_info)
            # game.step이후에는 action을 취하기 전의 game_info상태이다 (한번 delay됨)
            # reward는 현재 state에서 action을 취했을 때 얻는 보상이므로
            # game.step 이후의 game_info의 state를 이용한다
            reward = get_reward(next_state, action)
            
            counter = next_counter
            score = next_score
            
            """
            this code performs TD Update. (Update Q value)
            You may change this part for the given task.
            """
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
        
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("Episode {}/{} (Score: {})\n".format(i_episode + 1, num_episodes, score), end="")
            sys.stdout.flush()

    return Q

def train(num_episodes, params):
    g = Game(False)
    Q = q_learning(g, num_episodes, params)
    return Q


## This function will be called in the game.py
def get_action(Q, counter, score, game_info, params):
    epsilon = 0 # params[0]
    policy = make_policy(Q, epsilon, 3)
    action_probs = policy(get_state(game_info))
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    return action

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_episode", help="# of the episode (size of training data)",
                    type=int, required=True)
    parser.add_argument("-e", "--epsilon", help="the probability of random movement, 0~1",
                    type=float, default=0.1)
    parser.add_argument("-lr", "--learning_rate", help="learning rate of training",
                    type=float, default=0.1)
    
    args = parser.parse_args()
    
    if args.num_episode is None:
        parser.print_help()
        exit(1)

    # you can pass your parameter as list or dictionary.
    # fix corresponding parts if you want to change the parameters
    
    num_episodes = args.num_episode
    epsilon = args.epsilon
    learning_rate = args.learning_rate
    
    Q = train(num_episodes, [epsilon, learning_rate, 0.5])
    save_q(Q, num_episodes, [epsilon, learning_rate, 0.5])
    
    Q, n, params = load_q()

    
if __name__ == "__main__":
    main()
