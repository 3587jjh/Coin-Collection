import random
import numpy as np
from game import Game

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="your model file",
                    type=str, default="model_q.pkl")
					
    play_num = 500
    score_list = []
    
    args = parser.parse_args()
    
    model_fn = args.model
    print("load model: ", model_fn)
    from q_learning import load_q, get_action
    Q, n_ep, params  = load_q(model_fn)
    
    for N in range(play_num):
        g = Game(show_game=False)
        is_done, counter, score, game_info = g.reset()
        
        while not is_done:
            action = get_action(Q, counter, score, game_info, params)
            is_done, counter, score, game_info = g.step(action, user_input=False)
        
        score_list.append(score)
        if N % 50 == 0:
            print(N, score)	
    score_list = sorted(score_list)
    score_list = score_list[100:400]
    print(int(np.mean(score_list)))
    
if __name__ == "__main__":
    main()
