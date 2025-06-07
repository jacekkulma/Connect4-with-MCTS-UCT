import numpy as np
import random
import os
from connect4_model_vs_model import play_game
from mcts_interface import get_mcts_move
from mcts_pb_interface import get_mcts_move_progressive_bias, get_mcts_move_progressive_widening, get_mcts_move_dynamic_exploration
from utils import get_random_move, get_heuristic_move
from itertools import combinations_with_replacement, combinations
from collections import defaultdict
import csv

output_folder = "tests"
os.makedirs(output_folder, exist_ok=True)

models = {
    "Random": get_random_move,
    "Heuristic": get_heuristic_move,
    "Base MCTS": get_mcts_move,
    "Progressive Bias": get_mcts_move_progressive_bias,
    "Progressive Wideness": get_mcts_move_progressive_widening,
    "Dynamic Exploration": get_mcts_move_dynamic_exploration
}

model_names = {v: k for k, v in models.items()}
func_name_to_model_name = {v.__name__: k for k, v in models.items()}

# models = [get_random_move, get_mcts_move]
# seeds = [42, 1234, 987654, 20240406, 777, 314159, 8675309, 99999, 1337, 55555]
# seeds = [42, 1234]
seeds = [
    42, 1234, 987654, 20240406, 777, 314159, 8675309, 99999, 1337, 55555,
    88888, 271828, 123456, 7654321, 22222, 31415, 1618033, 8080, 9001, 42069,
    67890, 101010, 111111, 121212, 98765, 20250606, 696969, 777777, 112358, 1729
]

def main(skip_self_comparison=True, model_to_compare=None, opponent_model=None) -> None:
    model_funcs = list(models.values())
    # Skip comparing the model with itself, if the flag is set
    if skip_self_comparison and model_to_compare is None:
        models_to_compare = combinations(model_funcs, 2)
    else:
        models_to_compare = combinations_with_replacement(model_funcs, 2)

    if model_to_compare is not None and opponent_model is None:
        chosen_model = models[model_to_compare]
        model_funcs = [func for name, func in models.items() if name != model_to_compare]
        models_to_compare = [(chosen_model, other) for other in model_funcs]
    elif opponent_model is not None:
        model_func_1 = models[model_to_compare]
        model_func_2 = models[opponent_model]
        models_to_compare = [(model_func_1, model_func_2)]

    # print(list(models_to_compare))
    # import sys
    # sys.exit()
    for model1, model2 in models_to_compare:
        print(f"Comparing models: {model_names[model1]} vs {model_names[model2]}")
        # Count wins and turns for each model inside seeds loop
        win_counter = defaultdict(int)
        turn_counter = defaultdict(list)

        for seed in seeds:
            random.seed(seed)
            ai1_moves, ai2_moves, winner, winner_alg = play_game(model1, model2, console_output=False, pygame_window=False)
            # win_counter[winner_alg] += 1
            # turn_counter[winner_alg].append(max(ai1_moves, ai2_moves))
            if winner_alg in func_name_to_model_name:
                name = func_name_to_model_name[winner_alg]
                win_counter[name] += 1
                turn_counter[name].append(max(ai1_moves, ai2_moves))

        name1 = model_names[model1]
        name2 = model_names[model2]
    
        file_name = f"{name1.replace(' ', '_')}_vs_{name2.replace(' ', '_')}.csv"
        with open(os.path.join(output_folder, file_name), "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["name", "wins", "avg_turns", "turns"])
            for model in [model1, model2]:
                name = model_names[model]
                wins = win_counter.get(name, 0)
                turns = turn_counter.get(name, [])
                avg_turns = sum(turns) / len(turns) if turns else 0

                # turns_str = ";".join(str(t) for t in turns)
                writer.writerow([name, wins, f"{avg_turns:.2f}", turns])


if __name__ == "__main__":
    ## compares 2 specified models with eachother (here Heuristic vs Base MCTS)
    main(model_to_compare="Heuristic", opponent_model="Base MCTS")

    ## compares specified model (here Heuristic) with all other models
    # main(model_to_compare="Heuristic")

    ## compares all unique combinations of model pairs
    # main()

    ## compares all unique combinations of model pairs including coparing each model with itself
    # main(skip_self_comparison=False) 
