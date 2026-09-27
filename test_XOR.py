from NEAT import Neat
from time import time


SIZE_POP = 128
neat = Neat(sizepop = SIZE_POP,
            evaluation_parralel = SIZE_POP,
            input_num = 2,
            output_num = 2,
            func_out = 'sig',
            func_hidden = 'relu',
            add_node_chance       = 0.02,
            modify_weight_chance  = 0.7,
            change_weight_chance  = 0.05,
            modify_bias_chance    = 0.7,
            change_bias_chance    = 0.05,
            connect_nodes_chance  = 0.04,
            enable_weight_chance  = 0.03,
            disable_weight_chance = 0.03)





XOR_INPUTS = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

XOR_TARGETS = [
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0],
]

def evaluate_agent(agent, inputs, targets):
    error = 0.0

    for inp, target in zip(inputs, targets):
        output = agent.forward(inp)

        s = sum(output)
        output = [x / s for x in output]

    
        error += sum(
            (out - tar) ** 2
            for out, tar, in zip(output, target)
        )

    return -error


def print_best_output(neat_class:Neat, input:list[float], fitnesses:list[float] = None):
    if not fitnesses:
        fitnesses = [0 for _ in range(neat_class.sizepop)]

    ranked = sorted(range(neat_class.sizepop),
                key= lambda i: fitnesses[i],
                reverse = True)


    output = neat_class.population[ranked[0]].forward(input)



    output = [out/sum(output) for out in output]
    print(f"{input} -> {output}")



start = time()
print("###############    before training    ###############")
print_best_output(neat, [0,0])
print_best_output(neat, [1,0])
print_best_output(neat, [0,1])
print_best_output(neat, [1,1])
print("#####################################################\n")

for generation in range(1000):
    fitness = [evaluate_agent(agent, XOR_INPUTS, XOR_TARGETS)
               for agent in neat.population]

    neat.selection(fitness)

fitness = [evaluate_agent(agent, XOR_INPUTS, XOR_TARGETS)
           for agent in neat.population]


print("###############    after training    ###############")
print_best_output(neat, [0,0], fitness)
print_best_output(neat, [1,0], fitness)
print_best_output(neat, [0,1], fitness)
print_best_output(neat, [1,1], fitness)
print("#######################################################")

print(f"run time : {time()-start:.2f} s")