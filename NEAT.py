from random import uniform, random, shuffle, choice
from math import ceil, tanh, exp

class Neat:
    def __init__(self,
                 sizepop : int,
                 evaluation_parralel : int,
                 input_num : int, 
                 output_num : int,
                 func_hidden = "relu",
                 func_out = "tanh",

                 add_node_chance       = 0.03,
                 modify_weight_chance  = 0.7,
                 change_weight_chance  = 0.05,
                 modify_bias_chance    = 0.7,
                 change_bias_chance    = 0.05,
                 connect_nodes_chance  = 0.02,
                 enable_weight_chance  = 0.06,
                 disable_weight_chance = 0.06
                ):

        # Used to use species but without it makes it a lot lot faster
        # to avoid completely ruining an Agent when it gain a node, I make it's bias small, I do the same for new connection
        """
        func : "relu" or "tanh" or "sig"
        """
        #global_id counter
        self.id_counter = input_num+output_num
        self.weights_innovs_shared = {}


        if func_hidden not in ["relu", "tanh", "sig"] or func_out not in ["relu", "tanh", "sig"]:
            raise NameError(f'func_hidden and func_out must be "relu" or "tanh" or "sig" but func_hidden is "{func_hidden}" and func_out is "{func_out}"')

        func_hidden_use = {"relu":self.relu, "tanh":tanh, "sig":self.sig}[func_hidden]
        func_out_use = {"relu":self.relu, "tanh":tanh, "sig":self.sig}[func_out]

        self.population = [Agent(input_num, 
                                 output_num, 
                                 self, 
                                 func_hidden_use, 
                                 func_out_use) 
                          for _ in range(sizepop)]

        self.eval_para = evaluation_parralel
        self.current_agent = 0
        self.sizepop = sizepop


        self.add_node_chance       = add_node_chance
        self.modify_weight_chance  = modify_weight_chance
        self.change_weight_chance  = change_weight_chance
        self.modify_bias_chance    = modify_bias_chance
        self.change_bias_chance    = change_bias_chance
        self.connect_nodes_chance  = connect_nodes_chance
        self.enable_weight_chance  = enable_weight_chance
        self.disable_weight_chance = disable_weight_chance

        




    sig = lambda _, x: (1 / (1 + exp(-x))) if x>=0 else (exp(x) / (1 + exp(x)))
    relu = lambda _, x: max(0, x)

    def get_new_id(self):
        node_id = self.id_counter
        self.id_counter += 1
        return node_id

    def get_new_innov_connection(self, input_id, output_id):
        innov = self.weights_innovs_shared.get((input_id, output_id))
        if innov is not None:
            return innov


        innov = self.id_counter
        self.id_counter += 1
        self.weights_innovs_shared[(input_id, output_id)] = innov
        return innov
        
        

    

    def forward(self, inputs:list[list[float]]):
        return [
            self.population[self.current_agent + index].forward(
                inputs[index]
            )
            for index in range(
                min(self.eval_para, len(self.population))
            )
        ]

    def selection(self, fitnesses = list[float]):
        if len(fitnesses) != self.sizepop:
            raise ValueError(f"size of fitness array must be equal to the size of the population but : len(fitness) = {len(fitnesses)} and size of population = {self.sizepop}")


        ranked = sorted(range(self.sizepop),
                        key= lambda i: fitnesses[i],
                        reverse = True)

        bests = ranked[:self.sizepop//2]
        worsts= ranked[self.sizepop//2:]

        for index in worsts:
            parent1 = self.population[choice(bests)]
            parent2 = self.population[choice(bests)]
            self.population[index].reproduce(
                parent1,
                parent2,
                add_node_chance       = self.add_node_chance,
                modify_weight_chance  = self.modify_weight_chance,
                change_weight_chance  = self.change_weight_chance,
                modify_bias_chance    = self.modify_bias_chance,
                change_bias_chance    = self.change_bias_chance,
                connect_nodes_chance  = self.connect_nodes_chance,
                enable_weight_chance  = self.enable_weight_chance,
                disable_weight_chance = self.disable_weight_chance)


    def new_batch(self):
        self.current_agent += self.eval_para








class Agent:
    def __init__(self, input_num, output_num, neat, hidden_func = tanh, out_func = tanh, modif_amp = 0.1, init_amp = 1., init_amp_small = 0.5):
        self.neat = neat

        self.nodes = []

        self.weights = {}

        self.bias = {}
        self.modif_amp = modif_amp
        self.init_amp  = init_amp
        self.init_amp_small = init_amp_small

        self.hidden_func = hidden_func
        self.out_func    = out_func


        current_id_node = 0
        for layer in [input_num, output_num]:
            self.nodes.append([])
            for _ in range(layer):
                self.nodes[-1].append(current_id_node)
                current_id_node += 1


        for input_id in self.nodes[0]:
            self.weights[input_id] = {}
            for output_id in self.nodes[1]:
                #weights, then if enabled
                self.weights[input_id][output_id] = [uniform(-self.init_amp, self.init_amp), True, self.neat.get_new_innov_connection(input_id, output_id)]


        for current_id_node in self.nodes[1]:
            self.bias[current_id_node] = uniform(-self.init_amp, self.init_amp)

        #flatten nodes
        self.inputs_ids = self.nodes[0]
        self.output_ids = self.nodes[1]

        self.nodes = list(zip(self.nodes[0], ['inp']*len(self.nodes[0]))) + list(zip(self.nodes[1], ['out']*len(self.nodes[1])))


    
    def forward(self, input_list: list[float]):
        #set value for node

        dict_NN = {node_id: 0.0 for node_id, _ in self.nodes}

        for index, node_id in enumerate(self.inputs_ids):
            dict_NN[node_id] = input_list[index]
     

        hidden_nodes = [node_id for node_id in self.nodes if type(node_id[1])==int]
        node_to_process = []
        for node_id in sorted(hidden_nodes, key = lambda x: x[1]):
            while len(node_to_process) <= node_id[1]:
                node_to_process.append([])
            node_to_process[-1].append(node_id[0])

        
        node_to_process = [self.inputs_ids] + node_to_process
     
        #just to exist if unreachable exit node:
        for index_layer, layer in enumerate(node_to_process):
            for node_id in layer:

                if index_layer != 0:
                    dict_NN[node_id] = self.hidden_func(dict_NN.get(node_id, 0.0) + self.bias.get(node_id, 0.0))
     
     
                #if no weight continue
                if node_id not in self.weights.keys() or self.weights.get(node_id) is None:
                    continue
     
     
                #sum over
                for connect_node_id, connection_data in self.weights[node_id].items():
                    weight_val, enabled, _ = connection_data
                    if enabled:
                        dict_NN[connect_node_id] += weight_val * dict_NN[node_id]
     
     
        #Sort and get final layer
        output_list = [self.out_func(dict_NN[node_id] + self.bias.get(node_id, 0.0)) for node_id in self.output_ids]
     
        return output_list

    
    #TO DO : fix nug _reproduce_nodes
    def _reproduce_nodes(self, agent1:Agent, agent2:Agent):
        #common node
        self.nodes = list(set(agent2.nodes).intersection(agent1.nodes))

        sim_diff = list(set(agent2.nodes).symmetric_difference(set(agent1.nodes)))
        shuffle(sim_diff)

        if random()>0.5:
            self.nodes += sim_diff[0:ceil(len(sim_diff)/2)]
        else:
            self.nodes += sim_diff[0:int(len(sim_diff)/2)]


    def _reproduce_weights(self, agent1:Agent, agent2:Agent):
        self.weights = {}


        all_targets = set(agent2.weights.keys()).union(agent1.weights.keys())
        child_nodes_set = set([node[0] for node in self.nodes])


        for output_id in all_targets:
            if output_id not in child_nodes_set: continue


            #get source node both parent
            parent1_inputs = set(  agent2.weights.get(output_id, {}).keys())
            parent2_inputs = set(agent1.weights.get(output_id, {}).keys())

            all_inputs = parent1_inputs.union(parent2_inputs)

            for input_id in all_inputs:
                if input_id not in child_nodes_set: continue


                #get connection
                parent1_weight =   agent2.weights.get(output_id, {}).get(input_id)
                parent2_weight = agent1.weights.get(output_id, {}).get(input_id)


                list_weights_choose = list(filter(lambda x: x is not None, [parent1_weight, parent2_weight])) 
                chosen_weight = choice(list_weights_choose)


                if chosen_weight is None: continue

                if output_id not in self.weights:
                    self.weights[output_id] = {}

                self.weights[output_id][input_id] = list(chosen_weight).copy()

            

    def _reproduce_bias(self, agent1:Agent, agent2:Agent):
        self.bias = {}
        

        for node_id in self.nodes:
            node_id = node_id[0]
            if node_id in self.inputs_ids:
                continue

            #get both bias
            parent1_bias = agent2.bias.get(node_id, None)
            parent2_bias = agent1.bias.get(node_id, None)



            if not (parent1_bias is None or parent2_bias is None):
                #remove None
                list_bias_choose = list(filter(lambda x: x is not None, [parent1_bias, parent2_bias])) 
                self.bias[node_id] = choice(list_bias_choose)

            else:
                self.bias[node_id] = uniform(-self.init_amp, self.init_amp)


    def _add_node(self):
        new_node_id = self.neat.get_new_id()

        
        
        #small bias to not perturb system
        self.bias[new_node_id] = uniform(-self.init_amp_small, -self.init_amp_small)

        possible_target = [target for target in self.weights.keys() if self.weights[target]]
        if not possible_target:
            return

        node_to_link = choice(possible_target)
        layer_of_node_link = sorted(self.nodes, key = lambda x: x[0] == node_to_link, reverse=True)[0][1]

        layer_of_node_link = 0 if layer_of_node_link == 'inp' else layer_of_node_link
        
        self.nodes.append((new_node_id, layer_of_node_link+1))

        

        #assume every node is connected to a least one other node
        key_input_rand = choice(list(self.weights[node_to_link].keys()))


        #remove old connection
        old_weight = (self.weights[node_to_link].pop(key_input_rand))[0]

        #new connection
        if new_node_id not in self.weights:
            self.weights[new_node_id] = {}


        self.weights[new_node_id][key_input_rand] = [1.0, True, self.neat.get_new_innov_connection(new_node_id, key_input_rand)]

        #auto enable weight
        self.weights[node_to_link][new_node_id] = [old_weight, True, self.neat.get_new_innov_connection(node_to_link, new_node_id)]
        

    def _modify_weight(self):
        key1_rand = list(self.weights.keys())
        if not key1_rand: return
        key1_rand = choice(key1_rand)

        key2_rand = [item for item in list(self.weights[key1_rand].keys()) if self.weights[key1_rand][item][1]]
        if not key2_rand: return
        key2_rand = choice(key2_rand)

        self.weights[key1_rand][key2_rand][0] = max(min(
            self.weights[key1_rand][key2_rand][0] + uniform(-self.modif_amp, self.modif_amp)
            , 30), -30)

    def _change_weight(self):
        key1_rand = choice(list(self.weights.keys()))
        key2_rand = choice(list(self.weights[key1_rand].keys()))

        self.weights[key1_rand][key2_rand] = [uniform(-self.init_amp, self.init_amp), True, self.weights[key1_rand][key2_rand][2]]


    def _modify_bias(self):
            key1_rand = choice(list(self.bias.keys()))

            self.bias[key1_rand] = max(min(
                self.bias[key1_rand] + uniform(-self.modif_amp, self.modif_amp)
                , 30), -30)
    
    def _change_bias(self):
        key1_rand = choice(list(self.bias.keys()))

        self.bias[key1_rand] = uniform(-self.init_amp, self.init_amp)


    
    def _connect_weights(self):

        all_sources = [node if node != 'inp' else 0 for node in self.nodes if node[1] != 'out']
        if not all_sources : return
        shuffle(all_sources)
        
        all_targets = [node for node in self.nodes if node[1] != 'inp']
        if not all_targets : return
        shuffle(all_targets)


        for source in all_sources:
            for target in all_targets:

                if target[1] != 'out':
                    if source[0]>=target[1]: continue 

                if source[0] == target[0]: continue

                if target in self.weights.get(source, {}): continue


                #add the weight
                if source not in self.weights:
                    self.weights[source] = {}
                    
                self.weights[source][target] = [
                    uniform(-self.init_amp_small, self.init_amp_small),
                    True
                ]



    def _disable_enable_weight(self, set_to_bool:bool):
        weights = list(self.weights.keys())
        shuffle(weights)
        for key1 in weights:
            all_items = list(self.weights[key1].keys())
            shuffle(all_items)
            for key2 in all_items:
                if self.weights[key1][key2][1] == set_to_bool: continue
                self.weights[key1][key2][1] = set_to_bool
                return



    def reproduce(self, agent1:Agent, agent2:Agent,
                  add_node_chance       = 0.03,
                  modify_weight_chance  = 0.7,
                  change_weight_chance  = 0.05,
                  modify_bias_chance    = 0.7,
                  change_bias_chance    = 0.05,
                  connect_nodes_chance  = 0.02+0.02,
                  enable_weight_chance  = 0.06,
                  disable_weight_chance = 0.06
                  ):

        #nodes
        self._reproduce_nodes(agent1, agent2)

        #weights
        self._reproduce_weights(agent1, agent2)

        #bias
        self._reproduce_bias(agent1, agent2)


        #mutate
        #node
        if random()<add_node_chance: self._add_node()

        #weights
        if random()<modify_weight_chance: self._modify_weight()
        if random()<change_weight_chance: self._change_weight()
        if random()<connect_nodes_chance: self._connect_weights()
        if random()<enable_weight_chance: self._disable_enable_weight(True)
        if random()<disable_weight_chance: self._disable_enable_weight(False)
      
        #bias
        if random()<modify_bias_chance: self._modify_bias()
        if random()<change_bias_chance: self._change_bias()




    def distance(self: Agent, agent2: Agent, c1=1.0, c2=1.0, c3=0.4):
       
        temp = [list(inner.values()) for inner in self.weights.values()]
        genes1 = []
        for item in temp: genes1 += item
        len_weights_1 = len(genes1)
        genes1 = {g[2]: g[0] for g in genes1}

        
        temp = [list(inner.values()) for inner in agent2.weights.values()]
        genes2 = []
        for item in temp: genes2 += item
        len_weights_2 = len(genes2)
        genes2 = {g[2]: g[0] for g in genes2}


        innovations1 = set(genes1.keys())
        innovations2 = set(genes2.keys())

        matching = innovations1 & innovations2
        disjoint = (innovations1 ^ innovations2)
        excess = set()

        max_innov1 = max(innovations1) if innovations1 else 0
        max_innov2 = max(innovations2) if innovations2 else 0
        max_innov = min(max_innov1, max_innov2)

        # Separate excess from disjoint
        for innov in disjoint.copy():
            if innov > max_innov:
                excess.add(innov)
                disjoint.remove(innov)

        # Weight difference of matching genes
        if matching:
            weight_diff = sum(
                abs(genes1[i] - genes2[i]) for i in matching
            )
            avg_weight_diff = weight_diff / len(matching)
        else:
            avg_weight_diff = 0

        # Normalize by N
        N = max(len_weights_1, len_weights_2)
        if N < 20:  # NEAT typically uses 1 if N < 20
            N = 1

        delta = (c1 * len(excess)) / N + (c2 * len(disjoint)) / N + c3 * avg_weight_diff
        return delta