#include <iostream>
#include <map>
#include <random>
#include <algorithm>
#include <vector>
#include <cmath>
#include <unordered_set>

struct Node{
    int id;
    int layer;
    double value;
    double bias;
};

struct Connection
{
    double weight;
    bool enabled;
    int innov;
};



class Agent{
// explicit private even if not necessary
public:
    //struct : {node_id, layer->0 = input, -1 = output, 1,2,3,... = layer}
    std::vector<Node> nodes;

    // the data struct for one weight is : 
    // weight:double,  enabled = 0 or 1, innov : int but stored as double
    std::map<int, std::map<int, Connection>> weights;

    int* pid_counter;
    std::map<std::pair<int, int>, int>* innov_connection;
    std::map<std::pair<int, int>, int>* innov_nodes;
    std::mt19937* prng;

    std::vector<double> output_foward;

    //0 -> sig, 1-> tanh,  2 -> relu
    int hid_func;
    //0 -> sig, 1-> tanh,  2 -> relu
    int out_func;

    int AddNode(int layer){
        int size_layer = 0;
        for (Node& node : nodes)
        {
            if (node.layer == layer){
                size_layer++;
            }
        }

        std::pair<int, int> layer_size_pair = {layer, size_layer};
        int new_innov = -1;
        if ((*innov_connection).find(layer_size_pair) == (*innov_connection).end())
        {
            (*pid_counter)++;
            new_innov = *pid_counter;
            (*innov_connection)[layer_size_pair] = new_innov;

        } else
        {
            new_innov = (*innov_connection)[layer_size_pair];
        }


        nodes.push_back(Node {new_innov, layer, 0., random_double()});

        return new_innov;
    }
    int AddConnection(int input_id, int output_id, double weight_to_use = -1, bool use_weight = false){
        std::pair<int, int> inp_out_pair = {input_id, output_id};

        int new_innov = -1;
        if ((*innov_connection).find(inp_out_pair) == (*innov_connection).end())
        {
            (*pid_counter)++;
            new_innov = *pid_counter;
            (*innov_connection)[inp_out_pair] = new_innov;
        } else
        {
            new_innov = (*innov_connection)[inp_out_pair];
        }
    

        if (use_weight){
            weights[input_id][output_id] = Connection {weight_to_use, true, new_innov};
        }
        else{
            weights[input_id][output_id] = Connection {random_double(), true, new_innov};
        }

        return new_innov;
    }

    
    // generate random number from -1 to 1
    double random_double(){
        return std::generate_canonical< double, 128 >(*prng)*2-1;
    };

    // generate random number from 0 to 1
    double random_01(){
        return std::generate_canonical< double, 128 >(*prng);
    };

    double relu(double x){
        return std::max(x, 0.);
    };

    double sig(double x){
        if (x>=0){
            return 1 / (1 + std::exp(-x));
        }
        else{
            return std::exp(x) / (1 + std::exp(x));
        }
    };


private:

    void reproduce_nodes(Agent parent1, Agent parent2){
        nodes.clear();
        std::copy(parent1.nodes.begin(), parent1.nodes.end(), std::back_inserter(nodes));
        std::copy(parent2.nodes.begin(), parent2.nodes.end(), std::back_inserter(nodes));

        /*remove duplicates*/
        std::shuffle(std::begin(nodes), std::end(nodes), *prng);
        std::unordered_set<int> us;
        auto it = nodes.begin();

        while (it != nodes.end()) {
            if (us.find((*it).id) != us.end())
                it = nodes.erase(it);
        

            else {
                us.insert((*it).id);
                it++;
            }
        }
    }

    void reproduce_weight(Agent parent1, Agent parent2){
        weights.clear();

        std::vector<int> input_ids;
        for (auto it = parent1.weights.begin(); it != parent1.weights.end(); ++it)
        {
            input_ids.push_back(it->first);
        }
        for (auto it = parent2.weights.begin(); it != parent2.weights.end(); ++it)
        {
            input_ids.push_back(it->first);
        }




        std::unordered_set<int> already_seen_out;
        std::unordered_set<int> already_seen_inp;
        for (int input_id : input_ids)
        {   
            
            /*skip if duplicate*/ 
            if (already_seen_out.find(input_id) != already_seen_out.end()) { 
                continue;
            }
            already_seen_out.insert(input_id);


            /*if weights[input_id] does not work as entended*/
            /*create vectors of conntection node id from both parent*/
            std::vector<int> keys_parents_inps;
            for (auto it = parent1.weights[input_id].begin(); it != parent1.weights[input_id].end(); ++it)
            {
                keys_parents_inps.push_back(it->first);
            }
            for (auto it = parent2.weights[input_id].begin(); it != parent2.weights[input_id].end(); ++it)
            {
                keys_parents_inps.push_back(it->first);
            }



            already_seen_inp.clear();
            for (int output_id : keys_parents_inps)
            {   
            
                /*skip if duplicate*/ 
                if (already_seen_inp.find(output_id) != already_seen_inp.end()) { 
                    continue;
                }
                already_seen_inp.insert(output_id);


                /*get connection details from parent 1 and 2 if they exist*/
                bool conn_p1_OK = true;
                Connection conn_p1;
                try {
                    conn_p1 = parent1.weights.at(input_id).at(output_id);
                }
                catch(...) {
                    conn_p1_OK = false;
                }

                bool conn_p2_OK = true;
                Connection conn_p2;
                try {
                    conn_p2 = parent2.weights.at(input_id).at(output_id);
                }
                catch(...) {
                    conn_p2_OK = false;
                }

                
                if (!(conn_p1_OK || conn_p2_OK)){
                    continue;
                }
                


                /*select random connection*/
                bool takeconn1 = true;
                if (conn_p1_OK && conn_p2_OK){
                    if (random_double() > 0){
                        takeconn1 = true;
                    } else {
                        takeconn1 = false;
                    }
                } else if(conn_p1_OK){
                    takeconn1 = true;
                } else if (conn_p2_OK){
                    takeconn1 = false;
                }


                if (weights.find(input_id) == weights.end()){
                    weights.insert({input_id,  {}});
                }


                if(takeconn1){
                    weights.at(input_id).insert({output_id, conn_p1});
                }
                else {
                    weights.at(input_id).insert({output_id, conn_p2});
                }

            }
        }
    }

    void reproduce_hidden_func(int parent1_hidfunc, int parent2_hidfunc){
        if (random_01()>0.5){
            hid_func = parent1_hidfunc;
        } else {
            hid_func = parent2_hidfunc;   
        }
    }





    void modify_weight(bool replace = false, double modify_amount = 0.0625){
        auto it1 = weights.begin();
        std::advance(it1, rand() % weights.size());
        int random_key1 = it1->first;


        auto it2 = weights.at(random_key1).begin();
        std::advance(it2, rand() % weights.at(random_key1).size());
        int random_key2 = it2->first;

        if (replace){
            weights.at(random_key1).at(random_key2).weight = random_double();
        } else {
            weights.at(random_key1).at(random_key2).weight += random_double()*modify_amount;
            /*clamp value*/
            weights.at(random_key1).at(random_key2).weight = std::max(std::min(weights.at(random_key1).at(random_key2).weight, 32.), -32.);
        }
    }

    void modify_bias(bool replace = false, double modify_amount = 0.0625){
        std::uniform_int_distribution<int> dist(0, nodes.size() - 1);
        int random_key = dist(*prng);

        if (replace){
            nodes[random_key].bias = random_double();
        } else {
            nodes[random_key].bias += random_double()*modify_amount;
            nodes[random_key].bias = std::max(std::min(nodes[random_key].bias, 32.), -32.);
        }
    }

    void add_node(){
        /*input node*/
        std::vector<int> possible_targets;

        for (auto it = weights.begin(); it != weights.end(); ++it)
        {
            if (!it->second.empty()){
                possible_targets.push_back(it->first);
            }   
        }

        if(possible_targets.empty()){
            return;
        }


        //choose node to link randomly

        std::uniform_int_distribution<int> dist(0, possible_targets.size() - 1);
        int node_inp = possible_targets[dist(*prng)];


        int layer_of_node_inp = -1;
        for (Node& node : nodes)
        {
            if (node.id == node_inp){
                layer_of_node_inp = node.layer;
            }
        }

        if (layer_of_node_inp == -1){
            std::cout<<"not found"<<'\n';
            return;
        }
        

        /*output nodes*/
        possible_targets.clear();

        for (auto it = weights[node_inp].begin(); it != weights[node_inp].end(); ++it)
        {
            possible_targets.push_back(it->first);
        }

        if(possible_targets.empty()){
            return;
        }
        

        std::uniform_int_distribution<int> dist2(0, possible_targets.size() - 1);
        int node_out = possible_targets[dist2(*prng)];


        /*add new node*/
        int new_node_id = AddNode(layer_of_node_inp+1);

        double old_weight = weights.at(node_inp).at(node_out).weight;

        weights.at(node_inp).erase(node_out);

        AddConnection(new_node_id, node_out);
        AddConnection(node_inp, new_node_id, old_weight, true);
    }

    void add_connection(){
        std::vector<std::pair<int, int>> all_sources;
        std::vector<std::pair<int, int>> all_targets;

        for (Node& node : nodes)
        {
            if (node.layer != -1){
                all_sources.push_back({node.id, node.layer});
            }
            if (node.layer != 0){
                all_targets.push_back({node.id, node.layer});
            }
        }


        for (std::pair<int, int> source : all_sources){
            for (std::pair<int, int> target : all_targets){
                
                if (target.second != -1){
                    if (source.second >= target.second){
                        continue;
                    }
                }

                if (source.first == target.first){
                    continue;
                }

                // if weight already exist continue
                if (weights[source.first].find(target.first) !=  weights[source.first].end()){
                    continue;
                }

                AddConnection(source.first, target.first);

            }
        }
        
    }

    void disable_enable_weight(bool set_to_bool){
        std::vector<int> weights1;
        for (auto it = weights.begin(); it != weights.end(); ++it)
        {
            weights1.push_back(it->first);
        }

        std::shuffle(std::begin(weights1), std::end(weights1), *prng);
        for (int key1 : weights1){
            std::vector<int> weights2;
            for (auto it = weights[key1].begin(); it != weights[key1].end(); ++it)
            {
                weights2.push_back(it->first);
            }
            std::shuffle(std::begin(weights2), std::end(weights2), *prng);

            for (int key2 : weights2){
                if (weights.at(key1).at(key2).enabled == set_to_bool){
                    continue;
                }
                weights.at(key1).at(key2).enabled = set_to_bool;
                return;
            }

        }
    }

    void change_hidden_func(){
        hid_func = (hid_func+((int)(random_01()*2))+1) % 3;
    }


public:
    Agent(int inputnum,
          int outputnum, 
          int* id_counter,
          std::mt19937* rng_pointer,
          std::map<std::pair<int, int>, int>* innov_connect,
          std::map<std::pair<int, int>, int>* innov_node,
          int hidden_func = 0,
          int output_func = 0
        ){

        pid_counter = id_counter;
        innov_connection = innov_connect;
        innov_nodes = innov_node;
        prng = rng_pointer;
        hid_func = hidden_func;
        out_func = output_func;

        output_foward.resize(outputnum);



        // add node and bias
        for (int i = 0; i < inputnum; i++)
        {
            AddNode(0);
        };
        for (int i = 0; i < outputnum; i++)
        {
            AddNode(-1);
        };

        //add weight
        for (int i = 0; i < inputnum; i++)
        {
            for (int j = 0; j < outputnum; j++)
            {

                AddConnection(i, j+inputnum);
            }; 
        };

    };

    std::vector<Node> get_nodes(){
        return nodes;
    }
    std::map<int, std::map<int, Connection>> get_connections(){
        return weights;
    }

    void print(){
        switch (hid_func)
        {
        case 0:
            std::cout<<"Hidden function : 0 | sig"<<'\n';
            break;

        case 1:
            std::cout<<"Hidden function : 1 | tanh"<<'\n';
            break;
        
        case 2:
            std::cout<<"Hidden function : 2 | ReLU"<<'\n';
            break;
            
        default:
            std::cout<<"Hidden function : Error"<<'\n';
            break;
        }
    
        std::cout<<"Nodes : ";
        for (int i = 0; i < nodes.size(); i++)
        {
            std::cout << "["<<nodes[i].id <<", " << nodes[i].layer <<", " << nodes[i].bias << "] ";
        };

        std::cout << "\nWeights :\n";
        for (auto& weight_data : weights){
            std::cout << weight_data.first << " : ";
            for (auto& weight_data2 : weight_data.second){
                std::cout << weight_data2.first << ": ["<<weight_data2.second.weight << ", "<< weight_data2.second.enabled << ", "<< weight_data2.second.innov << "]  ";
            }
            std::cout<<"\n";
        }
        
    };


    std::vector<double>& forward(std::vector<double> input_arr){
        /*get max layer and reset values*/
        int maxlayer = 0;
        int counter_inp = 0;
        for (int i = 0; i < nodes.size(); i++)
        {

            if (nodes[i].layer == 0){
                nodes[i].value = input_arr[counter_inp];
                counter_inp++;
            }
            else{
                nodes[i].value = 0.;
            }
            


            if (nodes[i].layer>maxlayer){
                maxlayer = nodes[i].layer;
            };
        };
        maxlayer++;

        for (int layer_idx = 0; layer_idx < maxlayer ; layer_idx++){
            for (Node& node : nodes)
            {
                //verify it is the right layer
                if (node.layer != layer_idx){
                    continue;
                };

                
                if (layer_idx != 0){
                    //[] instead of .at() to not get error and have default value 0
                    node.value += node.bias;

                    //0 -> sig, 1-> tanh,  2 -> relu
                    switch (hid_func)
                    {
                    case 0:
                        node.value = sig(node.value);
                        break;

                    case 1:
                        node.value = tanh(node.value);
                        break;
                    
                    case 2:
                        node.value = relu(node.value);
                        break;
            
                    default:
                        throw std::invalid_argument( "hidden func must me 0, 1 or 2" );
                        break;
                    }
                    
                }

                auto pos_memory = weights.find(node.id);

                if (pos_memory != weights.end()) {
                    for (const auto& weight : weights.at(node.id))
                    {                   
                        int pos = -1;
                    
                        for (int index = 0; index < nodes.size(); index++)
                        {
                            if (nodes[index].id == weight.first)
                            {
                                pos = index;
                                break;
                            }
                        }
                    
                    
                        if (pos == -1)
                            throw std::runtime_error("Invalid connection");
                    
                        nodes[pos].value += weight.second.weight * node.value;
                    }
                }

            }
            
        }

        int output_index = 0;

        for (Node& node : nodes) {
            if (node.layer == -1) {
                //output func
                switch (out_func) {
                    case 0:
                        output_foward[output_index] = sig(node.value + node.bias);
                        break;
                
                    case 1:
                        output_foward[output_index] = std::tanh(node.value + node.bias);
                        break;
                
                    case 2:
                        output_foward[output_index] = relu(node.value + node.bias);
                        break;
                
                    default:
                        throw std::invalid_argument("output func must be 0, 1 or 2");
                }
            
                output_index++;
            }
        }

        return output_foward;
    };


    void reproduce(Agent parent1, Agent parent2, 
        double modify_weight_prob = 0.8,
        double change_weight_prob = 0.1,
        double modify_bias_prob = 0.8,
        double change_bias_prob = 0.1,
        double add_node_mutation = 0.01,
        double add_connection_mutation = 0.04,
        double disable_weight_mutation = 0.06,
        double enable_weight_mutation = 0.06,
        double change_hidden_function = 0.002
    ){
        reproduce_nodes(parent1, parent2);
        reproduce_weight(parent1, parent2);
        reproduce_hidden_func(parent1.hid_func, parent2.hid_func);


        for (int i = 0; i < std::min((int)nodes.size()/32, 8) + 1; i++)
        {
            if (random_01()<modify_weight_prob){modify_weight();}
            if (random_01()<change_weight_prob){modify_weight(true);}

            if (random_01()<modify_bias_prob){modify_bias();}
            if (random_01()<change_bias_prob){modify_bias(true);}

            if (random_01()<add_node_mutation){add_node();}
            if (random_01()<add_connection_mutation){add_connection();}

            if (random_01()<enable_weight_mutation){disable_enable_weight(true);}
            if (random_01()<disable_weight_mutation){disable_enable_weight(false);}

            if (random_01()<change_hidden_function){change_hidden_func();}
        }
        

    }


    double genetic_distance(Agent agent2, 
        double weight_cost = 0.05,
        double node_cost = 2.,
        double connection_cost = 1.,
        double func_cost = 7.
    ){
        /*node*/
        std::vector<int> ids_agent1;
        for (Node& node : nodes) {
            ids_agent1.push_back(node.id);
        }
        std::vector<int> ids_agent2;
        for (Node& node : agent2.nodes) {
            ids_agent2.push_back(node.id);
        }

        std::sort(ids_agent1.begin(), ids_agent1.end());
        std::sort(ids_agent2.begin(), ids_agent2.end());

        std::vector<int> symetric_diff_ids;

        std::set_symmetric_difference(
            ids_agent1.begin(), ids_agent1.end(),
            ids_agent2.begin(), ids_agent2.end(),
            std::back_inserter(symetric_diff_ids)
        );


        /*weight*/
        std::vector<int> innovs_agent1;
        std::map<int, double> weights_agent1;
        for (auto& first_conn : weights){
            for (auto& second_conn : first_conn.second){
                innovs_agent1.push_back(second_conn.second.innov);
                weights_agent1[second_conn.second.innov] = second_conn.second.weight * second_conn.second.enabled;
            }
        }

        std::vector<int> innovs_agent2;
        std::map<int, double> weights_agent2;
        for (auto& first_conn : agent2.weights){
            for (auto& second_conn : first_conn.second){
                innovs_agent2.push_back(second_conn.second.innov);
                weights_agent2[second_conn.second.innov] = second_conn.second.weight * second_conn.second.enabled;
            }
        }

        std::vector<int> symetric_diff_innovs;
        std::set_symmetric_difference(
            innovs_agent1.begin(), innovs_agent1.end(),
            innovs_agent2.begin(), innovs_agent2.end(),
            std::back_inserter(symetric_diff_innovs)
        );

        double diff_weight = 0.;
        for (int innov : innovs_agent1){
            if (std::find(symetric_diff_innovs.begin(), symetric_diff_innovs.end(), innov) == symetric_diff_innovs.end()){
                double value = weights_agent1.at(innov) - weights_agent2.at(innov);
                diff_weight += value * value;
            }
        }

        /*std::cout<<symetric_diff_ids.size()<<' '<<symetric_diff_innovs.size()<<'\n';*/

        return (symetric_diff_ids.size() * node_cost
                + (hid_func != agent2.hid_func) * func_cost
                + symetric_diff_innovs.size() * connection_cost
                + diff_weight * weight_cost);
    }

};

class Neat{
private:
    int id_counter = -1;

    double modify_weight_prob;
    double change_weight_prob;
    double modify_bias_prob;
    double change_bias_prob;
    double add_node_mutation;
    double add_connection_mutation;
    double disable_weight_mutation;
    double enable_weight_mutation;
    double change_hidden_function;


    int max_species;

    std::mt19937* random_num_gen;
    
    std::vector<int> range_vec;

    int sizepop;

    std::map<std::pair<int, int>, int> innov_conn;
    std::map<std::pair<int, int>, int> innov_nodes_all;

    std::map<int, std::pair<int, double>> species_best_scores;

public:
    std::vector<Agent*> population;

    Neat(
        int sizepop,
        int input_num,
        int output_num,

        int hidden_func,
        int output_func,

        int max_species = -1,

        double modify_weight_prob = 0.8,
        double change_weight_prob = 0.1,
        double modify_bias_prob = 0.8,
        double change_bias_prob = 0.1,
        double add_node_mutation = 0.01,
        double add_connection_mutation = 0.04,
        double disable_weight_mutation = 0.06,
        double enable_weight_mutation = 0.06,
        double change_hidden_function = 0.002
    ){

        this->modify_weight_prob = modify_weight_prob;
        this->change_weight_prob = change_weight_prob;
        this->modify_bias_prob   = modify_bias_prob;
        this->change_bias_prob   = change_bias_prob;
        this->add_node_mutation  = add_node_mutation;
        this->add_connection_mutation = add_connection_mutation;
        this->disable_weight_mutation = disable_weight_mutation;
        this->enable_weight_mutation = enable_weight_mutation;
        this->change_hidden_function = change_hidden_function;

        this->sizepop = sizepop;
        this->max_species = max_species;
        
        random_num_gen = new std::mt19937{std::random_device{}()};

        for (int i = 0; i < sizepop; i++)
        {
            population.push_back(new Agent{input_num, output_num, &id_counter, random_num_gen, &innov_conn, &innov_nodes_all, hidden_func, output_func});
        }

        range_vec.resize(sizepop);
        
    }




    int get_sizepop(){
        return sizepop;
    }
    std::vector<Agent*> get_population(){
        return population;
    }


    void selection_species(std::vector<double>& fitnesses, int stagnation_treashold = 20) {
        if (fitnesses.size() != population.size()) {
            throw std::invalid_argument("Invalid fitness vector");
        }

        std::vector<std::vector<int>> all_species = allocate_species();

        
        std::vector<std::vector<int>> all_species_keep;
        all_species_keep.resize(all_species.size());

        std::vector<int> agent_to_delete;

        std::vector<double> average_fitness_species;
        std::vector<double> max_fitness_species;
        average_fitness_species.resize(all_species.size());
        max_fitness_species.resize(all_species.size());


        int index_specie = 0;
        /*create copy vector to manipulate it */
        for (std::vector<int> current_specie : all_species){
            std::sort(
                current_specie.begin(),
                current_specie.end(),
                [&](int x, int y) {
                    return fitnesses[x] > fitnesses[y];
                }
            );


            for (auto it = current_specie.begin(); it != current_specie.begin() + current_specie.size()/2 + 1; ++it) {
                all_species_keep[index_specie].push_back(*it);
            }

            for (auto it = current_specie.begin() + current_specie.size()/2 + 1; it != current_specie.end(); ++it) {
                agent_to_delete.push_back(*it);
            }

            for (int index : current_specie){
                average_fitness_species[index_specie] += fitnesses[index];
                max_fitness_species[index_specie] = std::max(max_fitness_species[index_specie], fitnesses[index]);
            }
            average_fitness_species[index_specie] /=  (double)all_species[index_specie].size();

            index_specie++;
        }

        double max_max_fitness = 0.;
        int max_fitness_idx = 0;
        for (int i = 0; i < max_fitness_species.size(); i++)
        {
            if (max_fitness_species[i] > max_max_fitness){
                max_max_fitness = max_fitness_species[i];
                max_fitness_idx = i;
            }
        }

        /*update species_best_scores to see if stagnation*/
        std::vector<int> species_to_remove;

        for (int i = 0; i < static_cast<int>(max_fitness_species.size()); ++i)
        {
            auto it = species_best_scores.find(i);
        
            if (it == species_best_scores.end())
            {
                species_best_scores[i] = {0, max_fitness_species[i]};
            }
            else if (max_fitness_species[i] > it->second.second)
            {
                species_best_scores[i] = {0, max_fitness_species[i]};
            }
            else
            {
                it->second.first++;
            
                if (it->second.first > stagnation_treashold &&
                    i != max_fitness_idx)
                {
                    species_to_remove.push_back(i);
                }
            }
        }

        // Never remove the species while iterating over the vectors.
        for (auto it = species_to_remove.rbegin();
             it != species_to_remove.rend(); ++it)
        {
            int i = *it;
        
            // Every member of this species must be replaced.
            for (int agent_idx : all_species_keep[i])
            {
                agent_to_delete.push_back(agent_idx);
            }
        
            // Remove all corresponding statistics and members.
            average_fitness_species.erase(average_fitness_species.begin() + i);
            max_fitness_species.erase(max_fitness_species.begin() + i);
            all_species_keep.erase(all_species_keep.begin() + i);
        
            // Remove the deleted species' history.
            species_best_scores.erase(i);
        }

        // Shift remaining history indices to match the vectors.
        for (auto it = species_to_remove.rbegin();
             it != species_to_remove.rend(); ++it)
        {
            int removed_idx = *it;
            std::map<int, std::pair<int, double>> updated_scores;
            for (const auto& entry : species_best_scores)
            {
                int idx = entry.first;
                if (idx > removed_idx)
                    --idx;
            
                updated_scores[idx] = entry.second;
            }
            species_best_scores = std::move(updated_scores);
        }
        
        

        
        double total_fitness = 0.;
        for (double fit : average_fitness_species){
            total_fitness += fit;
        }

        //if total fitness = 0 to avoid division by 0, make an equal repartition 
        if (total_fitness == 0.){
            for (int i = 0; i < average_fitness_species.size(); i++)
            {
                average_fitness_species[i] = 1;
            }
            total_fitness = average_fitness_species.size();
            
        }


        
        std::vector<int> num_allocation_specie;
        int total_alocated = 0;
        for (int i = 0; i < average_fitness_species.size(); i++)
        {
            if (i != average_fitness_species.size() - 1){
                num_allocation_specie.push_back(agent_to_delete.size()*(average_fitness_species[i]/total_fitness));
                total_alocated += num_allocation_specie[i];
            }
            else{
                num_allocation_specie.push_back(agent_to_delete.size() - total_alocated);
            }
        }
        
        

        total_alocated = 0;
        for (int i = 0; i < all_species_keep.size(); i++)
        {
            std::vector<int> delete_sub_vec(
                agent_to_delete.begin() + total_alocated, 
                agent_to_delete.begin() + total_alocated + num_allocation_specie[i]
            );



            selection_for_one_specie(all_species_keep[i], delete_sub_vec);

            
            total_alocated += num_allocation_specie[i];
        }

    }


    void selection_for_one_specie(std::vector<int> to_reproduce, std::vector<int> to_delete) {
        if (to_delete.empty())
        return;

        if (to_reproduce.empty())
        {
            throw std::runtime_error(
                "Cannot reproduce: species has no eligible parents."
            );
        }

        std::uniform_int_distribution<int> parent_dist(
            0, to_reproduce.size() - 1
        );

        //Replace bad ones
        for (int index_del : to_delete) {
            const int parent1_idx = to_reproduce[parent_dist(*random_num_gen)];
            const int parent2_idx = to_reproduce[parent_dist(*random_num_gen)];
        
            (*population[index_del]).reproduce(
                *population[parent1_idx],
                *population[parent2_idx],
                modify_weight_prob,
                change_weight_prob,
                modify_bias_prob,
                change_bias_prob,
                add_node_mutation,
                add_connection_mutation
            );
        }
    }


    std::vector<std::vector<double>> forward (std::vector<std::vector<double>> inputs){
        std::vector<std::vector<double>> outputs;
        for (int i = 0; i < sizepop; i++)
        {
            outputs.push_back((*population[i]).forward(inputs[i]));
        }
        
        return outputs;
    }

private:
    std::vector<std::vector<int>> allocate_species(double threashold = 15.){
        std::vector<std::vector<int>> species_members = {{0}};

        for (int i = 1; i < sizepop; i++)
        {
            bool is_alocated = false;
            double min_diff = 65536; //2**16
            int min_diff_idx = 0;

            int index = 0;
            for (std::vector<int>& specie_members : species_members){
                if(specie_members.empty()){
                    throw std::runtime_error("specie_members is empty");
                }

                double distance_gene = (*population[specie_members[0]]).genetic_distance(*population[i]);
                if(distance_gene < threashold){
                    specie_members.push_back(i);
                    is_alocated = true;
                    break;
                }

                if (distance_gene < min_diff){
                    min_diff = distance_gene;
                    min_diff_idx = index;
                }

                index++;
            }

            /*if not allocated create new species*/
            if(!is_alocated){
                if (max_species == -1 || species_members.size() < max_species){
                    species_members.push_back({i});
                }

                else {
                    species_members[min_diff_idx].push_back(i);
                }
                
            }
            
        }

        return species_members;
    }

};


double square(double x) {
    return x * x;
}

void test_XOR(Neat neat, int best_idx = 0){
    std::vector<double> test_inputs[4] = {{1.,1.},{0.,1.},{1.,0.},{0.,0.}};

    std::vector<double> outputs;

    for (int i = 0; i < 4; i++)
    {
        outputs = (*neat.population[best_idx]).forward(test_inputs[i]);

        for (double inp : test_inputs[i]){
            std::cout<<inp<<' ';
        }
        std::cout<<": ";
        
        

        for (double output : outputs){
            std::cout<<output<<' ';
        }
        std::cout<<'\n';
    }
    
    
}


double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


int main(){
    const int SIZEPOP = 128;
    const int OUTPUTSIZE = 2;

    /*very small penalty for larger network just to avoid network complexity shooting off to infinity for no reason
    which can hurt badly perfs*/
    const double num_node_penalty = 0.00005;

    Neat neat(
        /*size pop = */SIZEPOP,
        /*input_size = */ 2, 
        /*output_size = */ OUTPUTSIZE, 
        /*funcs 0 : sig, 1: tanh, 2 : relu*/ 2, 0,
    8);
    

    /*using a 2D vector is overkill and inneficient here 
    but this is made while having in mind bigger use cases than XOR
    with different inputs for each agent*/
    std::vector<std::vector<double>> XOR_inputs[4];
    std::vector<std::vector<double>> XOR_outputs[4];

    for (size_t i = 0; i < SIZEPOP; i++)
    {
        XOR_inputs[0].push_back({1.,1.});
        XOR_inputs[1].push_back({0.,1.});
        XOR_inputs[2].push_back({1.,0.});
        XOR_inputs[3].push_back({0.,0.});
    }


    
    
    std::vector<double> XOR_target[4] = {{1.,0.}, {0.,1.}, {0.,1.}, {1.,0.}};


    std::vector<double> fitness;
    fitness.resize(SIZEPOP);

    test_XOR(neat);
    (*neat.population[0]).print();
    std::cout<<'\n';

    for (int generation = 0; generation < 1000; generation++)
    {
        std::fill(fitness.begin(), fitness.end(), 0.);

        /*4 for the 4 different input*/
        for (int i = 0; i < 4; i++)
        {
            XOR_outputs[i] = neat.forward(XOR_inputs[i]);
            

            for (int agent_i = 0; agent_i < SIZEPOP; agent_i++)
            {
                
                double sum = 0;

                double error = 0.;
                for (int index_out = 0; index_out < OUTPUTSIZE; index_out++)
                {
                    error += square(XOR_outputs[i][agent_i][index_out] - XOR_target[i][index_out]);
                }


                fitness[agent_i] -= error;

                fitness[agent_i] -= (*neat.get_population()[agent_i]).get_nodes().size() * num_node_penalty;
                

            }

        }

        /*log data every 50 gen*/
        if (generation % 50 == 0) {
            const double best = *std::max_element(
                fitness.begin(), fitness.end()
            );
        
            double average = 0.0;
        
            for (double fit : fitness) {
                average += fit;
            }

            average /= fitness.size();

            std::cout << "Generation " << generation
                      << " | best: " << best
                      << " | average: " << average
                      << '\n';
        }

        neat.selection_species(fitness, 20);
    }

    int best_idx = static_cast<int>(
        std::max_element(fitness.begin(), fitness.end())
        - fitness.begin()
    );

    std::cout<<'\n';
    test_XOR(neat, best_idx);
    std::cout<<'\n';
    (*neat.population[best_idx]).print();

    return 0;
}