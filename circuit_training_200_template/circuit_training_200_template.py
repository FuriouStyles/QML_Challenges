#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml


# DO NOT MODIFY any of these parameters
NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.

    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)

    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """

    max_ind_set = []

    # QHACK #
    WIRES = NODES
    device = qml.device('default.qubit', wires=WIRES)

    cost_H, mixer_H = qml.qaoa.max_independent_set(graph, constrained = True)

    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_H)
        qml.qaoa.mixer_layer(alpha, mixer_H)

    def circuit(params, **kwargs):
        qml.layer(qaoa_layer, N_LAYERS, params[0], params[1])
        # qml.Hadamard(wires=WIRES)

  

    @qml.qnode(device)
    def probability_circuit(gamma, alpha):
        circuit([gamma, alpha])
        return qml.probs(wires=range(WIRES))

    probs = probability_circuit(params[0], params[1])#.reshape(2 ** WIRES, )
    result = probs.tolist()
    data = result.index(max(result))
    binary = '{0:06b}'.format(data)
    binary_str = str(binary)

    count = 0
    while count < 6:
        if binary_str[count] == "1":
            max_ind_set.append(count)
        count += 1

    max_ind_set = list(dict.fromkeys(max_ind_set))
    

    # def find_max_set(proba):
    #     pass
    
    # print(probs[0])
    # for idx, proba in enumerate(probs):
    #     print(f'{format(idx, f"#05b")[2:]}\t{proba}')
    # # print(format((np.argmax), f'#05b'))
    # print(type(probs[0]))

    # QHACK #

    return max_ind_set


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    graph_string = sys.stdin.read()
    graph_data = json.loads(graph_string)

    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)
