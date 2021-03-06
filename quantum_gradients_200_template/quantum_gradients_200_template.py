#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)
    forward_values = np.zeros([5], dtype=np.float64)
    backward_values = np.zeros([5], dtype=np.float64)
    values = circuit(weights)
    # QHACK #
    def parameter_shift(params, i):
        shifted = params.copy()
        shifted[i] += np.pi/2
        forward = circuit(shifted)
        forward_values[i] = forward
        shifted[i] -= np.pi
        backward = circuit(shifted)
        backward_values[i] = backward
        return 0.5*(forward-backward)

    def shift(weights):
        for i in range(len(weights)):
            gradient[i] = parameter_shift(weights, i)
        return gradient

    def hessian_calculate():
        for i in range(len(weights)):
            for j in range(i+1, len(weights)):
                weights[i] += np.pi/2
                weights[j] += np.pi/2
                plusplus = circuit(weights)
                weights[j] -= np.pi
                plusminus = circuit(weights)
                weights[i] -= np.pi
                minusminus = circuit(weights)
                weights[j] += np.pi
                minusplus = circuit(weights)
                hessian[i][j] = 0.25*(plusplus+minusminus-plusminus-minusplus)
                hessian[j][i] = hessian[i][j]
                weights[i] += np.pi/2
                weights[j] -= np.pi/2

        for i in range(len(weights)):
            hessian[i][i] = (forward_values[i]+backward_values[i]-2*values)/2

    shift(weights)
    hessian_calculate()
    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
