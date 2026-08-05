import argparse

import pennylane as qp
from pennylane import numpy as np
from types import SimpleNamespace

#fixed seed for reproducibility
np.random.seed(0)

#burgers equation: u_t + u * u_x = NU * u_xx
NU = 0.01 /np.pi #viscosity coefficient
X_Min, X_Max = -1.0, 1.0 #spatial domain
T_Min, T_Max = 0.0, 1.0 #time domain

N_U = 100 #total labelled (initial and boundary) points
N_F = 2000 #total collocation points

def make_training_data():
    #initial conditions
    n_ic = N_U // 2
    x_ic = np.random.uniform(X_Min, X_Max, size=(n_ic, 1)) #pics 50 random x-values from [-1, 1]
    t_ic = np.zeros((n_ic,1)) #t=0 for initial conditions
    u_ic = -np.sin(np.pi * x_ic) #initial condition u(x,0) = -sin(pi*x)

    #boundary conditions
    n_bc = N_U - n_ic
    t_bc = np.random.uniform(T_Min, T_Max, size=(n_bc, 1)) #random t-values from [0, 1]
    x_bc = np.where(np.random.rand(n_bc, 1) <0.5, X_Min, X_Max) #randomly choose x=-1 or x=1
    u_bc = np.zeros((n_bc, 1)) #boundary condition u(-1,t) = u(1,t) = 0

    #combine into one labelled dataset (N_U data)
    t_data = np.vstack([t_ic, t_bc])
    x_data = np.vstack([x_ic, x_bc])
    u_data = np.vstack([u_ic, u_bc])

    #collocation points: random (t,x)
    t_f = np.random.uniform(T_Min, T_Max, size=(N_F, 1))
    x_f = np.random.uniform(X_Min, X_Max, size=(N_F, 1))

    return t_data, x_data, u_data, t_f, x_f

#builds a fresh (quantum_circuit, init_params, network, pde_residual, loss_fn) set for a given architecture, so n_qubits/n_reuploads become a config instead of a file edit - sweep.py calls this directly with whatever combo it's currently on, the module-level names below are just build_model()'s default output kept around so nothing that already imports from main.py breaks
def build_model(n_qubits=4, n_reuploads=3):
    dev = qp.device("default.qubit", wires=n_qubits) #create a quantum device with n_qubits qubits

    @qp.qnode(dev)
    def quantum_circuit(inputs, weights):
        for layer in range(n_reuploads):
            #Encoding block S(x): fixes frequency spectrum (omega)
            for q in range(n_qubits):
                qp.RY(inputs[q], wires=q)

            #training block W(theta): sets the Fourier coefficients (amplitude)
            for q in range(n_qubits):
                qp.RY(weights[layer, q, 0], wires=q)
                qp.RZ(weights[layer, q, 1], wires=q)

            #entangling rings: CNOT gates between adjacent qubits
            for q in range(n_qubits):
                qp.CNOT(wires=[q, (q + 1) % n_qubits])

        #reading out every qubit instead of just qubit 0 - this matches how BQP-style architectures use the circuit, and it stops the quantum layer from getting squeezed down to a single scalar before it even reaches the post-layer
        return [qp.expval(qp.PauliZ(i)) for i in range(n_qubits)]

    def init_params():
        W1 = np.random.randn(n_qubits, 2) * 0.1 #prelayer weights
        b1 = np.zeros(n_qubits) #prelayer bias
        W_q = np.random.randn(n_reuploads, n_qubits, 2) * 0.1 #quantum layer weights
        W2 = np.random.randn(1, n_qubits) * 0.1 #post-layer weight, one per measured qubit
        b2 = np.zeros(1) #post-layer bias

#tells pennylane these numbers are trainable parameters
        for p in (W1, b1, W_q, W2, b2):
            p.requires_grad = True

        return W1, b1, W_q, W2, b2

    PRE_LAYER_SCALE = np.pi #how far the tanh output gets stretched before it hits the encoding gates

    def pre_layer(t, x, params): #pulled out of network() so anything that needs the exact encoding transform (e.g. frequency_unit_conversion.py) can reuse it instead of re-deriving it
        W1, b1 = params[0], params[1]
        inp = np.array([t, x]) #combine t and x into a single input array
        return np.tanh(W1 @ inp + b1) * PRE_LAYER_SCALE #pre-layer: linear transformation followed by tanh activation bounded to [-pi, pi]

    def network(t, x, params):
        W_q, W2, b2 = params[2], params[3], params[4]

        angles = pre_layer(t, x, params)

        q_out = quantum_circuit(angles, W_q) #quantum layer: outputs one number in [-1, 1] per qubit

        u = (W2 @ np.array(q_out) +b2)[0] #post-layer: linear transformation to produce final output u(t,x)
        return u #this is the predicted value of u at the given (t,x) point

    def pde_residual(t, x, params):
        # u as a function of t alone (x held constant) for computing u_t
        u_of_t = lambda t_: network(t_, x, params)
        u_t = qp.grad(u_of_t, argnums=0)(t)#compute u_t using automatic differentiation

        # u as a function of x alone (t held constant) for computing u_x and u_xx
        u_of_x = lambda x_: network(t, x_, params)
        u_x_fn = qp.grad(u_of_x, argnums=0) #function to compute u_x
        u_x = u_x_fn(x) #compute u_x using automatic differentiation
        u_xx = qp.grad(u_x_fn, argnums=0)(x) #compute u_xx using automatic differentiation

        u = network(t, x, params) #compute u(t,x) using the network
        f = u_t + u * u_x - NU * u_xx #compute the PDE residual f(t,x)
        return f #this is the residual of the PDE at the given (t,x) point

    def loss_fn(params, t_data, x_data, u_data, t_f, x_f):
        #MSE_u: How well predictions match the known data
        u_pred = np.array([network(t_data[i,0], x_data[i, 0], params)
                           for i in range(len(t_data))]) #compute predictions for all labelled data points
        mse_u = np.mean((u_pred - u_data[:, 0])**2) #compute mean squared error for labelled data

        #MSE_f: how far the PDE residual is from zero
        f_pred = np.array([pde_residual(t_f[i,0], x_f[i, 0], params) #compute PDE residuals for all collocation points
                           for i in range(len(t_f))])
        mse_f = np.mean(f_pred**2) #compute mean squared error for collocation points

        return mse_u + mse_f #return the total loss

    def checkpoint_config(): #metadata tag so a saved checkpoint can be told apart from other runs/architectures
        return {"model": "quantum_pinn", "n_qubits": n_qubits, "n_reuploads": n_reuploads, "measured_qubits": n_qubits}

    return SimpleNamespace(
        n_qubits=n_qubits, n_reuploads=n_reuploads,
        quantum_circuit=quantum_circuit, init_params=init_params,
        network=network, pde_residual=pde_residual, loss_fn=loss_fn,
        checkpoint_config=checkpoint_config,
        pre_layer=pre_layer, pre_layer_scale=PRE_LAYER_SCALE,
    )

#default 4-qubit/3-reupload model, kept at module level so `import main` still gives you main.network(), main.N_Qubits, etc like before build_model() existed
_default = build_model(n_qubits=4, n_reuploads=3)
N_Qubits = _default.n_qubits
N_Reuploads = _default.n_reuploads
Measured_Qubits = _default.n_qubits
quantum_circuit = _default.quantum_circuit
init_params = _default.init_params
network = _default.network
pde_residual = _default.pde_residual
loss_fn = _default.loss_fn
checkpoint_config = _default.checkpoint_config

#lets a config be trained straight from the command line (python main.py --n-qubits 5 --n-reuploads 5 --epochs 100 --seed 1) instead of only through sweep.py/extend_training.py - loop imported here rather than at module level since loop.py itself does `import main`, and main is running as __main__ here rather than under the name "main", so importing loop any earlier would hit `from main import make_training_data` before that name exists
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the quantum PINN directly from the command line")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-reuploads", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="path to save the trained checkpoint")
    args = parser.parse_args()

    import loop

    np.random.seed(args.seed)
    model = build_model(n_qubits=args.n_qubits, n_reuploads=args.n_reuploads)
    params, history = loop.train(epochs=args.epochs, model=model, checkpointPath=args.checkpoint)
    print(f"final loss = {float(history[-1]):.6f}")
