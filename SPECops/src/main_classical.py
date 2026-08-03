import pennylane as qp
from pennylane import numpy as np

from main import make_training_data, NU, N_Qubits

#classical control for the quantum PINN in main.py - same pre/post-layer shapes, same data, same loss, same optimizer (via loop.py), the only thing missing is the quantum_circuit() call itself: the pre-layer output goes straight into the post-layer instead of through the quantum device

def init_params():
    W1 = np.random.randn(N_Qubits, 2) * 0.1 #prelayer weights
    b1 = np.zeros(N_Qubits) #prelayer bias
    W2 = np.random.randn(1, N_Qubits) * 0.1 #post-layer weight, one per pre-layer output
    b2 = np.zeros(1) #post-layer bias

#tells pennylane these numbers are trainable parameters
    for p in (W1, b1, W2, b2):
        p.requires_grad = True

    return W1, b1, W2, b2

def network(t, x, params):
    W1, b1, W2, b2 = params

    inp = np.array([t, x]) #combine t and x into a single input array
    angles = np.tanh(W1 @ inp + b1) * np.pi #pre-layer: identical to the quantum version, just no quantum layer to feed it into

    u = (W2 @ angles + b2)[0] #post-layer reads the pre-layer output directly - no quantum_circuit() in between
    return u #this is the predicted value of u at the given (t,x) point

def pde_residual(t, x, params):
    # u as a function of t alone (x held constant) for computing u_t
    u_of_t = lambda t_: network(t_, x, params)
    u_t = qp.grad(u_of_t, argnums=0)(t) #compute u_t using automatic differentiation

    # u as a function of x alone (t held constant) for computing u_x and u_xx
    u_of_x = lambda x_: network(t, x_, params)
    u_x_fn = qp.grad(u_of_x, argnums=0) #function to compute u_x
    u_x = u_x_fn(x) #compute u_x using automatic differentiation
    u_xx = qp.grad(u_x_fn, argnums=0)(x) #compute u_xx using automatic differentiation

    u = network(t, x, params) #compute u(t,x) using the network
    f = u_t + u * u_x - NU * u_xx #compute the PDE residual f(t,x)
    return f #this is the residual of the PDE at the given (t,x) point

def loss_fn(params, t_data, x_data, u_data, t_f, x_f):
    #MSE_u: how well predictions match the known data
    u_pred = np.array([network(t_data[i,0], x_data[i, 0], params)
                       for i in range(len(t_data))])
    mse_u = np.mean((u_pred - u_data[:, 0])**2)

    #MSE_f: how far the PDE residual is from zero
    f_pred = np.array([pde_residual(t_f[i,0], x_f[i, 0], params)
                       for i in range(len(t_f))])
    mse_f = np.mean(f_pred**2)

    return mse_u + mse_f

def checkpoint_config():
    return {"model": "classical_control", "n_qubits": None, "n_reuploads": None, "measured_qubits": None}

if __name__ == "__main__":
    params = init_params()

    t_test = 0.5
    x_test = 0.3

    u_test = network(t_test, x_test, params)
    f_test = pde_residual(t_test, x_test, params)
    print(f"network({t_test}, {x_test}) = {u_test}")
    print(f"pde_residual({t_test}, {x_test}) = {f_test}")

    t_data, x_data, u_data, t_f, x_f = make_training_data()
    t_f_small = t_f[:20]
    x_f_small = x_f[:20]

    loss_value = loss_fn(params, t_data, x_data, u_data, t_f_small, x_f_small)
    print(f"loss = {loss_value}")
