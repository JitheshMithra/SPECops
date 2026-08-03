import pennylane as qp
from pennylane import numpy as np
import matplotlib.pyplot as plt

#gives a different seed each time the code is run (for reproducibility)
np.random.seed(0)

#burgers equation: u_t + u * u_x = NU * u_xx
NU = 0.01 /np.pi #viscosity coefficient
X_Min, X_Max = -0.1, 1.0 #spatial domain
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

N_Qubits = 4
N_Reuploads = 3 #number of re-uploading layers

dev = qp.device("default.qubit", wires=N_Qubits) #create a quantum device with 4 qubits

@qp.qnode(dev)
def quantum_circuit(inputs, weights):
    for layer in range(N_Reuploads):
        #Encoding block S(x): fixes frequency spectrum (omega)
        for q in range(N_Qubits):
            qp.RY(inputs[q], wires=q)

        #training block W(theta): sets the Fourier coefficients (amplitude)
        for q in range(N_Qubits):
            qp.RY(weights[layer, q, 0], wires=q)
            qp.RZ(weights[layer, q, 1], wires=q)

        #entangling rings: CNOT gates between adjacent qubits
        for q in range(N_Qubits):
            qp.CNOT(wires=[q, (q + 1) % N_Qubits])

    return qp.expval(qp.PauliZ(0)) #turns quantum state into a real number in [-1, 1] by measuring the first qubit

def init_params():
    W1 = np.random.randn(N_Qubits, 2) * 0.1 #prelayer weights
    b1 = np.zeros(N_Qubits) #prelayer bias
    W_q = np.random.randn(N_Reuploads, N_Qubits, 2) * 0.1 #quantum layer weights
    W2 = np.random.randn(1, 1) * 0.1 #post-layer weight
    b2 = np.zeros(1) #post-layer bias

#tells pennylane these numbers are trainable parameters
    for p in (W1, b1, W_q, W2, b2):
        p.requires_grad = True

    return W1, b1, W_q, W2, b2

def network(t, x, params):
    W1, b1, W_q, W2, b2 = params

    inp = np.array([t, x]) #combine t and x into a single input array
    angles = np.tanh(W1 @ inp + b1) * np.pi #pre-layer: linear transformation followed by tanh activation bounded to [-pi, pi]

    q_out = quantum_circuit(angles, W_q) #quantum layer: outputs a single number in [-1, 1]

    u = (W2 @ np.array([q_out]) +b2)[0] #post-layer: linear transformation to produce final output u(t,x)
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

if __name__ == "__main__":
    params = init_params()

    t_test = 0.5
    x_test = 0.3


    u_test = network(t_test, x_test, params)
    f_test = pde_residual(t_test, x_test, params)
    print(f"network({t_test}, {x_test}) = {u_test}")
    print(f"pde_residual({t_test}, {x_test}) = {f_test}")

    t_data, x_data, u_data, t_f, x_f = make_training_data()
    # NOTE: for a quick first test, only using a handful of collocation points
    t_f_small = t_f[:20]
    x_f_small = x_f[:20]

    loss_value = loss_fn(params, t_data, x_data, u_data, t_f_small, x_f_small)
    print(f"loss = {loss_value}")