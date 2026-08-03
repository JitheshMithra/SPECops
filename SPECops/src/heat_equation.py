from functools import lru_cache
from types import SimpleNamespace

import pennylane as qp
from pennylane import numpy as np
from scipy.integrate import quad

import main
from main import X_Min, X_Max, T_Min, T_Max, make_training_data

#cross-validation PDE for the QAPINN architecture: same domain, same IC/BC, same encoding/quantum-circuit/readout as main.py's Burgers setup (reused directly from main.build_model, since none of that is PDE-specific) - only the physics term and the reference solution change; the heat equation's reference solution is a Fourier sine series instead of the Cole-Hopf integral, and it's linear (no shock), so it's a much easier sanity check for whether a given (n_qubits, n_reuploads) config is expressive/trainable enough at all, before trusting it on the harder Burgers problem
ALPHA = 0.01 / np.pi #same diffusivity scale as main.py's NU, for a like-for-like comparison
N_MODES = 20 #-sin(pi*x) is exactly the n=2 Dirichlet eigenmode on this domain (see below), so in practice only one term is non-negligible - N_MODES=20 is headroom for whatever IC this ends up getting reused with later, not tuned to this one

DOMAIN_LENGTH = X_Max - X_Min

def initial_condition(x):
    #identical to main.py's u(x,0) = -sin(pi*x) / u(-1,t) = u(1,t) = 0, so a QAPINN trained here is directly comparable, architecture-for-architecture, to the Burgers results
    return -np.sin(np.pi * x)

def eigenfunction(n, x):
    #Dirichlet sine eigenbasis on [X_Min, X_Max]: zero at both boundaries by construction
    return np.sin(n * np.pi * (x - X_Min) / DOMAIN_LENGTH)

def eigenvalue(n):
    return (n * np.pi / DOMAIN_LENGTH) ** 2

@lru_cache(maxsize=None)
def fourierCoefficient(n):
    integrand = lambda x: initial_condition(x) * eigenfunction(n, x)
    value, _ = quad(integrand, X_Min, X_Max, limit=200)
    return (2.0 / DOMAIN_LENGTH) * value

#exact solution via separation of variables: u(x,t) = sum_n b_n * phi_n(x) * exp(-ALPHA * lambda_n * t) - for initial_condition() as defined above, phi_2(x) = sin(pi*(x+1)) = -sin(pi*x) = initial_condition(x) exactly, so b_2 = 1 and every other b_n is ~0 (up to quad's numerical error), which collapses to the single closed form u(x,t) = -sin(pi*x) * exp(-ALPHA * pi**2 * t), but going through the general series keeps this reusable if initial_condition() ever changes to something with more than one mode in it
@lru_cache(maxsize=None)
def heatEquationU(x, t, nModes=N_MODES):
    total = 0.0
    for n in range(1, nModes + 1):
        bn = fourierCoefficient(n)
        if abs(bn) < 1e-12: #skip modes the IC doesn't excite
            continue
        total += bn * eigenfunction(n, x) * np.exp(-ALPHA * eigenvalue(n) * t)
    return total

#builds a (network, pde_residual, loss_fn) set for the heat equation, reusing main.build_model()'s architecture (pre-layer, quantum circuit, post-layer) wholesale since none of that is specific to which PDE is being solved - only the residual (u_t - ALPHA*u_xx instead of u_t + u*u_x - NU*u_xx) and the checkpoint tag differ from the Burgers version
def build_model(n_qubits=4, n_reuploads=3):
    burgersModel = main.build_model(n_qubits=n_qubits, n_reuploads=n_reuploads)
    network = burgersModel.network

    def pde_residual(t, x, params):
        u_of_t = lambda t_: network(t_, x, params)
        u_t = qp.grad(u_of_t, argnums=0)(t)

        u_of_x = lambda x_: network(t, x_, params)
        u_x_fn = qp.grad(u_of_x, argnums=0)
        u_xx = qp.grad(u_x_fn, argnums=0)(x)

        return u_t - ALPHA * u_xx

    def loss_fn(params, t_data, x_data, u_data, t_f, x_f):
        u_pred = np.array([network(t_data[i, 0], x_data[i, 0], params)
                           for i in range(len(t_data))])
        mse_u = np.mean((u_pred - u_data[:, 0]) ** 2)

        f_pred = np.array([pde_residual(t_f[i, 0], x_f[i, 0], params)
                           for i in range(len(t_f))])
        mse_f = np.mean(f_pred ** 2)

        return mse_u + mse_f

    def checkpoint_config():
        return {"model": "heat_equation_qapinn", "n_qubits": n_qubits, "n_reuploads": n_reuploads,
                "measured_qubits": n_qubits}

    return SimpleNamespace(
        n_qubits=n_qubits, n_reuploads=n_reuploads,
        quantum_circuit=burgersModel.quantum_circuit, init_params=burgersModel.init_params,
        network=network, pde_residual=pde_residual, loss_fn=loss_fn,
        checkpoint_config=checkpoint_config,
        pre_layer=burgersModel.pre_layer, pre_layer_scale=burgersModel.pre_layer_scale,
    )

if __name__ == "__main__":
    #sanity check only - not a training run, confirms the Fourier series collapses to -sin(pi*x) at t=0 like the closed form predicts, and decays toward 0 as t grows (no shock, so this should look nothing like Burgers)
    for x in (-0.9, -0.5, 0.0, 0.5, 0.9):
        print(f"x={x:+.2f}  u(x, t=0)={float(heatEquationU(x, 0.0)):+.4f}  "
              f"vs initial_condition={float(initial_condition(x)):+.4f}")
    for t in (0.0, 0.1, 0.5, 1.0):
        print(f"u(x=0.5, t={t}) = {float(heatEquationU(0.5, t)):.6f}")
