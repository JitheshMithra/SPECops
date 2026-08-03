from types import SimpleNamespace

import pennylane as qp
from pennylane import numpy as np

from main import make_training_data, NU, N_Qubits

#classical control for the quantum PINN in main.py - same pre/post-layer shapes, same data, same loss, same optimizer (via loop.py), the only thing missing is the quantum_circuit() call itself: the pre-layer output goes straight into the post-layer instead of through the quantum device

#hidden_width defaults to N_Qubits so the pre-layer stays the same size as the quantum control's encoding layer by default (the original, param-mismatched control) - pass a wider hidden_width to build a capacity-matched control instead, since that's the only knob needed to move the classical side's param count without touching data/loss/optimizer/eval
def build_model(hidden_width=None):
    hiddenWidth = hidden_width or N_Qubits

    def init_params():
        W1 = np.random.randn(hiddenWidth, 2) * 0.1 #prelayer weights
        b1 = np.zeros(hiddenWidth) #prelayer bias
        W2 = np.random.randn(1, hiddenWidth) * 0.1 #post-layer weight, one per pre-layer output
        b2 = np.zeros(1) #post-layer bias

        for p in (W1, b1, W2, b2): #tells pennylane these numbers are trainable parameters
            p.requires_grad = True

        return W1, b1, W2, b2

    def pre_layer(t, x, params): #pulled out of network() to mirror main.py's pre_layer(), which activation_analysis.py relies on being present on both models
        W1, b1 = params[0], params[1]
        inp = np.array([t, x]) #combine t and x into a single input array
        return np.tanh(W1 @ inp + b1) * np.pi #pre-layer: identical to the quantum version, just no quantum layer to feed it into

    def network(t, x, params):
        W2, b2 = params[2], params[3]
        angles = pre_layer(t, x, params)
        u = (W2 @ angles + b2)[0] #post-layer reads the pre-layer output directly - no quantum_circuit() in between
        return u #this is the predicted value of u at the given (t,x) point

    def pde_residual(t, x, params):
        u_of_t = lambda t_: network(t_, x, params) # u as a function of t alone (x held constant) for computing u_t
        u_t = qp.grad(u_of_t, argnums=0)(t)

        u_of_x = lambda x_: network(t, x_, params) # u as a function of x alone (t held constant) for computing u_x and u_xx
        u_x_fn = qp.grad(u_of_x, argnums=0)
        u_x = u_x_fn(x)
        u_xx = qp.grad(u_x_fn, argnums=0)(x)

        u = network(t, x, params)
        f = u_t + u * u_x - NU * u_xx #the PDE residual f(t,x)
        return f

    def loss_fn(params, t_data, x_data, u_data, t_f, x_f):
        u_pred = np.array([network(t_data[i, 0], x_data[i, 0], params) for i in range(len(t_data))])
        mse_u = np.mean((u_pred - u_data[:, 0]) ** 2) #how well predictions match the known data

        f_pred = np.array([pde_residual(t_f[i, 0], x_f[i, 0], params) for i in range(len(t_f))])
        mse_f = np.mean(f_pred ** 2) #how far the PDE residual is from zero

        return mse_u + mse_f

    def checkpoint_config():
        return {"model": "classical_control", "n_qubits": None, "n_reuploads": None,
                "measured_qubits": None, "hidden_width": hiddenWidth}

    return SimpleNamespace(
        hidden_width=hiddenWidth, n_qubits=None, n_reuploads=None,
        init_params=init_params, network=network, pde_residual=pde_residual,
        loss_fn=loss_fn, checkpoint_config=checkpoint_config, pre_layer=pre_layer,
    )

#default hidden_width=N_Qubits control, kept at module level so `import main_classical` still gives you main_classical.network(), main_classical.init_params(), etc like before build_model() existed
_default = build_model()
init_params = _default.init_params
network = _default.network
pde_residual = _default.pde_residual
loss_fn = _default.loss_fn
checkpoint_config = _default.checkpoint_config
pre_layer = _default.pre_layer

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
