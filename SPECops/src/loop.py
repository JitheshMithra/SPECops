import pennylane as qp
from pennylane import numpy as np

import main
from main import make_training_data
from checkpoint import saveCheckpoint


def train(epochs = 100, lr=0.05, n_f_batch=20, checkpointPath=None, model=None): #this function trains a PINN (quantum by default, but any module with init_params/loss_fn/checkpoint_config works) to solve the Burgers' equation
    model = model or main #defaults to the quantum model in main.py, main_classical.py plugs in here for the control run

    params = model.init_params() #these are the parameters of the model that will be optimized during training
    opt = qp.GradientDescentOptimizer(stepsize=lr) #this is the optimizer that will be used to update the parameters of the model

    t_data, x_data, u_data, t_f, x_f = make_training_data() #this function generates the training data for the PINN - same data regardless of which model is training

    history = [] #this will store the loss values during training
    for epoch in range(1, epochs + 1):
        #randomly sample a batch of collocation points (this takes a small amount to account for speed)
        idx = np.random.choice(len(t_f), size=n_f_batch, replace=False) #this randomly selects n_f_batch indices from the collocation points without replacement
        #this selects the collocation points corresponding to the randomly selected indices
        t_f_batch = t_f[idx]
        x_f_batch = x_f[idx]

        #params has to be passed to opt.step as separate positional args, not one bundled tuple - pennylane's autograd only tracks requires_grad on individual arrays, so a single tuple argument differentiates to an empty gradient and opt.step is a silent no-op (this was previously the case here: params never updated across any epoch)
        cost_fn = lambda *p: model.loss_fn(p, t_data, x_data, u_data, t_f_batch, x_f_batch) #this defines the cost function that will be minimized during training

        params = opt.step(cost_fn, *params) #this updates the parameters of the quantum circuit using the optimizer
        current_loss = cost_fn(*params) #this computes the current loss value after the parameter update
        history.append(current_loss) #this appends the current loss value to the history list

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d} loss = {current_loss:.6f}") #this prints the current epoch and loss value every 10 epochs and at the first epoch

    if checkpointPath: #only bother saving if the caller actually wants a checkpoint
        saveCheckpoint(checkpointPath, params, model.checkpoint_config())
        print(f"saved checkpoint to {checkpointPath}")

    return params, history #this returns the optimized parameters and the history of loss values during training


if __name__ == "__main__":
    train()
