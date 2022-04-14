import torch
from torch import nn
import numpy as np


def derivative(y, x):
    dim = y.shape[-1]
    dydx = torch.zeros_like(y, dtype=y.dtype)
    for i in range(dim):
        yi = y[:, i].sum()
        dydx[:, i] = torch.autograd.grad(yi, x, create_graph=True, allow_unused=True)[
            0
        ].squeeze()
    return dydx


class PiNN:
    class NN(nn.Module):
        def __init__(self, hidden_layers):
            super().__init__()

            prev_nodes_per_layer = 2
            hidden = []
            for nodes_per_layer in hidden_layers:

                hidden += [nn.Linear(prev_nodes_per_layer, nodes_per_layer), nn.Tanh()]
                prev_nodes_per_layer = nodes_per_layer

            self.ff_graph = nn.Sequential(*hidden)

            # Output Layer is assumed to be linear
            self.output = nn.Linear(prev_nodes_per_layer, 1)

        def forward(self, x, t):
            return self.output(self.ff_graph(torch.cat((x, t), 1)))

    def __init__(self, pde, bc, ic, hyperparameters=None, verbose=False):
        super().__init__()

        self.pde = pde
        self.bc = bc
        self.ic = ic

        # Default Hyperparameters
        if hyperparameters is None:
            self.hidden_layers = [20, 20, 20]
            self.epochs = 100

        # User-specified Hyperparameters
        else:
            for parameter, val in hyperparameters.items():
                if hasattr(self, parameter):
                    setattr(self, parameter, val)

        self.verbose = verbose
        if self.verbose:
            self.mse_train = []
            self.report_after_e_epochs = min(self.epochs, 10)

        self.net = self.NN(self.hidden_layers)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        string = self.net.__str__() + f'{"# of model parameters:":<40}{params:>30}\n'
        return string

    def fit(self, X_train, y_train):
        # Call to fit forces retraining
        self.net = self.NN(self.hidden_layers)

        # Unpack the training data into interior/ic/bc points and set up the
        # minibatch indices
        (
            (x_int, t_int),
            (x_init, t_init),
            (x_boundary, x_boundary2, t_boundary),
        ) = X_train
        batchsize_int = x_int.shape[0]
        batchsize_bc = x_boundary.shape[0]
        batchsize_ic = x_init.shape[0]
        number_of_minibatches = 1

        # Minibatch index set to iterate over
        idx_int = np.array_split(np.arange(x_int.shape[0]), number_of_minibatches)
        idx_init = np.array_split(np.arange(x_init.shape[0]), number_of_minibatches)
        idx_boundary = np.array_split(
            np.arange(x_boundary.shape[0]), number_of_minibatches
        )

        optimizer = torch.optim.LBFGS(self.net.parameters())
        loss_fn = torch.nn.MSELoss()

        if self.verbose:
            points_per_iteration = (
                batchsize_int * batchsize_bc * batchsize_ic * number_of_minibatches
            )
            print(f'{"Hyperparameters":-^70}')
            print(f'{"# of collocation points per batch:":<40}{batchsize_int:>30}')
            print(f'{"# of boundary points per batch:":<40}{batchsize_bc:>30}')
            print(f'{"# of initial condition points per batch:":<40}{batchsize_ic:>30}')
            print(f'{"# of minbatches:":<40}{number_of_minibatches:>30}')
            print(f'{"Points per Iteration:":<40}{points_per_iteration:>30}')
            print(f'{"Iterations:":<40}{self.epochs:>30}')
            print(
                f'{"Total number of collocation points":<40}'
                f"{self.epochs*points_per_iteration:>30}"
            )
            print(f'{"":-^70}')
            print(
                f'{"Epoch":^10}|'
                f'{"Total Loss":^15}|'
                f'{"Loss (PDE)":^15}|'
                f'{"Loss (BC)":^15}|'
                f'{"Loss (IC)":^15}'
            )

        for e in range(self.epochs + 1):
            for minibatch in range(number_of_minibatches):

                x = x_int[idx_int[minibatch], :]
                t = t_int[idx_int[minibatch], :]

                x_ic = x_init[idx_init[minibatch], :]
                t_ic = t_init[idx_init[minibatch], :]

                x_bc_0 = x_boundary[idx_boundary[minibatch], :]
                x_bc_L = x_boundary2[idx_boundary[minibatch], :]
                t_bc = t_boundary[idx_boundary[minibatch], :]

                x.requires_grad_(True)
                t.requires_grad_(True)

                last_total_loss_in_step = 0
                last_interior_loss_in_step = 0
                last_bc_loss_in_step = 0
                last_ic_loss_in_step = 0

                def compute_loss():
                    nonlocal last_total_loss_in_step
                    nonlocal last_interior_loss_in_step
                    nonlocal last_bc_loss_in_step
                    nonlocal last_ic_loss_in_step

                    optimizer.zero_grad()

                    # Interior
                    u = self.net(x, t)
                    mse_interior = loss_fn(self.pde(u, x, t), torch.zeros_like(u))

                    # Boundary Condition
                    u_bc_0 = self.net(x_bc_0, t_bc)
                    u_bc_L = self.net(x_bc_L, t_bc)
                    mse_boundary = loss_fn(
                        u_bc_0, self.bc(u_bc_0, x_bc_0, t_bc)
                    ) + loss_fn(u_bc_L, self.bc(u_bc_L, x_bc_L, t_bc))

                    # Initial conditions
                    u_ic = self.net(x_ic, t_ic)
                    mse_ic = loss_fn(u_ic, self.ic(x_ic, t_ic))

                    # Total Loss
                    loss = mse_interior + mse_boundary + mse_ic

                    loss.backward()

                    last_total_loss_in_step = loss.item()
                    last_interior_loss_in_step = mse_interior.item()
                    last_bc_loss_in_step = mse_boundary.item()
                    last_ic_loss_in_step = mse_ic.item()

                    # pytorch requires that only the total loss is returned from the closure
                    return loss

                optimizer.step(compute_loss)
            if self.verbose:
                self.mse_train.append(
                    [
                        last_total_loss_in_step,
                        last_interior_loss_in_step,
                        last_bc_loss_in_step,
                        last_ic_loss_in_step,
                    ]
                )

                if e % self.report_after_e_epochs == 0:
                    print(
                        f"{e:^10}|"
                        f"{last_total_loss_in_step:>15.6e}|"
                        f"{last_interior_loss_in_step:>15.6e}|"
                        f"{last_bc_loss_in_step:>15.6e}|"
                        f"{last_ic_loss_in_step:>15.6e}"
                    )

    def predict(self, X_test):
        x, t = X_test
        with torch.autograd.no_grad():
            return self.net(x, t)
