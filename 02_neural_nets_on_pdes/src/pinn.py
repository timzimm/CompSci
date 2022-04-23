import torch
from torch import nn
import numpy as np


def derivative(y, x):
    dim = y.shape[-1]
    dydx = torch.zeros_like(y, dtype=y.dtype)
    for i in range(dim):
        yi = y[:, i].sum()
        dydx[:, i] = torch.autograd.grad(
            yi ** (1.0), x, create_graph=True, allow_unused=True
        )[0].squeeze()
    return dydx


class PiNN:
    class NN(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_layers):
            super().__init__()

            prev_nodes_per_layer = input_dim
            hidden = []
            for nodes_per_layer in hidden_layers:

                hidden += [nn.Linear(prev_nodes_per_layer, nodes_per_layer), nn.Tanh()]
                prev_nodes_per_layer = nodes_per_layer

            self.ff_graph = nn.Sequential(*hidden)

            # Output Layer is assumed to be linear
            self.output = nn.Linear(prev_nodes_per_layer, output_dim)

        def forward(self, *domain_points):
            return self.output(self.ff_graph(torch.cat(domain_points, 1)))

    def __init__(self, 
                 pde, 
                 ic, 
                 bc, 
                 solution_structure=lambda net_output, *domain_point: net_output,
                 hooks=None,
                 events_for_loss=None,
                 hyperparameters={}, 
                 verbose=False):

        self.pde = pde
        self.ic = ic
        self.bc = bc
        self.solution_structure = solution_structure
        
        # Functions used to monitor user-defined observables during the training
        self.hooks = hooks
        if self.hooks is not None:
            self.hooks_returns = {}
            for hook in self.hooks:
                self.hooks_returns[hook.__name__] = []
                
        # Events used to control the training
        self.events_for_loss = events_for_loss

        self.input_dim = pde.domain_dim
        self.output_dim = pde.target_dim

        # Hyperparameters to be defined
        self.hidden_layers = None
        self.epochs = None
        self.number_of_minibatches = None
        self.optimizer = None
        self.optimizer_params = {}

        # User-specified Hyperparameters
        for parameter, val in hyperparameters.items():
            if hasattr(self, parameter):
                setattr(self, parameter, val)

        # Populate unset parameters with sane default values
        self.hyperparameters_default = {
            "hidden_layers": [20, 20, 20],
            "epochs": 200,
            "number_of_minibatches": 1,
            "optimizer": "lbfgs",
            "optimizer_params": None
        }
        for parameter, val in self.hyperparameters_default.items():
            if getattr(self, parameter, val) is None:
                setattr(self, parameter, val)

        self.verbose = verbose
        if self.verbose:
            self.mse_train = []
            self.report_after_e_epochs = self.epochs // 10

        self.net = self.NN(self.input_dim, self.output_dim, self.hidden_layers)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        string = f'{"Model":-^70}\n'
        string = self.net.__str__() + "\n"
        string += f'{"model parameters:":<40}{params:>30}\n'
        string += f'{"Hyperparameters":-^70}\n'
        for parameter, val in self.hyperparameters_default.items():
            string += f"{parameter:<40}{str(getattr(self, parameter, val)):>30}\n"
        return string

    def fit(self, X_train):

        # Call to fit forces retraining
        self.net = self.NN(self.input_dim, self.output_dim, self.hidden_layers)

        # Unpack the training data into interior/ic/bc points and set up the
        # minibatch indices
        domain_int, domain_ic, domain_bc = X_train
        N_int = domain_int.shape[0]
        N_bc = domain_bc.shape[0]
        N_ic = domain_ic.shape[0]

        # Minibatch index set to iterate over
        idx_int = np.array_split(np.arange(N_int), self.number_of_minibatches)
        idx_init = np.array_split(np.arange(N_ic), self.number_of_minibatches)
        idx_boundary = np.array_split(np.arange(N_bc), self.number_of_minibatches)

        loss_fn = torch.nn.MSELoss()

        if self.optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS(self.net.parameters(), **self.optimizer_params)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.net.parameters(), **self.optimizer_params)

        if self.verbose:
            points_per_iteration = (N_int + N_bc + N_ic) / self.number_of_minibatches
            print(f'{"Training Log":-^70}')
            print(f'{"# of collocation points:":<40}{N_int:>30}')
            print(f'{"# of boundary points:":<40}{N_bc:>30}')
            print(f'{"# of initial condition:":<40}{N_ic:>30}')
            print(f'{"Points per Iteration:":<40}{points_per_iteration:>30}')
            print(f'{"":-^70}')
            print(
                f'{"Epoch":^10}|'
                f'{"Total Loss":^15}|'
                f'{"Loss (PDE)":^15}|'
                f'{"Loss (BC)":^15}|'
                f'{"Loss (IC)":^15}'
            )

        for e in range(self.epochs + 1):
            for minibatch in range(self.number_of_minibatches):

                domain_int_batch = domain_int[idx_int[minibatch]]
                coordinates_int_batch = [
                    c.reshape(-1, 1) for c in torch.unbind(domain_int_batch, dim=-1)
                ]

                domain_ic_batch = domain_ic[idx_init[minibatch]]
                coordinates_ic_batch = [
                    c.reshape(-1, 1) for c in torch.unbind(domain_ic_batch, dim=-1)
                ]

                domain_bc_batch = domain_bc[idx_boundary[minibatch]]
                coordinates_bc_batch = [
                    c.reshape(-1, 1) for c in torch.unbind(domain_bc_batch, dim=-1)
                ]

                for coordinate in coordinates_int_batch:
                    coordinate.requires_grad_(True)

                total_loss_in_step = 0
                interior_loss_in_step = 0
                bc_loss_in_step = 0
                ic_loss_in_step = 0

                def compute_loss():
                    nonlocal total_loss_in_step
                    nonlocal interior_loss_in_step
                    nonlocal bc_loss_in_step
                    nonlocal ic_loss_in_step

                    optimizer.zero_grad()

                    # Interior
                    net_int = self.net(*coordinates_int_batch)
                    u_int = self.solution_structure(net_int,*coordinates_int_batch)
                    mse_interior = loss_fn(*self.pde(u_int,*coordinates_int_batch))
                    
                    # Initial conditions
                    net_ic = self.net(*coordinates_ic_batch)
                    u_ic = self.solution_structure(net_ic, *coordinates_ic_batch)
                    mse_ic = loss_fn(*self.ic(u_ic, *coordinates_ic_batch))

                    # Boundary Condition
                    net_bc = self.net(*coordinates_bc_batch)
                    u_bc = self.solution_structure(net_bc, *coordinates_bc_batch)
                    mse_boundary = loss_fn(*self.bc(u_bc, *coordinates_bc_batch))

                    # Total Loss
                    loss = mse_interior + mse_boundary + mse_ic

                    loss.backward()

                    total_loss_in_step = loss.item()
                    interior_loss_in_step = mse_interior.item()
                    bc_loss_in_step = mse_boundary.item()
                    ic_loss_in_step = mse_ic.item()

                    # pytorch requires that only the total loss is returned from the closure
                    return loss

                optimizer.step(compute_loss)
            if self.verbose:
                self.mse_train.append(
                    [
                        total_loss_in_step,
                        interior_loss_in_step,
                        bc_loss_in_step,
                        ic_loss_in_step,
                    ]
                )
                if self.hooks is not None:
                    with torch.autograd.no_grad():
                        for hook in self.hooks:
                            self.hooks_returns[hook.__name__].append(hook(self.net))

                if e % self.report_after_e_epochs == 0:
                    print(
                        f"{e:^10}|"
                        f"{total_loss_in_step:^15.5e}|"
                        f"{interior_loss_in_step:^15.5e}|"
                        f"{bc_loss_in_step:^15.5e}|"
                        f"{ic_loss_in_step:^15.5e}"
                    )
            if self.events_for_loss is not None:
                terminate_early = False
                for event in self.events_for_loss:
                    if event(total_loss_in_step) and event.terminate:
                        if self.verbose:
                            print(f"{event.__name__} fired terminal event after {e} epochs")
                        terminate_early = True
            if terminate_early: break

    def predict(self, X_test):
        coordinates_Xtest = [
                    c.reshape(-1, 1) for c in torch.unbind(X_test, dim=-1)
        ]
        with torch.autograd.no_grad():
            net_Xtest = self.net(X_test)
            
        return self.solution_structure(net_Xtest, *coordinates_Xtest)