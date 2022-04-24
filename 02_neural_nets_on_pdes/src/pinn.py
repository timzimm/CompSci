import torch
from torch import nn
import numpy as np
from ray import tune


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
        def __init__(self, input_dim, output_dim, hidden_layers, activation_function):
            super().__init__()

            prev_nodes_per_layer = input_dim
            hidden = []
            for nodes_per_layer in hidden_layers:

                hidden += [nn.Linear(prev_nodes_per_layer, nodes_per_layer), activation_function]
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
        self.activation_function = None
        self.epochs = None
        self.number_of_minibatches = None
        self.optimizer_name = None
        self.optimizer_params = {}

        # User-specified Hyperparameters
        for parameter, val in hyperparameters.items():
            if hasattr(self, parameter):
                setattr(self, parameter, val)

        # Populate unset parameters with sane default values
        self.hyperparameters_default = {
            "hidden_layers": [20, 20, 20],
            "activation_function": nn.Tanh(),
            "epochs": 200,
            "number_of_minibatches": 1,
            "optimizer_name": "lbfgs",
            "optimizer_params": None
        }
        for parameter, val in self.hyperparameters_default.items():
            if getattr(self, parameter, val) is None:
                setattr(self, parameter, val)

        self.verbose = verbose
        if self.verbose:
            self.mse_train = []
            self.report_after_e_epochs = self.epochs // 10
            self.total_loss_in_step = np.inf
            self.interior_loss_in_step = np.inf
            self.bc_loss_in_step = np.inf
            self.ic_loss_in_step = np.inf

        self.net = self.NN(self.input_dim, self.output_dim, self.hidden_layers, self.activation_function)
        
        if self.optimizer_name == "lbfgs":
            self.optimizer = torch.optim.LBFGS(self.net.parameters(), **self.optimizer_params)
        elif self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), **self.optimizer_params)
        
        
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
    
    def __generate_data_minibatch(self, X, number_of_minibatches):
        # Unpack the data into interior/ic/bc points and set up the minibatch indices
        domain_int, domain_ic, domain_bc = X
        
        N_int = domain_int.shape[0]
        N_bc = domain_bc.shape[0]
        N_ic = domain_ic.shape[0]

        # Minibatch index set to iterate over
        idx_int = np.array_split(np.arange(N_int), number_of_minibatches)
        idx_init = np.array_split(np.arange(N_ic), number_of_minibatches)
        idx_boundary = np.array_split(np.arange(N_bc), number_of_minibatches)
        
        for minibatch in range(number_of_minibatches):
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
            
            #We take derivatives wrt to the input parameters of the network
            for coordinate in coordinates_int_batch:
                coordinate.requires_grad_(True)
            
            yield coordinates_int_batch, coordinates_ic_batch, coordinates_bc_batch
      
    def __compute_loss(self, net, coordinates_int, coordinates_ic, coordinates_bc):
        
        loss_fn = torch.nn.MSELoss()
        
        #Interior
        net_int = net(*coordinates_int)
        u_int = self.solution_structure(net_int,*coordinates_int)
        mse_interior = loss_fn(*self.pde(u_int,*coordinates_int))
                    
        # Initial conditions
        net_ic = net(*coordinates_ic)
        u_ic = self.solution_structure(net_ic, *coordinates_ic)
        mse_ic = loss_fn(*self.ic(u_ic, *coordinates_ic))

        # Boundary Condition
        net_bc = net(*coordinates_bc)
        u_bc = self.solution_structure(net_bc, *coordinates_bc)
        mse_boundary = loss_fn(*self.bc(u_bc, *coordinates_bc))
        
        return mse_interior, mse_ic, mse_boundary
        
    def __compute_loss_and_gradient(self, net, optimizer, coordinates_int, coordinates_ic, coordinates_bc):
        
        optimizer.zero_grad()
        
        mse_interior, mse_ic, mse_boundary = self.__compute_loss(net, coordinates_int, coordinates_ic, coordinates_bc)
        loss = mse_interior + mse_ic + mse_boundary

        loss.backward()

        self.total_loss_in_step = loss.item()
        self.interior_loss_in_step = mse_interior.item()
        self.bc_loss_in_step = mse_boundary.item()
        self.ic_loss_in_step = mse_ic.item()

        return loss
        
    def __train_epoch(self, net, optimizer, train_loader):
        for coordinates in train_loader:
            coordinates_int_batch, coordinates_ic_batch, coordinates_bc_batch = coordinates          
            optimizer.step(lambda: self.__compute_loss_and_gradient(net, optimizer, 
                                                                         coordinates_int_batch, 
                                                                         coordinates_ic_batch, 
                                                                         coordinates_bc_batch))
            
    def __validate_epoch(self, net, test_loader):
        loss = 0
        for coordinates in test_loader:
            coordinates_int_batch, coordinates_ic_batch, coordinates_bc_batch = coordinates
            mse_interior, mse_ic, mse_boundary = self.__compute_loss(net, coordinates_int_batch, 
                                                                          coordinates_ic_batch, 
                                                                          coordinates_bc_batch)
            loss += (mse_interior + mse_ic + mse_boundary).item()
        return loss/self.number_of_minibatches
    
    def train(self, config, X_train=None, X_test=None):

        assert X_train is not None and X_test is not None
        
        net = self.NN(self.input_dim, self.output_dim, 
                      config["layers"] * [config["nodes"]], 
                      config["activation_function"])
        if config["optimizer_name"] == "lbfgs":
            optimizer = torch.optim.LBFGS(net.parameters(), lr=config["lr"])
        elif config["optimizer_name"] == "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
                
        for e in range(self.epochs):
            train_loader = self.__generate_data_minibatch(X_train, config["number_of_minibatches"])
            test_loader = self.__generate_data_minibatch(X_test, config["number_of_minibatches"])
            
            self.__train_epoch(net, optimizer, train_loader)
            validation_loss = self.__validate_epoch(net, test_loader)
            if e == self.epochs-1:
                # Safe final model (if we make it until here)
                torch.save(net.state_dict(), "./model.pth")
            tune.report(validation_loss=validation_loss)
    
    def fit(self, X_train):
        
        if self.verbose:
            domain_int, domain_ic, domain_bc = X_train
            N_int = domain_int.shape[0]
            N_bc = domain_bc.shape[0]
            N_ic = domain_ic.shape[0]
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
        
        terminate_early = False
        for e in range(self.epochs + 1):
            train_loader = self.__generate_data_minibatch(X_train, self.number_of_minibatches)
            self.__train_epoch(self.net, self.optimizer, train_loader)

            if self.verbose:
                self.mse_train.append(
                    [
                        self.total_loss_in_step,
                        self.interior_loss_in_step,
                        self.bc_loss_in_step,
                        self.ic_loss_in_step,
                    ]
                )
                if self.hooks is not None:
                    with torch.autograd.no_grad():
                        for hook in self.hooks:
                            self.hooks_returns[hook.__name__].append(hook(self.net))

                if e % self.report_after_e_epochs == 0:
                    print(
                        f"{e:^10}|"
                        f"{self.total_loss_in_step:^15.5e}|"
                        f"{self.interior_loss_in_step:^15.5e}|"
                        f"{self.bc_loss_in_step:^15.5e}|"
                        f"{self.ic_loss_in_step:^15.5e}"
                    )
                    
            if self.events_for_loss is not None:
                for event in self.events_for_loss:
                    if event(self.total_loss_in_step) and event.terminate:
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