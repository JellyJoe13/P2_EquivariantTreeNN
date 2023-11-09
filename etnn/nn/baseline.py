import torch
import typing


def create_baseline_model(
        n_params: int,
        input_dim: int,
        output_dim: int = 1,
        n_layer: int = 2,
        tolerance: float = 1.05,
        end_tolerance: float = 1.1,
        iteration_increase: float = 0.1,
        start_factor: float = 2.
) -> typing.Tuple[torch.nn.Sequential, bool]:
    # for this we want to use a factor that is closes to producing the wanted parameter count

    # init factor
    factor = start_factor
    # try different factors until we find one that is closest to the initial parameters
    temp_factor = factor
    while True:
        # increase factor
        temp_factor += iteration_increase

        # init parameter counter
        params = 0

        # iterate over layers (pseudo-inverted)
        for i in range(n_layer):

            # cover the last layer
            if i == 0:
                params += output_dim * int(temp_factor)

            # cover the first layer
            elif i == (n_layer-1):
                params += input_dim * int(temp_factor ** i)

            else:
                params += int(temp_factor ** i) * int(temp_factor ** (i+1))

        # check if with this factor the number of parameters is still smaller
        if params <= (n_params*tolerance):
            factor = temp_factor
        else:
            break

    # create the layers
    layers = []
    for idx in range(n_layer):

        # create a pseudo-index
        i = n_layer - idx

        # first layer
        if idx == 0:
            layers += [torch.nn.Linear(input_dim, int(factor ** (i-1)))]
            layers += [torch.nn.ReLU()]

        # last layer
        elif idx == (n_layer-1):
            layers += [torch.nn.Linear(int(factor), output_dim)]

        # intermediate layers
        else:
            temp = (factor ** (i-1))
            layers += [torch.nn.Linear(int(temp*factor), int(temp))]
            layers += [torch.nn.ReLU()]

    # use layers to build module and return it
    model = torch.nn.Sequential(*layers)

    # last layer check
    should_use = calc_params(model) < (end_tolerance * n_params)

    return model, should_use


def calc_params(
        model: torch.nn.Module
) -> int:
    return sum([p.numel() for p in model.parameters()])
