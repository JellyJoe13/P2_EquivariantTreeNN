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
    """
    Creates a baseline model consisting of linear layers and relu activation functions in between.

    :param n_params: number of parameter the original model has
    :type n_params: int
    :param input_dim: input dimension this model should have. should be number of elements in the input sequence
        times the dimension of each element
    :type input_dim: int
    :param output_dim: dimension of the output
    :type output_dim: int
    :param n_layer: number of layers the model should have
    :type n_layer: int
    :param tolerance: factor by how much the number of parameters may be exceeded (only roughly as the RELU layer
        parameters are not accounted for here)
    :type tolerance: float
    :param end_tolerance: at the end the model parameters of the constructed model will be checked again. This
        parameter controls the parameter by which factor the parameter count may exceed the input parameters with
        still being considered a model one could and should use
    :type end_tolerance: float
    :param iteration_increase: value by how much the factor increases in each iteration
    :type iteration_increase: float
    :param start_factor: value controlling with which value the factor should start
    :type start_factor: float
    :return: model and bool whether the model should be used or has too many parameters
    :rtype: typing.Tuple[torch.nn.Sequential, bool]
    """
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
    """
    Small function calculating the total number of parameters of a model.

    :param model: Model for which to count the parameters for
    :type model: torch.nn.Module
    :return: Number of parameters
    :rtype: int
    """
    return sum([p.numel() for p in model.parameters()])
