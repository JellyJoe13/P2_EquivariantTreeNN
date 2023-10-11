import torch
from torch.nn import Module, Linear, ReLU, ELU, Parameter


class ChiralNodeNetworkTypeC(Module):
    def __init__(
            self,
            k: int = 2,
            hidden_dim: int = 128,
            use_state: bool = False
    ):
        super().__init__()
        self.k = k
        self.k_layers = [
            Linear(hidden_dim, hidden_dim)
            for _ in range(self.k)
        ]
        self.final_layer_elu = ELU()
        self.final_layer_linear = Linear(hidden_dim, hidden_dim)

        self.use_state = use_state
        if self.use_state:
            self.own_state = Parameter(torch.empty((1, hidden_dim)))
            self.state_layer = Linear(hidden_dim, hidden_dim)
        return

    def forward(self, embedded_x, inverted: bool = True):
        # run embedding through k layers
        k_embedded_x = [
            k_layer(embedded_x)
            for k_layer in self.k_layers
        ]

        # shift stacking
        _idx = torch.arange(embedded_x.shape[-2])
        shifted_embedding_stack = [
            embedding[..., torch.roll(_idx, shifts=idx), :]
            for idx, embedding in enumerate(k_embedded_x)
        ]
        shifted_embedding_stack = torch.stack(shifted_embedding_stack, dim=-2)

        # sum over k_layers
        summed_shifted_embedding_stack = torch.sum(
            shifted_embedding_stack,
            dim=-2
        )

        # ELU layer
        after_elu = self.final_layer_elu(summed_shifted_embedding_stack)

        # final linear layer (hidden, hidden)
        after_final_linear = self.final_layer_linear(after_elu)

        if self.use_state:
            temp = self.state_layer(self.own_state)
            after_final_linear = torch.cat([after_final_linear, temp], dim=-2)

        # aggregate
        aggregate = torch.sum(after_final_linear, dim=-2)

        # if inverting then fetch the other way round as well (inspired by q type implementation)
        if inverted:
            inv_result = self.forward(embedded_x.flip(-2), inverted=False)
            temp_stack = torch.stack([
                aggregate,
                inv_result
            ])
            aggregate = torch.mean(temp_stack, axis=0)

        return aggregate
