import torch
from torch.nn import Module, RNN, ELU, Linear, Parameter


class RnnNetworkTypeS(Module):
    """
    Node Network of type S using a RNN.
    """
    def __init__(
            self,
            k: int = 2,
            hidden_dim: int = 128,
            use_state: bool = False,
            bidirectional: bool = False
    ):
        super().__init__()
        self.k = k
        self.rnn = RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.final_layer_elu = ELU()
        self.final_layer_linear = Linear(
            2*hidden_dim if bidirectional else hidden_dim,
            hidden_dim
        )
        self.use_state = use_state
        if self.use_state:
            self.own_state = Parameter(torch.empty((1, hidden_dim)))
            self.state_layer = Linear(hidden_dim, hidden_dim)
        return

    def forward(self, embedded_x):
        # shift stack
        shift_stack = torch.stack(
            [
                embedded_x[..., idx:embedded_x.shape[-2]-self.k+idx, :]
                for idx in range(self.k)
            ],
            dim=-2
        )

        # problem: 4d vector - reshape
        reshaped_shift_stack = shift_stack.reshape(-1, *shift_stack.shape[2:])

        # send through RNN
        # throw away hidden representation
        rnn_output, _ = self.rnn(reshaped_shift_stack)

        print(rnn_output.shape)

        # shape back
        rnn_output_reshaped = rnn_output.reshape(*shift_stack.shape[:-1], -1)

        # use the last result of the rnn per sequence (RNN: many to one)
        rnn_selection = rnn_output_reshaped[..., -1, :]

        # elu
        elu_sum_seq = self.final_layer_elu(rnn_selection)

        # final linear layer
        after_final_linear = self.final_layer_linear(elu_sum_seq)

        if self.use_state:
            temp = self.state_layer(self.own_state)
            after_final_linear = torch.cat([after_final_linear, temp], dim=-2)
        # aggregate and return
        return torch.sum(after_final_linear, dim=-2)
