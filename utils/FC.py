from torch import nn

class FC(nn.Module):
    def __init__(self, dims_list, act=None, output_act=None):
        super(FC, self).__init__()

        neurons = dims_list[:-1]
        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        self.act_func = act
        self.lastLayer = nn.Linear(neurons[-1], dims_list[-1])

        self.output_act = output_act

        # weights initialization could have a large effect on final result
        self._weight_initialization(self.hidden, self.act_func)
        self._weight_initialization(self.lastLayer, self.output_act)

    def _weight_initialization(self, layers, act_func):
        for m in layers.modules():
            if isinstance(m, nn.Linear):
                if type(act_func) == type(nn.Tanh()):
                    nn.init.xavier_uniform_(m.weight.data)
                elif type(act_func) == type(nn.ReLU()):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif type(act_func) == type(nn.LeakyReLU()):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
                else:
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        for layer in self.hidden:
            if self.act_func is not None:
                x = self.act_func(layer(x))
            else:
                x = layer(x)
        if self.output_act is not None:
            return self.output_act(self.lastLayer(x))
        else:
            return self.lastLayer(x)