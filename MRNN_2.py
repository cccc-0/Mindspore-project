import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import XavierUniform

class TGCNGraphConvolution(nn.Cell):
    def __init__(self, adj, num_gru_units, output_dim, bias=0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.laplacian = P.MatMul().transpose_b()(adj, adj)
        self.weights = nn.Parmeter(XavierUniform()(self._num_gru_units + 1, self._output_dim))
        self.biases = nn.Parameter(XavierUniform()(self._output_dim))

    def construct(self, inputs, hidden_state):
        batch_size, num_nodes = P.Shape()(inputs)
        inputs = P.Reshape()(inputs, (batch_size, num_nodes, 1))
        hidden_state = P.Reshape()(hidden_state, (batch_size, num_nodes, self._num_gru_units))
        concatenation = P.Concat(-1)((inputs, hidden_state))
        concatenation = P.Reshape()(concatenation, (num_nodes, (self._num_gru_units + 1) * batch_size))
        a_times_concat = P.MatMul()(self.laplacian, concatenation)
        a_times_concat = P.Reshape()(a_times_concat, (num_nodes, self._num_gru_units + 1, batch_size))
        a_times_concat = P.Transpose()(a_times_concat, (0, 2, 1))
        a_times_concat = P.Reshape()(a_times_concat, (batch_size * num_nodes, self._num_gru_units + 1))
        outputs = P.MatMul()(a_times_concat, self.weights) + self.biases
        outputs = P.Reshape()(outputs, (batch_size, num_nodes, self._output_dim))
        outputs = P.Reshape()(outputs, (batch_size, num_nodes * self._output_dim))
        return outputs

class TGCNCell(nn.Cell):
    def __init__(self, adj, input_dim, hidden_dim):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.adj = adj
        self.graph_conv1 = TGCNGraphConvolution(adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0)
        self.graph_conv2 = TGCNGraphConvolution(adj, self._hidden_dim, self._hidden_dim)

    def construct(self, inputs, hidden_state):
        concatenation = nn.Sigmoid()(self.graph_conv1(inputs, hidden_state))
        r, u = P.Split(2, 2)(concatenation, 2)
        c = nn.Tanh()(self.graph_conv2(inputs, r * hidden_state))
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

class TGCN(nn.Cell):
    def __init__(self, adj, hidden_dim):
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.adj = adj
        self.tgcn_cell = TGCNCell(adj, self._input_dim, self._hidden_dim)

    def construct(self, inputs):
        batch_size, seq_len, num_nodes = P.Shape()(inputs)
        assert self._input_dim == num_nodes
        hidden_state = P.ZerosLike()(inputs)
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            output = P.Reshape()(output, (batch_size, num_nodes, self._hidden_dim))
        return output

class MLTGCN_2(nn.Cell):
    def __init__(self, adj1, adj2, hidden_dim, config):
        super(MLTGCN_2, self).__init__()
        self.tgcn1 = TGCN(adj1, hidden_dim)
        self.tgcn2 = TGCN(adj2, hidden_dim)
        self.task1_head = nn.Dense(hidden_dim, config.pre_len)
        self.task2_head = nn.Dense(hidden_dim, config.pre_len)

    def construct(self, inputs1, inputs2):
        hidden_state1 = self.tgcn1(inputs1)
        hidden_state2 = self.tgcn2(inputs2)
        output_task1 = self.task1_head(hidden_state1)
        output_task2 = self.task2_head(hidden_state2)
        return output_task1, output_task2


class MLTGCN_3(nn.Cell):
    def __init__(self, adj1, adj2, adj3, hidden_dim, config):
        super(MLTGCN_3, self).__init__()

        self.tgcn1 = TGCN(adj1, hidden_dim)
        self.tgcn2 = TGCN(adj2, hidden_dim)
        self.tgcn3 = TGCN(adj3, hidden_dim)

        self.task1_head = nn.Dense(hidden_dim, config.pre_len)
        self.task2_head = nn.Dense(hidden_dim, config.pre_len)
        self.task3_head = nn.Dense(hidden_dim, config.pre_len)

    def construct(self, inputs1, inputs2, inputs3):
        # 分别计算两个TGCNCell的隐藏状态
        hidden_state1 = self.tgcn1(inputs1)
        hidden_state2 = self.tgcn2(inputs2)
        hidden_state3 = self.tgcn3(inputs3)

        # 分别通过线性层得到两个任务的输出
        output_task1 = self.task1_head(hidden_state1)
        output_task2 = self.task2_head(hidden_state2)
        output_task3 = self.task3_head(hidden_state3)
        return output_task1, output_task2, output_task3

class MLTGCN_4(nn.Cell):
    def __init__(self, adj1, adj2, adj3, adj4, hidden_dim, config):
        super(MLTGCN_4, self).__init__()

        self.tgcn1 = TGCN(adj1, hidden_dim)
        self.tgcn2 = TGCN(adj2, hidden_dim)
        self.tgcn3 = TGCN(adj3, hidden_dim)
        self.tgcn4 = TGCN(adj4, hidden_dim)

        self.task1_head = nn.Dense(hidden_dim, config.pre_len)
        self.task2_head = nn.Dense(hidden_dim, config.pre_len)
        self.task3_head = nn.Dense(hidden_dim, config.pre_len)
        self.task4_head = nn.Dense(hidden_dim, config.pre_len)

    def construct(self, inputs1, inputs2, inputs3, inputs4):
        # 分别计算两个TGCNCell的隐藏状态
        hidden_state1 = self.tgcn1(inputs1)
        hidden_state2 = self.tgcn2(inputs2)
        hidden_state3 = self.tgcn3(inputs3)
        hidden_state4 = self.tgcn4(inputs4)

        # 分别通过线性层得到两个任务的输出
        output_task1 = self.task1_head(hidden_state1)
        output_task2 = self.task2_head(hidden_state2)
        output_task3 = self.task3_head(hidden_state3)
        output_task4 = self.task4_head(hidden_state4)
        return output_task1, output_task2, output_task3, output_task4

class MLTGCN_5(nn.Cell):
    def __init__(self, adj1, adj2, adj3, adj4, adj5, hidden_dim, config):
        super(MLTGCN_5, self).__init__()

        self.tgcn1 = TGCN(adj1, hidden_dim)
        self.tgcn2 = TGCN(adj2, hidden_dim)
        self.tgcn3 = TGCN(adj3, hidden_dim)
        self.tgcn4 = TGCN(adj4, hidden_dim)
        self.tgcn5 = TGCN(adj5, hidden_dim)

        self.task1_head = nn.Dense(hidden_dim, config.pre_len)
        self.task2_head = nn.Dense(hidden_dim, config.pre_len)
        self.task3_head = nn.Dense(hidden_dim, config.pre_len)
        self.task4_head = nn.Dense(hidden_dim, config.pre_len)
        self.task5_head = nn.Dense(hidden_dim, config.pre_len)

    def construct(self, inputs1, inputs2, inputs3, inputs4, inputs5):
        # 分别计算两个TGCNCell的隐藏状态
        hidden_state1 = self.tgcn1(inputs1)
        hidden_state2 = self.tgcn2(inputs2)
        hidden_state3 = self.tgcn3(inputs3)
        hidden_state4 = self.tgcn4(inputs4)
        hidden_state5 = self.tgcn5(inputs5)

        # 分别通过线性层得到两个任务的输出
        output_task1 = self.task1_head(hidden_state1)
        output_task2 = self.task2_head(hidden_state2)
        output_task3 = self.task3_head(hidden_state3)
        output_task4 = self.task4_head(hidden_state4)
        output_task5 = self.task5_head(hidden_state5)
        return output_task1, output_task2, output_task3, output_task4, output_task5

class MLTGCN_6(nn.Cell):
    def __init__(self, adj1, adj2, adj3, adj4, adj5, adj6,hidden_dim, config):
        super(MLTGCN_6, self).__init__()

        self.tgcn1 = TGCN(adj1, hidden_dim)
        self.tgcn2 = TGCN(adj2, hidden_dim)
        self.tgcn3 = TGCN(adj3, hidden_dim)
        self.tgcn4 = TGCN(adj4, hidden_dim)
        self.tgcn5 = TGCN(adj5, hidden_dim)
        self.tgcn6 = TGCN(adj6, hidden_dim)

        self.task1_head = nn.Dense(hidden_dim, config.pre_len)
        self.task2_head = nn.Dense(hidden_dim, config.pre_len)
        self.task3_head = nn.Dense(hidden_dim, config.pre_len)
        self.task4_head = nn.Dense(hidden_dim, config.pre_len)
        self.task5_head = nn.Dense(hidden_dim, config.pre_len)
        self.task6_head = nn.Dense(hidden_dim, config.pre_len)

    def construct(self, inputs1, inputs2, inputs3, inputs4, inputs5, inputs6):
        # 分别计算两个TGCNCell的隐藏状态
        hidden_state1 = self.tgcn1(inputs1)
        hidden_state2 = self.tgcn2(inputs2)
        hidden_state3 = self.tgcn3(inputs3)
        hidden_state4 = self.tgcn4(inputs4)
        hidden_state5 = self.tgcn5(inputs5)
        hidden_state6 = self.tgcn6(inputs6)

        # 分别通过线性层得到两个任务的输出
        output_task1 = self.task1_head(hidden_state1)
        output_task2 = self.task2_head(hidden_state2)
        output_task3 = self.task3_head(hidden_state3)
        output_task4 = self.task4_head(hidden_state4)
        output_task5 = self.task5_head(hidden_state5)
        output_task6 = self.task6_head(hidden_state6)
        return output_task1, output_task2, output_task3, output_task4, output_task5, output_task6

class MLTGCN_7(nn.Cell):
    def __init__(self, adj1, adj2, adj3, adj4, adj5, adj6, adj7, hidden_dim, config):
        super(MLTGCN_7, self).__init__()

        self.tgcn1 = TGCN(adj1, hidden_dim)
        self.tgcn2 = TGCN(adj2, hidden_dim)
        self.tgcn3 = TGCN(adj3, hidden_dim)
        self.tgcn4 = TGCN(adj4, hidden_dim)
        self.tgcn5 = TGCN(adj5, hidden_dim)
        self.tgcn6 = TGCN(adj6, hidden_dim)
        self.tgcn7 = TGCN(adj7, hidden_dim)

        self.task1_head = nn.Dense(hidden_dim, config.pre_len)
        self.task2_head = nn.Dense(hidden_dim, config.pre_len)
        self.task3_head = nn.Dense(hidden_dim, config.pre_len)
        self.task4_head = nn.Dense(hidden_dim, config.pre_len)
        self.task5_head = nn.Dense(hidden_dim, config.pre_len)
        self.task6_head = nn.Dense(hidden_dim, config.pre_len)
        self.task7_head = nn.Dense(hidden_dim, config.pre_len)

    def construct(self, inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7):
        # 分别计算两个TGCNCell的隐藏状态
        hidden_state1 = self.tgcn1(inputs1)
        hidden_state2 = self.tgcn2(inputs2)
        hidden_state3 = self.tgcn3(inputs3)
        hidden_state4 = self.tgcn4(inputs4)
        hidden_state5 = self.tgcn5(inputs5)
        hidden_state6 = self.tgcn6(inputs6)
        hidden_state7 = self.tgcn7(inputs7)

        # 分别通过线性层得到两个任务的输出
        output_task1 = self.task1_head(hidden_state1)
        output_task2 = self.task2_head(hidden_state2)
        output_task3 = self.task3_head(hidden_state3)
        output_task4 = self.task4_head(hidden_state4)
        output_task5 = self.task5_head(hidden_state5)
        output_task6 = self.task6_head(hidden_state6)
        output_task7 = self.task7_head(hidden_state7)
        return output_task1, output_task2, output_task3, output_task4, output_task5, output_task6, output_task7

class MLTGCN_8(nn.Cell):
    def __init__(self, adj1, adj2, adj3, adj4, adj5, adj6, adj7, adj8, hidden_dim, config):
        super(MLTGCN_8, self).__init__()

        self.tgcn1 = TGCN(adj1, hidden_dim)
        self.tgcn2 = TGCN(adj2, hidden_dim)
        self.tgcn3 = TGCN(adj3, hidden_dim)
        self.tgcn4 = TGCN(adj4, hidden_dim)
        self.tgcn5 = TGCN(adj5, hidden_dim)
        self.tgcn6 = TGCN(adj6, hidden_dim)
        self.tgcn7 = TGCN(adj7, hidden_dim)
        self.tgcn8 = TGCN(adj8, hidden_dim)

        self.task1_head = nn.Dense(hidden_dim, config.pre_len)
        self.task2_head = nn.Dense(hidden_dim, config.pre_len)
        self.task3_head = nn.Dense(hidden_dim, config.pre_len)
        self.task4_head = nn.Dense(hidden_dim, config.pre_len)
        self.task5_head = nn.Dense(hidden_dim, config.pre_len)
        self.task6_head = nn.Dense(hidden_dim, config.pre_len)
        self.task7_head = nn.Dense(hidden_dim, config.pre_len)
        self.task8_head = nn.Dense(hidden_dim, config.pre_len)

    def construct(self, inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8):
        # 分别计算两个TGCNCell的隐藏状态
        hidden_state1 = self.tgcn1(inputs1)
        hidden_state2 = self.tgcn2(inputs2)
        hidden_state3 = self.tgcn3(inputs3)
        hidden_state4 = self.tgcn4(inputs4)
        hidden_state5 = self.tgcn5(inputs5)
        hidden_state6 = self.tgcn6(inputs6)
        hidden_state7 = self.tgcn7(inputs7)
        hidden_state8 = self.tgcn8(inputs8)

        # 分别通过线性层得到两个任务的输出
        output_task1 = self.task1_head(hidden_state1)
        output_task2 = self.task2_head(hidden_state2)
        output_task3 = self.task3_head(hidden_state3)
        output_task4 = self.task4_head(hidden_state4)
        output_task5 = self.task5_head(hidden_state5)
        output_task6 = self.task6_head(hidden_state6)
        output_task7 = self.task7_head(hidden_state7)
        output_task8 = self.task8_head(hidden_state8)
        return output_task1, output_task2, output_task3, output_task4, output_task5, output_task6, output_task7, output_task8

class MLTGCN_9(nn.Cell):
    def __init__(self, adj1, adj2, adj3, adj4, adj5, adj6, adj7, adj8, adj9, hidden_dim, config):
        super(MLTGCN_9, self).__init__()

        self.tgcn1 = TGCN(adj1, hidden_dim)
        self.tgcn2 = TGCN(adj2, hidden_dim)
        self.tgcn3 = TGCN(adj3, hidden_dim)
        self.tgcn4 = TGCN(adj4, hidden_dim)
        self.tgcn5 = TGCN(adj5, hidden_dim)
        self.tgcn6 = TGCN(adj6, hidden_dim)
        self.tgcn7 = TGCN(adj7, hidden_dim)
        self.tgcn8 = TGCN(adj8, hidden_dim)
        self.tgcn9 = TGCN(adj9, hidden_dim)

        self.task1_head = nn.Dense(hidden_dim, config.pre_len)
        self.task2_head = nn.Dense(hidden_dim, config.pre_len)
        self.task3_head = nn.Dense(hidden_dim, config.pre_len)
        self.task4_head = nn.Dense(hidden_dim, config.pre_len)
        self.task5_head = nn.Dense(hidden_dim, config.pre_len)
        self.task6_head = nn.Dense(hidden_dim, config.pre_len)
        self.task7_head = nn.Dense(hidden_dim, config.pre_len)
        self.task8_head = nn.Dense(hidden_dim, config.pre_len)
        self.task9_head = nn.Dense(hidden_dim, config.pre_len)

    def construct(self, inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9):
        # 分别计算两个TGCNCell的隐藏状态
        hidden_state1 = self.tgcn1(inputs1)
        hidden_state2 = self.tgcn2(inputs2)
        hidden_state3 = self.tgcn3(inputs3)
        hidden_state4 = self.tgcn4(inputs4)
        hidden_state5 = self.tgcn5(inputs5)
        hidden_state6 = self.tgcn6(inputs6)
        hidden_state7 = self.tgcn7(inputs7)
        hidden_state8 = self.tgcn8(inputs8)
        hidden_state9 = self.tgcn9(inputs9)

        # 分别通过线性层得到两个任务的输出
        output_task1 = self.task1_head(hidden_state1)
        output_task2 = self.task2_head(hidden_state2)
        output_task3 = self.task3_head(hidden_state3)
        output_task4 = self.task4_head(hidden_state4)
        output_task5 = self.task5_head(hidden_state5)
        output_task6 = self.task6_head(hidden_state6)
        output_task7 = self.task7_head(hidden_state7)
        output_task8 = self.task8_head(hidden_state8)
        output_task9 = self.task9_head(hidden_state9)
        return output_task1, output_task2, output_task3, output_task4, output_task5, output_task6, output_task7, output_task8, output_task9

class MLTGCN_10(nn.Cell):
    def __init__(self, adj1, adj2, adj3, adj4, adj5, adj6, adj7, adj8, adj9, adj10, hidden_dim, config):
        super(MLTGCN_10, self).__init__()

        self.tgcn1 = TGCN(adj1, hidden_dim)
        self.tgcn2 = TGCN(adj2, hidden_dim)
        self.tgcn3 = TGCN(adj3, hidden_dim)
        self.tgcn4 = TGCN(adj4, hidden_dim)
        self.tgcn5 = TGCN(adj5, hidden_dim)
        self.tgcn6 = TGCN(adj6, hidden_dim)
        self.tgcn7 = TGCN(adj7, hidden_dim)
        self.tgcn8 = TGCN(adj8, hidden_dim)
        self.tgcn9 = TGCN(adj9, hidden_dim)
        self.tgcn10 = TGCN(adj10, hidden_dim)

        self.task1_head = nn.Dense(hidden_dim, config.pre_len)
        self.task2_head = nn.Dense(hidden_dim, config.pre_len)
        self.task3_head = nn.Dense(hidden_dim, config.pre_len)
        self.task4_head = nn.Dense(hidden_dim, config.pre_len)
        self.task5_head = nn.Dense(hidden_dim, config.pre_len)
        self.task6_head = nn.Dense(hidden_dim, config.pre_len)
        self.task7_head = nn.Dense(hidden_dim, config.pre_len)
        self.task8_head = nn.Dense(hidden_dim, config.pre_len)
        self.task9_head = nn.Dense(hidden_dim, config.pre_len)
        self.task10_head = nn.Dense(hidden_dim, config.pre_len)

    def construct(self, inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10):
        # 分别计算两个TGCNCell的隐藏状态
        hidden_state1 = self.tgcn1(inputs1)
        hidden_state2 = self.tgcn2(inputs2)
        hidden_state3 = self.tgcn3(inputs3)
        hidden_state4 = self.tgcn4(inputs4)
        hidden_state5 = self.tgcn5(inputs5)
        hidden_state6 = self.tgcn6(inputs6)
        hidden_state7 = self.tgcn7(inputs7)
        hidden_state8 = self.tgcn8(inputs8)
        hidden_state9 = self.tgcn9(inputs9)
        hidden_state10 = self.tgcn10(inputs10)

        # 分别通过线性层得到两个任务的输出
        output_task1 = self.task1_head(hidden_state1)
        output_task2 = self.task2_head(hidden_state2)
        output_task3 = self.task3_head(hidden_state3)
        output_task4 = self.task4_head(hidden_state4)
        output_task5 = self.task5_head(hidden_state5)
        output_task6 = self.task6_head(hidden_state6)
        output_task7 = self.task7_head(hidden_state7)
        output_task8 = self.task8_head(hidden_state8)
        output_task9 = self.task9_head(hidden_state9)
        output_task10 = self.task10_head(hidden_state10)
        return output_task1, output_task2, output_task3, output_task4, output_task5, output_task6, output_task7, output_task8, output_task9, output_task10