import numpy as np
import mindspore as ms
from mindspore import nn, ops,Tensor





def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    #for group_params in optimizer.group_params:
        #group_params['lr'] = lr
    #conv_params = list(filter(lambda x: 'conv' in x.name, optimizer.trainable_params()))
    #no_conv_params = list(filter(lambda x: 'conv' not in x.name, optimizer.trainable_params()))
    optimizer.group_params = [{'order_params': optimizer.trainable_params()}]
    for group_params in optimizer.group_params:
        group_params['lr'] = lr

#ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self,  board_height, board_width, Mheight, Nwidth):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.Mheight = Mheight
        self.Nwidth = Nwidth

        self.dense1 = nn.Dense(board_height * board_width + Mheight * Nwidth, 256, activation=ms.nn.ReLU())

        # action policy layers
        self.dense2 = nn.Dense(256, 128, activation=ms.nn.ReLU())
        self.act_fc1 = nn.Dense(128, board_height * board_width, activation=ms.nn.LogSoftmax())
        # state value layers
        self.dense3 = nn.Dense(256, 128, activation=ms.nn.ReLU())
        self.dense4 = nn.Dense(128, 64, activation=ms.nn.ReLU())
        self.val_fc2 = nn.Dense(64, 1, activation=ms.nn.Tanh())


    def construct(self, state_input):
        x = self.dense1(state_input)
        # action policy layers
        x_act = self.dense2(x)
        # x_act = x_act.view(-1, 128)
        x_act = self.act_fc1(x_act)
        # state value layers
        x_val = self.dense3(x)
        # x_val = x_val.view(-1, 128)
        x_val = self.dense4(x_val)
        x_val = self.val_fc2(x_val)
        return x_act, x_val



class PolicyValueNet():
    def __init__(self, board_width, board_height, Mheight, Nwidth,
                model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.Mheight = Mheight
        self.Nwidth = Nwidth
        self.l2_const = 1e-4
       # if self.use_gpu:
             #self.policy_value_net = Net(board_width, board_height, Mheight, Nwidth) set_train() .to_float(ms.float32).set_device('GPU')
        self.policy_value_net = Net(board_width, board_height, Mheight, Nwidth).set_train()
        #else:
             #self.policy_value_net = Net(board_width, board_height, Mheight, Nwidth).set_train().to_float(ms.float32)
        self.optimizer = nn.Adam(params=self.policy_value_net.trainable_params(), weight_decay=self.l2_const)

        if model_file:
             net_params =ms.load_checkpoint(model_file)
             ms.load_param_into_net(self.policy_value_net, net_params)
             #self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        #if self.use_gpu:
            #state_batch = ms.Tensor(state_batch, ms.float32).to_type(ms.float16).to('GPU')
            #log_act_probs, value = self.policy_value_net(state_batch)
            #act_probs = np.exp(log_act_probs.asnumpy())
            #return act_probs, value.asnumpy()
        #else:
            state_batch = ms.Tensor(state_batch, ms.float32)
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.asnumpy())
            return act_probs, value.asnumpy()

    def policy_value_fn(self, board):
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, self.board_height * self.board_width + self.Mheight * self.Nwidth
                ))
        #if self.use_gpu:
            #log_act_probs, value = self.policy_value_net(
                #ms.Tensor(current_state, ms.float32).to_type(ms.float16).to('GPU'))
            #act_probs = np.exp(log_act_probs.asnumpy())
        #else:
            #log_act_probs, value = self.policy_value_net(
                #ms.Tensor(current_state).to_float(ms.float32))
            #act_probs = np.exp(log_act_probs.asnumpy())
        #act_probs = zip(legal_positions, act_probs[legal_positions])
        #value = value.asnumpy()[0][0]
        log_act_probs, value = self.policy_value_net(ms.Tensor(current_state, ms.float32))
        act_probs = np.exp(log_act_probs.asnumpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.asnumpy()[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
       state_batch = Tensor(state_batch, ms.float32)
       mcts_probs = Tensor(mcts_probs, ms.float32)
       winner_batch = Tensor(winner_batch, ms.float32)
       # set learning rate
       set_learning_rate(self.optimizer, lr)
       # forward
       model= self.policy_value_net
       def forward_fn(state_batch, mcts_probs, winner_batch):
           log_act_probs, value = model(state_batch)
           los = nn.MSELoss()
           logits = winner_batch
           #logits = Tensor(winner_batch, ms.float32)
           value = value.reshape(-1)
           labels = value
           value_loss = los(logits, labels)
           su = ms.ops.ReduceSum()
           rsu = su(mcts_probs * log_act_probs, 1)
           policy_loss = -ops.mean(rsu)
           loss = value_loss + policy_loss
           ex = ops.exp(log_act_probs)
           sum = ops.ReduceSum()
           resum = sum(ex * log_act_probs, 1)
           entropy = -ops.mean(resum)
           return loss, entropy

       grad_fn = ops.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)
       (loss, entropy), grads = grad_fn(state_batch, mcts_probs, winner_batch)
       loss = ops.depend(loss, self.optimizer(grads))

       return loss, entropy


    def get_policy_param(self):
        net_params = self.policy_value_net.parameters_dict()
        return net_params

    def save_model(self, model_file):
        net_params = self.get_policy_param()
        ms.save_checkpoint(net_params, model_file)

