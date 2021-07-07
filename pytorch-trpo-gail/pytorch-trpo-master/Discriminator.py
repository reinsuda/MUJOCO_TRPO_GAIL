import torch
import torch.nn as nn
from mpi_running_mean_std import TorchRunningMeanStd


def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -torch.nn.functional.softplus(-a)


""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""


def logit_bernoulli_entropy(logits):
    ent = (1. - torch.nn.functional.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


class Discriminator(nn.Module):
    def __init__(self, env, hidden_size, device, entcoeff=0.001):
        super(Discriminator, self).__init__()
        self.observation_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.shape[0]
        self.input_shape = self.observation_shape + self.action_shape
        self.hidden_size = hidden_size
        #self.obs_rms = TorchRunningMeanStd(device, shape=self.observation_shape)
        self.fc1 = nn.Linear(self.input_shape, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)
        self.device = device
        self.ent_coeff = entcoeff
        self.criterion = torch.nn.BCELoss()

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        x = torch.tanh(self.fc1(state_action))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_reward(self, state, action):
        state = torch.Tensor(state).to(self.device)
        #state = (state - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var)
        action = torch.Tensor(action).to(self.device)
        logit = self.forward(state.unsqueeze(0).float(), action.unsqueeze(0))
        return -torch.log(1 - torch.sigmoid(logit) + 1e-8).cpu().detach().numpy()

    def compute_acc(self, expert_state, expert_action, learner_state, learner_action):
        expert_state = torch.Tensor(expert_state).to(self.device)
        expert_action = torch.Tensor(expert_action).to(self.device)
        learner_state = torch.Tensor(learner_state).to(self.device)
        learner_action = torch.Tensor(learner_action).to(self.device)
        expert_logits = self.forward(expert_state, expert_action)
        learner_logit = self.forward(learner_state, learner_action.squeeze(1))
        expert_accuracy = ((torch.sigmoid(expert_logits) > 0.5).float()).mean()
        learner_accuracy = ((torch.sigmoid(learner_logit) < 0.5).float()).mean()
        return expert_accuracy, learner_accuracy

    def process_loss(self, expert_state, expert_action, learner_state, learner_action):
        expert_state = torch.Tensor(expert_state).to(self.device)
        expert_action = torch.Tensor(expert_action).to(self.device)
        learner_state = torch.Tensor(learner_state).to(self.device)
        learner_action = torch.Tensor(learner_action).to(self.device)
        expert_logits = self.forward(expert_state, expert_action)
        #self.obs_rms.update(expert_state)
        #self.obs_rms.update(learner_state)
        #print(learner_action.shape, learner_state.shape)
        learner_logit = self.forward(learner_state, learner_action.squeeze(1))

        expert_loss = self.criterion(torch.sigmoid(expert_logits), torch.ones((expert_action.shape[0], 1)).to(self.device))
        learner_loss = self.criterion(torch.sigmoid(learner_logit), torch.zeros((learner_action.shape[0], 1)).to(self.device))
        d_entropy = torch.mean(logit_bernoulli_entropy(torch.cat([expert_logits, expert_logits], dim=0)))

        return expert_loss + learner_loss - d_entropy * self.ent_coeff
