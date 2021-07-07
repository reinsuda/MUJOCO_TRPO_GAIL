import argparse
from itertools import count

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory, Memory_Discriminator
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from Discriminator import Discriminator
from mujoco_dset import Mujoco_Dset
import datetime

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
from torch.utils.tensorboard import SummaryWriter

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Ant-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=10000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make(args.env_name)
writer = SummaryWriter('runs/{}_{}_batch_size_15000'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                                                            args.env_name))

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda:0")
policy_net = Policy(num_inputs, num_actions).to(device)
value_net = Value(num_inputs).to(device)
rewarder_giver = Discriminator(env, 100, device).to(device)
data_set = Mujoco_Dset("data/sac_Ant-v2_model_4695_mean_4706.npz", train_fraction=0.7,
                       traj_limitation=-1)
# rewarder_giver.obs_rms.update(torch.Tensor(data_set.obs))
rewarder_rmsprop = torch.optim.RMSprop(rewarder_giver.parameters(), lr=3e-4)


def select_action(state, evaluate=False):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state).to(device))
    if evaluate:
        action = action_mean
    else:
        action = torch.normal(action_mean, action_std)
    return action.cpu().detach()


def update_discriminator(batch, dataset):
    inputs, label = dataset.get_next_batch(len(batch.action))
    loss = rewarder_giver.process_loss(inputs, label, batch.state, batch.action)
    rewarder_rmsprop.zero_grad()
    loss.backward()
    rewarder_rmsprop.step()
    return loss.item


def discriminator_accuracy(batch, dataset):
    inputs, label = dataset.get_next_batch(len(batch.action))
    expert_accuracy, learner_accuracy = rewarder_giver.compute_acc(inputs, label, batch.state, batch.action)
    return expert_accuracy, learner_accuracy


def update_params(batch):
    rewards = torch.Tensor(batch.reward).to(device)
    masks = torch.Tensor(batch.mask).to(device)
    actions = torch.Tensor(np.concatenate(batch.action, 0)).to(device)
    states = torch.Tensor(batch.state).to(device)
    values = value_net(Variable(states).to(device))

    returns = torch.Tensor(actions.size(0), 1).to(device)
    deltas = torch.Tensor(actions.size(0), 1).to(device)
    advantages = torch.Tensor(actions.size(0), 1).to(device)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns).to(device)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().cpu().numpy(), get_flat_grad_from(value_net).data.double().cpu().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                            get_flat_params_from(value_net).cpu().double().numpy(),
                                                            maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))

        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()

    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)


running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)


def eval_policy():
    eval_env = gym.make(args.env_name)
    eval_env.seed(args.seed)
    r_list = []
    for i in range(10):
        ob = eval_env.reset()
        ob = running_state(ob, update=False)
        r_sum = 0
        d_v = False
        while not d_v:
            ac = select_action(ob, evaluate=True)
            ac = ac.data[0].numpy()
            n_ob, r, d_v, _ = eval_env.step(ac)
            r_sum += r
            n_ob = running_state(n_ob, update=False)
            ob = n_ob
        r_list.append(r_sum)
    return np.mean(r_list)


for i_episode in count(1):
    memory = Memory()
    origin_memory = Memory_Discriminator()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        state = env.reset()
        state_origin = state
        state = running_state(state)

        reward_sum = 0
        for t in range(1000):  # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()
            irl_reward = rewarder_giver.get_reward(torch.Tensor(state_origin), torch.Tensor(action))
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            next_state_origin = next_state
            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, irl_reward)
            origin_memory.push(state_origin, np.array([action]))
            if args.render:
                env.render()
            if done:
                break

            state = next_state
            state_origin = next_state_origin
        num_steps += (t - 1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes
    batch = memory.sample()

    update_params(batch)
    # if i_episode % 3 == 0:
    batch_discriminator = origin_memory.sample()
    e_acc, l_acc = discriminator_accuracy(batch_discriminator, data_set)
    print("expert accuracy: {} learner accuracy: {}".format(e_acc, l_acc))
    update_discriminator(batch_discriminator, data_set)
    writer.add_scalar("exp acc", e_acc, i_episode)
    writer.add_scalar("lea acc", l_acc, i_episode)

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, eval_policy(), reward_batch))
        writer.add_scalar("evalue policy returns ", eval_policy(), i_episode)
