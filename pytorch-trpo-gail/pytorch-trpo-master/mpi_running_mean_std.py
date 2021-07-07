try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import torch
import numpy as np
from mujoco_dset import Mujoco_Dset


def update_from_moments(mean, std, count, batch_mean, batch_std, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = std * (count - 1)
    m_b = batch_std * (batch_count - 1)
    M2 = m_a + m_b + torch.square(new_mean - mean) * count + torch.square(new_mean - batch_mean) * batch_count
    new_std = M2 / (tot_count - 1)
    return new_mean, new_std, tot_count


class TorchRunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, device, shape=()):
        self._sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self._sumsq = torch.zeros(shape, dtype=torch.float64, device=device)
        self._count = torch.zeros(1, dtype=torch.float64, device=device)
        self.shape = shape

        self.mean = torch.zeros_like(self._sumsq).to(device)
        self.var = torch.zeros_like(self._sum).to(device)

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_count = x.shape[0]
        if batch_count > 1:
            batch_var = torch.var(x, dim=0)
        if batch_count == 1:
            batch_var = torch.zeros_like(batch_mean)
        # print(batch_mean, batch_var)

        if batch_count >= 1:
            self.mean, self.var, self._count = update_from_moments(self.mean,
                                                                   self.var,
                                                                   self._count,
                                                                   batch_mean,
                                                                   batch_var,
                                                                   batch_count)




if __name__ == "__main__":
    # Run with mpirun -np 2 python <filename>
    # test_data = Mujoco_Dset("data/deterministic_SAC_Hopper-v2_johnny.npz", traj_limitation=-1)
    # print(np.mean(test_data.obs, axis=0), np.var(test_data.obs, axis=0))
    # #print(test_data.obs.shape)
    # rms = TorchRunningMeanStd(torch.device("cuda:0"), [test_data.obs.shape[1]])
    # #print(rms.shape, rms.mean, rms.var)
    # rms.update(torch.Tensor(test_data.obs).to(torch.device("cuda:0")))
    # print(rms.mean, rms.var)
    #
    # print(rms.mean,rms.var)
    l1 = [1, 2, 3, 4, 5]
    l2 = [7,8]
    device = torch.device("cuda:0")
    rms = TorchRunningMeanStd(device, [1])
    rms.update(torch.Tensor(l1).to(device))
    rms.update(torch.Tensor(l2).to(device))
    print(rms.mean, rms.var)
    print(torch.var(torch.Tensor(l1 + l2)))
    print(torch.mean(torch.Tensor(l1 + l2)))
