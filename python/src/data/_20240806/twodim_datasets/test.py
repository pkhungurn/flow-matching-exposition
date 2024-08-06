import torch
from torch.distributions import MultivariateNormal, Normal, Uniform

if __name__ == "__main__":
    dist = MultivariateNormal(
        loc=torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ]),
        covariance_matrix=torch.tensor([
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 3.0]],
        ]))
    sample = dist.sample(torch.Size([4]))
    print(dist.mean)
    print(sample.shape)
    print(dist.batch_shape)
    print(dist.event_shape)

    print("=========")
    dist = Normal(loc=torch.tensor([0.0, 1.0]), scale=torch.tensor([1.0, 0.5]))
    sample = dist.sample()
    print(sample.shape)
    print(dist.batch_shape)
    print(dist.event_shape)

    print("=========")
    dist = Uniform(torch.tensor([0.0, 1.0]), torch.tensor([5.0, 2.0]))
    print(dist.batch_shape)
    print(dist.event_shape)

    print("=======")
    print(torch.linspace(0.0, 1.0, steps=11))