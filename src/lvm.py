import torch
import numpy as np
from typing import List, Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PI = torch.tensor(3.14159265359)


def multivariate_normal_log_probability(
        elements: torch.tensor,
        means: torch.tensor,
        covariances: torch.tensor,
        device: str) -> torch.tensor:
    """
    Calculates the log probability of a multivariate normal distributions.
    Evaluation of all elements for all means/covariances.
    N := number of elements, K := Number of means, D := Dimensionality
    :param elements: Subject to evaluation (N, D)
    :param means: Means of the distribution (K, D)
    :param covariances: Covariances of the distributions
    :param device:
    :return:
    """
    N, D = elements.shape
    covariances_stabilized = covariances + torch.eye(covariances.shape[-1]).to(device).unsqueeze(0) * 1e-3
    log_normalizer = -0.5 * D * torch.log(2 * PI) - 0.5 * torch.log(torch.det(covariances))
    differences = elements.unsqueeze(1) - means.unsqueeze(0)  # (N, K, D)
    log_prob = log_normalizer - 0.5 * (differences.unsqueeze(-2) @ torch.inverse(covariances_stabilized) @ differences.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return log_prob


def multivariate_normal_probability(
        elements: torch.tensor,
        means: torch.tensor,
        covariances: torch.tensor,
        device: torch.tensor) -> torch.tensor:
    """
    Calculates the probability of a multivariate normal distributions.
    Evaluation of all elements for all means/covariances.
    N := number of elements, K := Number of means, D := Dimensionality
    :param elements: Subject to evaluation (N, D)
    :param means: Means of the distribution (K, D)
    :param covariances: Covariances of the distributions
    :param device:
    :return:
    """
    log_prob = multivariate_normal_log_probability(
        elements=elements,
        means=means,
        covariances=covariances,
        device=device
    )
    return torch.exp(log_prob)


class ImageDataModel(torch.nn.Module):
    def __init__(self, device: str, z_dim: int) -> None:
        super(ImageDataModel, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.log_std_ = torch.tensor([0.0])

        self.pre_encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
        )
        self.mu = torch.nn.Linear(64, 2)
        self.logsigma = torch.nn.Linear(64, 2)

        self.f = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 784),
            torch.nn.Sigmoid()
        )

    def encode(self, X: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Encode the image to mean and log standard deviance (amortization).
        :param X: Points of a shape
        :return: Predicted distribution parameters (means, standard deviations) of the points of the shape
        """
        h = self.pre_encoder(X)
        mu = self.mu(h)
        logsigma = self.logsigma(h)
        return mu, logsigma

    def sample_reparam(self, mu: torch.tensor, logsigma: torch.tensor) -> torch.tensor:
        """
        Sampling from the variational distribution using the reparametrization trick.
        :param mu: Means of the variational Gaussian distribution
        :param logsigma: Standard deviations of the variational Gaussian distribution
        :return:
        """
        sigma = torch.exp(logsigma).pow(2)
        epsilon = torch.normal(torch.zeros((len(mu), self.z_dim)), torch.ones((len(mu), self.z_dim)))
        z = sigma * epsilon + mu
        return z

    def sample(self, size: int) -> torch.tensor:
        """
        Generates samples from this generative model.
        :param size: Number of samples
        :return:
        """
        latent_z = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(self.z_dim).to(self.device),
            covariance_matrix=torch.eye(self.z_dim).to(self.device)
        )
        z = latent_z.sample(torch.Size([size])).to(self.device)
        with torch.no_grad():
            params = self.f(z)
            conditional = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=params,
                covariance_matrix=torch.exp(self.log_std_) * torch.eye(784).repeat(size, 1, 1).to(self.device)
            )
        X = torch.flatten(conditional.sample(torch.Size([1])), 0, 1)
        return X

    def log_prior(self, z: torch.tensor) -> torch.tensor:
        """
        Calcualtes the log of the prior distribution over the samples from z.
        :param z: Samples from latent space z
        :return:
        """
        log_prior = multivariate_normal_log_probability(
            elements=z,
            means=torch.zeros(self.z_dim).to(self.device),
            covariances=torch.eye(self.z_dim).to(self.device),
            device=self.device
        ).squeeze()
        return log_prior

    def log_likelihood(self, X: torch.tensor, z: torch.tensor) -> torch.tensor:
        """
        Calculates the log likelihood over the points in set X and samples from latent space z.
        :param X: Points of a shape
        :param z: Samples from latent space z
        :return:
        """
        params = self.f(z)
        log_likelihood = multivariate_normal_log_probability(
            elements=X,
            means=params,
            covariances=torch.exp(self.log_std_) * torch.eye(784).to(self.device),
            device=self.device
        )
        return log_likelihood

    def log_joint_probability(self, X: torch.tensor, z: torch.tensor) -> torch.tensor:
        """
        Calculates the log joint probability over the points in set X and samples from latent space z.
        :param X: Points of a shape
        :param z: Samples from latent space z
        :return:
        """
        log_prior = self.log_prior(z)
        log_likelihood = self.log_likelihood(X, z)
        log_joint_probability = log_likelihood + log_prior
        return log_joint_probability

    def log_posterior(self, X: torch.tensor, z: torch.tensor) -> torch.tensor:
        """
        Calculates the log posterior over the points in set X and samples from latent space z.
        :param X: Points of a shape
        :param z: Samples from latent space z
        :return:
        """
        log_joint_probability = self.log_joint_probability(X, z)
        log_evidence = torch.log(self.evidence(X, z))
        log_posterior = log_joint_probability - log_evidence.unsqueeze(-1)
        return log_posterior.T

    def evidence(self, X: torch.tensor, z: torch.tensor) -> torch.tensor:
        """
        Calculates the shape evidence over the points in set X.
        :param X: Points of a shape
        :param z: Samples from latent space z
        :return:
        """
        joint_probability = torch.exp(self.log_joint_probability(X, z))
        evidence = joint_probability.sum(-1) / len(z)
        return evidence

    def expected_log_likelihood(self, X: torch.tensor, z: torch.tensor) -> torch.tensor:
        """
        Calculates the optimization objective of the model.
        :param X: Points of a shape
        :param z: Samples from latent space z
        :return:
        """
        log_likelihood = self.log_likelihood(X, z)
        with torch.no_grad():
            num_stabilizer = torch.max(log_likelihood)
            log_likelihood_stabilized = log_likelihood - num_stabilizer
            log_prior = self.log_prior(z)
            log_joint_probability_stabilized = log_likelihood_stabilized + log_prior
            likelihood_stabilized = torch.exp(log_likelihood_stabilized)
            evidence_stabilized = likelihood_stabilized.sum(-1) / len(z)
            log_evidence_stabilized = torch.log(evidence_stabilized)
            log_posteriors = (log_joint_probability_stabilized - log_evidence_stabilized.unsqueeze(-1)).T
            importance_w_normalized = torch.exp(log_posteriors - log_prior.unsqueeze(-1))
        expected_log_likelihood = (importance_w_normalized.T * log_likelihood).sum(-1) / len(z)
        return expected_log_likelihood.sum() / len(X)

    def simplified_expected_log_likelihood_vae(self, X: torch.tensor) -> torch.tensor:
        mu, logsigma = self.encode(X)
        z = self.sample_reparam(mu, logsigma)
        reconstructed = self.f(z)
        diffs = X - reconstructed
        covariance = 1.0 / torch.exp(self.log_std_)
        l2_loss = diffs.pow(2).sum(-1) * covariance
        kl_divergence = 0.5 * torch.sum((torch.exp(logsigma).pow(2) + mu.pow(2) - 2 * logsigma - 1.0), dim=-1)
        objective = - l2_loss - kl_divergence
        return objective.mean()

    def objective(self, X: torch.tensor, z: torch.tensor, variational: bool = False) -> torch.tensor:
        if variational:
            objective = self.simplified_expected_log_likelihood_vae(X)
        else:
            objective = self.expected_log_likelihood(X, z)
        return objective


class ShapeModel(torch.nn.Module):
    def __init__(self, device: str, z_dim: int, w_dim: int) -> None:
        super(ShapeModel, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.register_parameter(name='log_std_', param=torch.nn.Parameter(torch.log(torch.tensor(0.01))))

        self.l1_dim = 16
        self.l2_dim = 16
        self.l3_dim = 2
        parameter_count = (self.z_dim + 1) * self.l1_dim + (self.l1_dim + 1) * self.l2_dim + (self.l2_dim + 1) * self.l3_dim

        self.g = torch.nn.Sequential(
            torch.nn.Linear(self.w_dim, int(0.25 * parameter_count)),
            torch.nn.ELU(),
            torch.nn.Linear(int(0.25 * parameter_count), int(0.5 * parameter_count)),
            torch.nn.ELU(),
            torch.nn.Linear(int(0.5 * parameter_count), parameter_count)
        )

        g_parameter_count = 0
        for params in self.g.parameters():
            g_parameter_count += len(params.flatten())
        print("f parameter count:", parameter_count, ". g parameter count:", g_parameter_count)

    def f(self, z: torch.tensor, w: torch.tensor) -> torch.tensor:
        """
        Neural Network function f, the parameters theta are the output of function g
        and used to transform the samples z.
        :param z: Samples from latent space z
        :param w: Samples from latent space w
        :return:
        """
        theta = self.g(w)

        # Separate parameter vector into the weights theta of the neural network f
        w1 = theta[0:self.z_dim * self.l1_dim].reshape((self.z_dim, self.l1_dim))
        b1 = theta[self.z_dim * self.l1_dim:self.z_dim * self.l1_dim + self.l1_dim].reshape((1, self.l1_dim))
        w2 = theta[(self.z_dim * self.l1_dim + self.l1_dim):(self.z_dim * self.l1_dim + self.l1_dim) + (self.l1_dim * self.l2_dim)].reshape((self.l1_dim, self.l2_dim))
        b2 = theta[(self.z_dim * self.l1_dim + self.l1_dim) + (self.l1_dim * self.l2_dim):(self.z_dim * self.l1_dim + self.l1_dim) + (self.l1_dim * self.l2_dim) + self.l2_dim].reshape((1, self.l2_dim))
        w3 = theta[(self.z_dim * self.l1_dim + self.l1_dim) + (self.l1_dim * self.l2_dim) + self.l2_dim:len(theta) - 2].reshape((self.l2_dim, self.l3_dim))
        b3 = theta[len(theta) - 2:len(theta)].reshape((1, self.l3_dim))

        # Calculate the neural network pass f manually
        x = torch.nn.functional.elu(z @ w1 + b1)
        x = torch.nn.functional.elu(x @ w2 + b2)
        x = torch.sigmoid(x @ w3 + b3)
        #x = x @ w3 + b3
        return x

    def sample(self, size: int, w: torch.tensor) -> torch.tensor:
        """
        Generates samples from this generative model.
        :param size: Number of samples
        :param w: Latent w vector this generative process conditionally depends on
        :return:
        """
        latent_z = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(self.z_dim).to(self.device),
            covariance_matrix=torch.eye(self.z_dim).to(self.device)
        )
        z = latent_z.sample(torch.Size([size])).to(self.device)
        with torch.no_grad():
            means = self.f(z, w)
        conditional = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=means,
            covariance_matrix=torch.exp(self.log_std_) * torch.eye(2).repeat(size, 1, 1).to(self.device)
        )
        X = torch.flatten(conditional.sample(torch.Size([1])), 0, 1)
        return X

    def log_prior(self, z: torch.tensor) -> torch.tensor:
        """
        Calcualtes the log of the prior distribution over the samples from z.
        :param z: Samples from latent space z
        :return:
        """
        log_prior = multivariate_normal_log_probability(
            elements=z,
            means=torch.zeros(self.z_dim).to(self.device),
            covariances=torch.eye(self.z_dim).to(self.device),
            device=self.device
        ).squeeze()
        return log_prior

    def log_likelihood(self, X: torch.tensor, z: torch.tensor, w: torch.tensor) -> torch.tensor:
        """
        Calculates the log likelihood over the points in set X and samples from latent space z.
        :param X: Points of a shape
        :param z: Samples from latent space z
        :param w: Samples from latent space w
        :return:
        """
        means = self.f(z, w)
        log_likelihood = multivariate_normal_log_probability(
            elements=X,
            means=means,
            covariances=torch.exp(self.log_std_) * torch.eye(2).to(self.device),
            device=self.device
        )
        return log_likelihood

    def log_joint_probability(self, X: torch.tensor, z: torch.tensor, w: torch.tensor) -> torch.tensor:
        """
        Calculates the log joint probability over the points in set X and samples from latent space z.
        :param X: Points of a shape
        :param z: Samples from latent space z
        :param w: Samples from latent space w
        :return:
        """
        log_prior = self.log_prior(z)
        log_likelihood = self.log_likelihood(X, z, w)
        log_joint_probability = log_likelihood + log_prior
        return log_joint_probability

    def log_posterior(self, X: torch.tensor, z: torch.tensor, w: torch.tensor) -> torch.tensor:
        """
        Calculates the log posterior over the points in set X and samples from latent space z.
        :param X: Points of a shape
        :param z: Samples from latent space z
        :param w: Samples from latent space w
        :return:
        """
        log_joint_probability = self.log_joint_probability(X, z, w)
        log_evidence = torch.log(self.evidence(X, z, w))
        log_posterior = log_joint_probability - log_evidence.unsqueeze(-1)
        return log_posterior.T

    def evidence(self, X: torch.tensor, z: torch.tensor, w: torch.tensor) -> torch.tensor:
        """
        Calculates the shape evidence over the points in set X.
        :param X: Points of a shape
        :param z: Samples from latent space z
        :param w: Samples from latent space w
        :return:
        """
        log_likelihood = torch.exp(self.log_likelihood(X, z, w))
        evidence = log_likelihood.sum(-1) / len(z)

        return evidence

    def expected_log_likelihood(self, X: torch.tensor, z: torch.tensor, w: torch.tensor) -> torch.tensor:
        """
        Calculates the optimization objective of the model.
        :param X: Points of a shape
        :param z: Samples from latent space z
        :param w: Samples from latent space w
        :return:
        """
        log_likelihood = self.log_likelihood(X, z, w)

        with torch.no_grad():
            num_stabilizer = torch.max(log_likelihood)
            log_likelihood_stabilized = log_likelihood - num_stabilizer

            log_prior = self.log_prior(z)
            log_joint_probability_stabilized = log_likelihood_stabilized + log_prior

            likelihood_stabilized = torch.exp(log_likelihood_stabilized)
            evidence_stabilized = likelihood_stabilized.sum(-1) / len(z)

            log_evidence_stabilized = torch.log(evidence_stabilized)
            log_posteriors = (log_joint_probability_stabilized - log_evidence_stabilized.unsqueeze(-1)).T

            importance_w_normalized = torch.exp(log_posteriors - log_prior.unsqueeze(-1))

        expected_log_likelihood = (importance_w_normalized.T * log_likelihood).sum(-1) / len(z)
        return expected_log_likelihood.sum()


class SetModel(torch.nn.Module):
    def __init__(self, device: str, z_dim: int, w_dim: int) -> None:
        super(SetModel, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.shape_model = ShapeModel(device=self.device, z_dim=self.z_dim, w_dim=self.w_dim)

    def sample(self, w_size: int, z_size: int) -> torch.tensor:
        """
        Generates samples from this generative model.
        :param w_size: Number of samples from latent space w
        :param z_size: Number of samples from latent space  z
        :return:
        """
        latent_w = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.ones(self.w_dim),
            covariance_matrix=torch.eye(self.w_dim)
        )
        w = latent_w.sample(torch.Size([w_size])).to(self.device)

        D = []
        for wi in w:
            X = self.shape_model.sample(size=z_size, w=wi)
            D.append(X)
        D = torch.stack(D, 0)
        return D

    def log_prior(self, w: torch.tensor) -> torch.tensor:
        """
        Calculates the log prior of samples from latent space w.
        :param w: Samples from latent space w
        :return:
        """
        log_prior = multivariate_normal_log_probability(
            elements=w,
            means=torch.ones(self.w_dim).to(self.device),
            covariances=torch.eye(self.w_dim).to(self.device),
            device=self.device
        ).squeeze()
        return log_prior

    def log_likelihood(self, D: List[torch.tensor], w: torch.tensor, z: torch.tensor) -> torch.tensor:
        """
        Calculates the log likelihood of the data set of shapes over a set of samples from w/z
        :param D: List of shapes
        :param w: Samples from latent distribution w
        :param z: Samples from latent distribution z
        :return:
        """
        log_likelihood = []
        for X in D:
            for wi in w:
                log_likelihood.append(self.shape_model.expected_log_likelihood(X, z, wi))
        log_likelihood = torch.stack(log_likelihood, 0).reshape((len(D), len(w)))
        return log_likelihood

    def log_joint_probability(self, D: List[torch.tensor], w: torch.tensor, z: torch.tensor) -> torch.tensor:
        """
        Calculates the log joint probability over a set of shapes and samples w/z.
        :param D: List of shapes
        :param w: Samples from latent distribution w
        :param z: Samples from latent distribution z
        :return:
        """
        log_prior = self.log_prior(w)
        log_likelihood = self.log_likelihood(D, w, z)
        log_joint_probability = log_likelihood + log_prior
        return log_joint_probability

    def log_posterior(self, D: List[torch.tensor], w: torch.tensor, z: torch.tensor):
        """
        Calculates the log posterior over a set of shapes and samples w/z.
        :param D: List of shapes
        :param w: Samples from latent distribution w
        :param z: Samples from latent distribution z
        :return:
        """
        log_joint_probability = self.log_joint_probability(D, w, z)
        log_evidence = torch.log(self.evidence(D, w, z))
        log_posterior = log_joint_probability - log_evidence.unsqueeze(-1)
        return log_posterior.T

    def evidence(self, D, w, z):
        """
        Calculates the data set evidence.
        :param D: List of shapes
        :param w: Samples from latent distribution w
        :param z: Samples from latent distribution z
        :return:
        """
        log_likelihood = torch.exp(self.log_likelihood(D, w, z))
        evidence = log_likelihood.sum(-1) / len(w)
        return evidence

    def expected_log_likelihood(self, D, w, z):
        """
        Calculates the optimization objective
        :param D: List of shapes
        :param w: Samples from latent distribution w
        :param z: Samples from latent distribution z
        :return:
        """
        log_likelihood = self.log_likelihood(D, w, z)

        with torch.no_grad():
            num_stabilizer = torch.max(log_likelihood)
            log_likelihood_stabilized = log_likelihood - num_stabilizer

            log_prior = self.log_prior(w)
            log_joint_probability_stabilized = log_likelihood_stabilized + log_prior

            likelihood_stabilized = torch.exp(log_likelihood_stabilized)
            evidence_stabilized = likelihood_stabilized.sum(-1) / len(w)

            log_evidence_stabilized = torch.log(evidence_stabilized)
            log_posteriors = (log_joint_probability_stabilized - log_evidence_stabilized.unsqueeze(-1)).T

            importance_w_normalized = torch.exp(log_posteriors - log_prior.unsqueeze(-1))

        expected_log_likelihood = (importance_w_normalized.T * log_likelihood).sum(-1) / len(w)
        return expected_log_likelihood.sum()


def image_optimize(
        model: ImageDataModel,
        optimizer: torch.optim.Optimizer,
        data: List[torch.tensor],
        epochs: int,
        report_at: int,
        sample_size_z: int = 128,
        variational: bool=False) -> None:
    """
    Optimization routine for the DataModel, including sampling and backup of model parameters.
    :param model: DataModel instance
    :param optimizer: Optimizer instance
    :param data: List of shapes
    :param epochs: Number of epochs to optimize
    :param report_at: Interval status report and model backup
    :param sample_size_z: Number of samples from latent distribution z
    :return:
    """
    latent_z = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(model.z_dim),
        covariance_matrix=torch.eye(model.z_dim)
    )

    for epoch in range(epochs):
        indices = torch.randperm(len(data)*10)[:10]
        batch = torch.cat([data]*10, axis=0)[indices]
        z = latent_z.sample(torch.Size([sample_size_z])).to(model.device)

        optimizer.zero_grad()
        objective = -model.objective(batch, z, variational)
        objective.backward()
        optimizer.step()

        # report, backup model and optimizer states
        with torch.no_grad():
            if epoch % report_at == 0 or epoch == epochs - 1:
                print(epoch, "objective", -objective, model.log_std_.data)
                torch.save(model.state_dict(), "model/image_model_{}.pt".format(epoch))
                torch.save(optimizer.state_dict(), "model/image_optimizer_{}.pt".format(epoch))


def optimize(
        model: SetModel,
        optimizer: torch.optim.Optimizer,
        data: List[torch.tensor],
        epochs: int,
        report_at: int,
        sample_size_z: int = 128,
        sample_size_w: int = 128) -> None:
    """
    Optimization routine for the SetModel, including sampling and backup of model parameters.
    :param model: SetModel instance
    :param optimizer: Optimizer instance
    :param data: List of shapes
    :param epochs: Number of epochs to optimize
    :param report_at: Interval status report and model backup
    :param sample_size_z: Number of samples from latent distribution z
    :param sample_size_w: Number of samples from latent distribution w
    :return:
    """
    latent_z = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(model.shape_model.z_dim),
        covariance_matrix=torch.eye(model.shape_model.z_dim)
    )
    latent_w = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(model.w_dim),
        covariance_matrix=torch.eye(model.w_dim)
    )

    for epoch in range(epochs):
        indices = torch.randperm(len(data)*10)[:10]
        batch = []
        for index in indices:
            batch.append((data*10)[index])
        z = latent_z.sample(torch.Size([sample_size_z])).to(model.shape_model.device)
        w = latent_w.sample(torch.Size([sample_size_w])).to(model.device)

        optimizer.zero_grad()
        objective = -model.expected_log_likelihood(batch, w, z)
        objective.backward()
        optimizer.step()

        # hacky way to "anneal" variance of conditional distribution to the final value
        with torch.no_grad():
            if model.shape_model.log_std_.data < torch.log(torch.tensor(0.001)):
                model.shape_model.log_std_.copy_(torch.log(torch.tensor(0.001)))

        # report, backup model and optimizer states
        with torch.no_grad():
            if epoch % report_at == 0 or epoch == epochs - 1:
                print(epoch, "objective:", -objective, "| model variance:", model.shape_model.log_std_.data)
                torch.save(model.state_dict(), "model/model_{}.pt".format(epoch))
                torch.save(optimizer.state_dict(), "model/optimizer_{}.pt".format(epoch))


def get_grid(axis_scale: np.array) -> np.array:
    """
    Wrapper for generating a regular grid of input vectors for function evaluation.
    :param axis_scale: Axis positions for evaluated probability distribution.
    :return:
    """
    xx, yy = np.meshgrid(axis_scale, axis_scale)
    grid = np.stack([xx, yy], axis=-1)
    return grid


def calculate_set_model_posterior_distributions(
        model: SetModel,
        D: List[torch.tensor],
        resolution: int) -> (np.array, np.array, np.array):
    """
    Calculates the posterior distributions of w for a set of shapes D.
    :param model: The instance of the SetModel
    :param D: The set of shapes
    :param resolution: Resolution of the grid over w
    :return:
    """
    latent_z = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(model.shape_model.z_dim),
        covariance_matrix=torch.eye(model.shape_model.z_dim)
    )
    latent_w = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(model.w_dim),
        covariance_matrix=torch.eye(model.w_dim)
    )
    axis_scale = np.linspace(-2.0, 2.0, resolution)
    grid = torch.from_numpy(get_grid(axis_scale).reshape((-1, 2)))

    z = latent_z.sample(torch.Size([128])).to(model.shape_model.device)
    w = latent_w.sample(torch.Size([128])).to(model.device)

    with torch.no_grad():
        log_joint_probability = model.log_joint_probability(D, grid, z)
        log_evidence = torch.log(model.evidence(D, w, z))

    log_posteriors = log_joint_probability - log_evidence.unsqueeze(-1)
    posteriors = torch.exp(log_posteriors).reshape((len(D), resolution, resolution)).numpy()
    return axis_scale, grid, posteriors


def calculate_shape_model_posterior_distributions(
        model: ShapeModel,
        X: List[torch.tensor],
        w: torch.tensor,
        resolution: int) -> (np.array, np.array, np.array):
    """
    Calcualtes the posterior distribution of z for a set of points.
    :param model: An instance of SetModel
    :param X: The set of points
    :param w: A sample from the latent distribution w, which defines the shape
    :param resolution: Resolution of the grid over z
    :return:
    """
    latent_z = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(model.z_dim),
        covariance_matrix=torch.eye(model.z_dim)
    )

    axis_scale = np.linspace(-2.0, 2.0, resolution)
    grid = torch.from_numpy(get_grid(axis_scale).reshape((-1, 2)))

    z = latent_z.sample(torch.Size([128])).to(model.device)

    with torch.no_grad():
        log_joint_probability = model.log_joint_probability(X=X, z=grid, w=w)
        log_evidence = torch.log(model.evidence(X=X, z=z, w=w))

    log_posteriors = log_joint_probability - log_evidence.unsqueeze(-1)
    posteriors = torch.exp(log_posteriors).reshape((len(X), resolution, resolution)).numpy()
    return axis_scale, grid, posteriors


def calculate_conditional_distributions(
        model: SetModel,
        w: torch.tensor,
        resolution: int) -> (int, int, List[np.array]):
    """
    Calculates the conditional distributions of x for a set of w.
    :param model: An instance of the SetModel
    :param w: A set of w samples
    :param resolution: Resolution of the grid over x
    :return:
    """
    latent_z = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(model.shape_model.z_dim),
        covariance_matrix=torch.eye(model.shape_model.z_dim)
    )
    z = latent_z.sample(torch.Size([1024])).to(model.shape_model.device)
    axis_scale = np.linspace(0.0, 1.0, resolution)
    grid = torch.from_numpy(get_grid(axis_scale).reshape((-1, 2)))
    conditional_distributions = []
    for wi in w:
        with torch.no_grad():
            evidence = model.shape_model.evidence(
                X=grid,
                z=z,
                w=wi
            ).numpy().reshape((resolution, resolution))
        conditional_distributions.append(evidence)
    return axis_scale, grid, conditional_distributions
