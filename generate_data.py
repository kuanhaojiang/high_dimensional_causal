import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.linalg import lstsq
from itertools import permutations
np.set_printoptions(suppress=True)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_prime(x):
    return np.exp(-x) / (1 + np.exp(-x))**2  # Derivative of the sigmoid, used in optimization algorithms.

# Logistic regression model training
# X: Input feature matrix
# A: Labels (target values)
# This function trains a logistic regression model and returns the learned coefficients (beta_hat).
def compute_beta_hat(X, A, C=1000):
    # Logistic regression model with L2 regularization,
    # C: inverse of regularization parameter (high value of C reduces regularization).
    reg_model = LogisticRegression(penalty='l2', C=C, fit_intercept=False).fit(X, A)
    return reg_model.coef_.reshape(-1)

# Ordinary Least Squares (OLS) for linear regression on stratified data (groups 0 and 1 in A)
# X: Input feature matrix
# Y: Target values
# A: Grouping labels (0 or 1)
# This function fits separate linear regression models for each group in A and returns the
# estimated intercepts (alpha) and coefficients (beta) for both groups.
def compute_OLE(X, Y, A):
    # ni: Number of samples in the dataset.
    ni = X.shape[0]
    
    # X_tilde: Feature matrix with a bias term added to account for the intercept in the regression.
    X_tilde = np.concatenate([np.ones(ni).reshape(-1,1), X], axis=1)
    
    # Split the data into two groups based on the values in A (A==0 and A==1).
    X_tilde_0, X_tilde_1 = X_tilde[A == 0, :], X_tilde[A == 1, :]  # Feature matrices for groups 0 and 1
    Y_0, Y_1 = Y[A == 0], Y[A == 1]  # Target values for groups 0 and 1
    
    # Fit OLS regression models separately for both groups.
    # lstsq: Least squares solver that returns the optimal coefficients (intercept and slope).
    ole0, ole1 = lstsq(X_tilde_0, Y_0)[0], lstsq(X_tilde_1, Y_1)[0]
    
    # Extract the intercept (alpha) and coefficients (beta) for both groups.
    alpha_0_hat, beta_0_hat = ole0[0], ole0[1:]  # Intercept and coefficients for group 0.
    alpha_1_hat, beta_1_hat = ole1[0], ole1[1:]  # Intercept and coefficients for group 1.
    return alpha_0_hat, beta_0_hat, alpha_1_hat, beta_1_hat

def get_cf_ests(kappa, n, r1=1/3, r2=1/3, alpha1=2, alpha0=0, gamma=0.1,\
                  rho_12=0.2, sigma=1, num_rep=1000):
    # This function generates synthetic data and compute the cross-fitting estimator for ATE.
    # Parameters:
    # kappa: Ratio of the number of features (p) to the number of observations (n).
    # n: Total number of observations (samples).
    # r1, r2: Proportions of the data split for different groups.
    # alpha1, alpha0: Coefficients for treatment and control responses.
    # gamma: signal strength for propensity score model parameters.
    # rho_12: Correlation between the two sets of beta coefficients (beta0, beta1) \
    # for outcome regression models (treatment & control)
    # sigma: standard deviation of the noise.
    # num_rep: Number of repetition of experiments.

    # Define the number of features (p) based on kappa and n.
    p = int(n * kappa)
    
    # Calculate the number of samples in the first two groups and the remaining samples in the third group.
    n1, n2 = int(n * r1), int(n * r2)
    n3 = n - n1 - n2  # Remaining samples in group 3
    r1, r2, r3 = n1 / n, n2 / n, n3 / n  # Recompute proportions based on actual sample counts.

    # Generate a random beta vector for propensity score model.
    beta = np.random.normal(size=p)
    beta = beta / np.linalg.norm(beta) * gamma * np.sqrt(n)

    # Define the signal strength for outcome regression model parameters.
    kappa = p / n
    gamma0, gamma1 = 0.1 / np.sqrt(kappa), 0.1 / np.sqrt(kappa)

    # Covariance matrix for beta0 and beta1.
    cov_mat = [[1, rho_12], [rho_12, 1]]
    
    # Generate beta0 and beta1 from a multivariate normal distribution with the given covariance matrix.
    betas = np.random.multivariate_normal(np.zeros(2), cov_mat, size=p)
    beta0, beta1 = betas[:, 0], betas[:, 1]
    
    # Scale beta0 and beta1 to match the given signal strength.
    beta0 = beta0 / np.linalg.norm(beta0) * gamma0 * np.sqrt(p)
    beta1 = beta1 / np.linalg.norm(beta1) * gamma1 * np.sqrt(p)

    # Generate the synthetic data and compute cross-fitted estimates.
    np.random.seed()
    crossfit_ests = []
    for _ in range(num_rep):
        # Generate input feature matrix X, with values sampled from a normal distribution.
        X = np.random.normal(scale=1/np.sqrt(n), size=n * p).reshape([n, p])
        
        # Generate treatment assignment A using logistic regression (based on X and beta).
        A = np.random.binomial(n=1, p=sigmoid(X @ beta))
        
        # Split the data into three groups based on n1, n2, and n3.
        X1, X2, X3 = X[:n1, :], X[n1:n1+n2, :], X[n1+n2:, :]
        A1, A2, A3 = A[:n1], A[n1:n1+n2], A[n1+n2:]

        # Generate random noise to add to the outcome variables.
        epsilons = np.random.normal(scale=sigma, size=n)

        # Generate outcome variables (Y) for control and treatment groups.
        Y_control = alpha0 + X @ beta0 + epsilons
        Y_treat = alpha1 + X @ beta1 + epsilons
        Y = Y_control.copy()  # Initialize Y as control outcomes.
        Y[A == 1] = Y_treat[A == 1]  # Update Y for treatment group.

        # Split Y into corresponding groups.
        Y1, Y2, Y3 = Y[:n1], Y[n1:n1+n2], Y[n1+n2:]

        Xs = [X1, X2, X3]  # Feature matrices for the three groups.
        As = [A1, A2, A3]  # Treatment assignments for the three groups.
        Ys = [Y1, Y2, Y3]  # Outcome variables for the three groups.

        # Precompute cross-fitted estimates.
        pre_crossfit_ests = []
        for s in permutations([0, 1, 2]):  # Loop over all permutations of the group indices.
            a, b, c = s[0], s[1], s[2]

            # Estimate the propensity score model coefficients from the first group.
            beta_hat = compute_beta_hat(Xs[a], As[a])
            
            # Estimate the outcome regression model coefficients from the second group.
            alpha_0_hat, beta_0_hat, alpha_1_hat, beta_1_hat = compute_OLE(Xs[b], Ys[b], As[b])
            
            # Compute the estimated propensity score for the third group.
            sigmoid_x_beta_hat = sigmoid(Xs[c] @ beta_hat)

            # Estimate the treatment effect (delta_hat_1) for the third group.
            t1 = np.mean(np.divide(np.multiply(As[c], Ys[c]), sigmoid_x_beta_hat))
            t2 = np.divide(As[c] - sigmoid_x_beta_hat, sigmoid_x_beta_hat)
            t3 = alpha_1_hat + Xs[c] @ beta_1_hat
            delta_hat_1 = t1 - np.mean(np.multiply(t2, t3))

            # Estimate the control effect (delta_hat_2) for the third group.
            t1 = np.mean(np.divide(np.multiply(1 - As[c], Ys[c]), 1 - sigmoid_x_beta_hat))
            t2 = np.divide(As[c] - sigmoid_x_beta_hat, 1 - sigmoid_x_beta_hat)
            t3 = alpha_0_hat + Xs[c] @ beta_0_hat
            delta_hat_2 = t1 + np.mean(np.multiply(t2, t3))

            # Store the estiamted difference between the treatment and control effects.
            pre_crossfit_ests.append(delta_hat_1 - delta_hat_2)
        
        # Compute the average of the pre-crossfitted estimates to obtain the
        # crossfitted estimate
        crossfit_ests.append(np.mean(pre_crossfit_ests))

    return crossfit_ests



