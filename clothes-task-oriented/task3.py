import numpy as np

def log_likelihood_joint(points, pi, mu, sigma, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            ll(log-likelihood): NxK array, where ll(i, j) = log pi(j) + log NormalPDF(points_i | mu[j], sigma[j])
            
        Hint: Assume that the three dimensions of our multivariate gaussian are independent.  
              This allows you to write treat it as a product of univariate gaussians.
        """
        lls = np.zeros(shape = (points.shape[0],np.size(pi)))
        for j in range(np.size(pi)):
            lls[:,j] = np.log(pi[j] + 1e-12) 
            temp = points - mu[j,:]
            #cov = sigma[j,:]**(-1)
            x = 1
            for i in range(points.shape[1]):
                x *= 1/(np.sqrt(2 * np.pi) * np.sqrt(sigma[j, i]))
                y = -1/(2*(sigma[j, i]))
                x *= np.exp(y * (temp[:,i]**2))
            #(x-mu).transpose*inverse(covariance matrix)*(x-mu)
            #1xD*D*K
            #x = np.exp(-1/2 * np.sum(np.dot(temp, np.diag(cov)) * temp, axis = 1))

            lls[:,j] += np.log(x + 1e-12)
        return lls