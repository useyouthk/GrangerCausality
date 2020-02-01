import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str

class MultipleGrangerCausality():
    def __init__(self, X):
        self.X = X
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]

    def decide_degree_best(self):
        # make a VAR model
        model = VAR(self.X)
        model.select_order(15)
        
        # determine the optimal VAR model order using AIC
        print(model.select_order(15))
        results = model.fit(maxlags=15, ic='aic')
        print(results.summary())
    
    def calc_causality(self, target_index, causal_index, alpha, degree_best): #Significance level alpha is generally 0.05
        self.target_index = target_index
        self.causal_index = causal_index
        self.degree_best = degree_best
        
        # output y consisting of target_index
        self.y_new = self.X[self.degree_best:, target_index]

        # make X_full to create full_model
        self.X_full = np.empty((self.n_samples - self.degree_best, 0))
        for delay_k in range(self.degree_best):
            self.X_full = np.hstack((self.X_full, self.X[delay_k : self.n_samples - self.degree_best + delay_k]))

        # input without causal_index
        X_reduced = np.delete(self.X, causal_index, 1)

        self.X_reduced_full = np.empty((self.n_samples - self.degree_best, 0))
        for delay_k in range(self.degree_best):
            self.X_reduced_full = np.hstack((self.X_reduced_full, X_reduced[delay_k : self.n_samples - self.degree_best + delay_k]))

        # make a full_model
        reg = linear_model.LinearRegression()
        reg_fit = reg.fit(self.X_full, self.y_new)

        y_hat_full = reg.predict(self.X_full)
        self.y_hat_full = y_hat_full
        e_full = self.y_new - y_hat_full

        # make a reduced_model
        reg_reduced = linear_model.LinearRegression()
        reg_fit_reduced = reg_reduced.fit(self.X_reduced_full, self.y_new)
        y_hat_reduced = reg_reduced.predict(self.X_reduced_full)
        e_reduced_full = self.y_new - y_hat_reduced

        # F test
        RSS_1 = np.sum(e_full**2) # residual sum of full model
        RSS_0 = np.sum(e_reduced_full**2) # residual sum of reduced model
        self.f_value = ((RSS_0 - RSS_1)/self.degree_best) / (RSS_1 / (self.n_samples - self.n_features * self.degree_best - 1))

        # Granger causality determination
        gamma_F = self.degree_best * self.f_value
        p_value = stats.distributions.chi2.sf(gamma_F,self.degree_best)
        print('gamma_F = {}'.format(gamma_F))
        print('p_value ={}'.format(p_value))

        if alpha > p_value:
            print('Granger causality from {} to {} exists.'.format(causal_index, target_index))
        else:
            print('Granger causality from {} to {} does not exists.'.format(causal_index, target_index))

