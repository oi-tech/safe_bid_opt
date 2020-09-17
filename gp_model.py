import gpflow
import logging
import numpy as np
import plot
import preprocess
import subcampaign as sc
import tensorflow as tf
import tensorflow_probability as tfp
from config import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from gpflow.utilities import tabulate_module_summary, set_trainable # print_summary
from gpflow.ci_utils import ci_niter

# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

logger = logging.getLogger(__name__)


class GPModel:

    def __init__(self, min_bid=0.0, max_bid=MAX_BID,
                 min_cost=0.0, max_cost=MAX_EXP, FIX_HP=False, OPT=True, MCMC=False):
        self.min_bid = min_bid
        self.max_bid = max_bid
        self.min_cost = min_cost
        self.max_cost = max_cost

        self.FIX_HP = FIX_HP
        self.OPT = OPT
        self.MCMC = MCMC

        # self.__kernel_cost = gpflow.kernels.SquaredExponential()
        # gpflow.kernels.Constant() * gpflow.kernels.SquaredExponential()
        # self.__kernel_rev = gpflow.kernels.SquaredExponential()
        # gpflow.kernels.Constant() * gpflow.kernels.SquaredExponential()
        # kernel_cost = gpflow.kernels.Constant() * gpflow.kernels.Matern52()  # lengthscales=2.0)  # gpflow.kernels.SquaredExponential()
        # kernel_rev = gpflow.kernels.Constant() * gpflow.kernels.Matern52()  # lengthscales=2.0)  # gpflow.kernels.SquaredExponential()
        kernel_cost = gpflow.kernels.SquaredExponential() #  * gpflow.kernels.Constant()  # * gpflow.kernels.SquaredExponential()
        kernel_rev = gpflow.kernels.SquaredExponential()  # * gpflow.kernels.Constant()  # * gpflow.kernels.SquaredExponential()

        # self.__kernel_cost = kernel_cost
        # self.__kernel_rev = kernel_rev
        A = np.zeros((1, 1))  # np.array([[0.], [0.]]).T
        b = np.zeros(1)  # 0.0  # np.array([[0.], [0.]]).T
        self.__mean_cost = None  #gpflow.mean_functions.Linear(A, b)  # None  # 0.0
        self.__mean_rev =  None  #gpflow.mean_functions.Linear(A, b)  # None  # 0.0

        self.__input_scaler = preprocess.Preprocess(with_scaler=True, with_mean=False, with_std=False)
        # self.__input_scaler = preprocess.Preprocess()  # scale_max=MAX_BID)  # ,
                                                    # with_mean=False,
                                                    # with_std=False)
        # self.__output_scaler = preprocess.Preprocess(with_scaler=False)  # ,
                                                     # with_mean=True,
                                                     # with_std=False)
        self.__output_cost_scaler = preprocess.Preprocess(scale_min=0.0, scale_max=100.0, with_scaler=True,  with_mean=True, with_std=False)
        self.__output_rev_scaler = preprocess.Preprocess(scale_min=0.0, scale_max=700.0, with_scaler=True,  with_mean=True, with_std=False)

        # data already knew
        data_X = np.array(0.0).reshape(-1, 1)  #  self.__input_scaler.fit(np.array(0.0).reshape(-1, 1))
        data_Y = np.array(0.0).reshape(-1, 1)  # self.__output_scaler.fit(np.array(0.0).reshape(-1, 1))
        self.last_obs = data_X[-1]
        data_X = self.__input_scaler.fit(np.array(0.0).reshape(-1, 1))
        data_Y_rev = self.__output_cost_scaler.fit(np.array(0.0).reshape(-1, 1))
        data_Y_cost = self.__output_rev_scaler.fit(np.array(0.0).reshape(-1, 1))



        data = (data_X, data_Y)
        data_rev = (data_X, data_Y_rev)
        data_cost = (data_X, data_Y_cost)


        self.model_cost = gpflow.models.GPR(data=data_cost,
                                            kernel=kernel_cost,  # self.__kernel_cost,
                                            mean_function=self.__mean_cost,
                                            noise_variance=1.0)
        self.model_rev = gpflow.models.GPR(data=data_rev,
                                           kernel=kernel_rev,  # self.__kernel_rev,
                                           mean_function=self.__mean_rev,
                                           noise_variance=1.0)

        
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(f'summary cost gp model after first initialization')
            logger.warning(f'{tabulate_module_summary(self.model_cost)}')
            logger.warning(f'summary rev gp model after first initialization')
            logger.warning(f'({tabulate_module_summary(self.model_rev)})')

        if self.FIX_HP:
            self.OPT = False
            self.MCMC = False
            self._set_hp()
        if self.OPT:
            self._optimize()
        if self.MCMC:
            self.hmcmc(self.model_cost)

            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f'summary cost gp model after hmcmc')
                logger.warning(f'{tabulate_module_summary(self.model_cost)}')
                logger.warning(f'after hmcmc self.model_cost.data:\n'
                            f'{self.model_cost.data}')
            self.hmcmc(self.model_rev)

            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f'summary rev gp model after hmcmc')
                logger.warning(f'{tabulate_module_summary(self.model_rev)}')
                logger.warning(f'after hmcmc self.model_rev.data:\n'
                            f'{self.model_rev.data}')

    def revenue(self, bid, return_var=False, sample_y=False):
        if logger.isEnabledFor(logging.INFO):
            logger.info('entering revenue method')

        bid = self.__input_scaler.transform(bid)

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'bid shape: {bid.shape}')

        if sample_y:
            if logger.isEnabledFor(logging.INFO):
                logger.info('sampling from revenue gp')
            return self.__output_rev_scaler.transform_back(
                            self.model_rev.predict_f_samples(bid)
                            )

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'predict revenue gp with {return_var}')

        mean, var = self.model_rev.predict_f(bid)

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'revenue mean before scaling_back\n{mean}')
        mean = self.__output_rev_scaler.transform_back(mean)

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'revenue mean after scaling_back\n{mean}')

        if return_var:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'revenue variance before scaling_back\n{var}')
            var = self.__output_rev_scaler.transform_var_back(var)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'revenue variance after scaling_back\n{var}')
                logger.info(f'revenue var\n{var}')
            return (mean, var)

        return mean


    def cost(self, bid, return_var=False, sample_y=False):
        if logger.isEnabledFor(logging.INFO):
            logger.info('entering cost method')

        bid = self.__input_scaler.transform(bid)

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'bid shape: {bid.shape}')

        if sample_y:
            if logger.isEnabledFor(logging.INFO):
                logger.info('sampling from cost gp')
            return self.__output_cost_scaler.transform_back(
                            self.model_cost.predict_f_samples(bid)
                            )

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'predict cost gp with {return_var}')

        mean, var = self.model_cost.predict_f(bid)

        mean = self.__output_cost_scaler.transform_back(mean)

        if return_var:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'cost variance before scaling_back\n{var}')
            var = self.__output_cost_scaler.transform_var_back(var)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'cost variance after scaling_back\n{var}')
                logger.info(f'cost var\n{var}')
            return (mean, var)

        return mean


    def _set_hp(self, hp):
        # fix parameters
        self.hp = hp
        self.model_cost.kernel.variance.assign(hp['cost_variance'])  # 0.356734)  # 0.031256)
        self.model_cost.kernel.lengthscales.assign(hp['cost_lengthscales'])  #0.342836)  # 0.0956548)
        self.model_cost.likelihood.variance.assign(hp['cost_likelihood'])  #0.00124288)  #0.000756103)

        self.model_rev.kernel.variance.assign(hp['rev_variance'])  #0.316219)  # 0.195807)
        self.model_rev.kernel.lengthscales.assign(hp['rev_lengthscales'])  #0.127569)  # 0.0709042)
        self.model_rev.likelihood.variance.assign(hp['rev_likelihood'])  #0.00234378)  #0.00118543)

        set_trainable(self.model_rev.kernel.variance, False)
        set_trainable(self.model_cost.kernel.variance, False)
        set_trainable(self.model_rev.kernel.lengthscales, False)
        set_trainable(self.model_cost.kernel.lengthscales, False)
        set_trainable(self.model_rev.likelihood.variance, False)
        set_trainable(self.model_cost.likelihood.variance, False)

        self.OPT = False
        self.MCMC = False
        self.FIX_HP = True


    def _optimize(self):
        if logger.isEnabledFor(logging.INFO):
            logger.info('entering _optimize method')
            logger.info(f'trainable variables {self.model_cost.trainable_variables}')

        adam_learning_rate = 0.01
        iterations = ci_niter(1000)
        opt = tf.optimizers.Adam(adam_learning_rate)

        opt_rev = tf.optimizers.Adam(adam_learning_rate)

        @tf.function
        def cost_optimization_step():
            opt.minimize(self.model_cost.training_loss, self.model_cost.trainable_variables)
        @tf.function
        def rev_optimization_step():
            opt_rev.minimize(self.model_rev.training_loss, self.model_rev.trainable_variables)

        for i in range(iterations):
            opt_logs = cost_optimization_step()


        # for i in range(iterations):
            opt_logs_rev = rev_optimization_step()

        # opt = gpflow.optimizers.Scipy()

        # opt_rev = gpflow.optimizers.Scipy()
        # opt_logs = opt.minimize(self.model_cost.training_loss,
        #                         self.model_cost.trainable_variables,
        #                         # method='COBYLA',  # -BFGS-B',
        #                         method='BFGS',  # L-BFGS-B',  # 'SLSQP',
        #                         options=dict(maxiter=ci_niter(2000)))  #dict(maxiter=500))
        # opt_logs_rev = opt_rev.minimize(self.model_rev.training_loss,
        #                                 self.model_rev.trainable_variables,
        #                                 # method='COBYLA',  # 'L-BFGS-B',
        #                                 method='BFGS',  # L-BFGS-B',  # 'SLSQP',
        #                                 options=dict(maxiter=ci_niter(2000)))  # dict(maxiter=500))

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'opt_logs:\n{opt_logs}')
            logger.info(f'opt_logs_rev:\n{opt_logs_rev}')
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(f'summary cost gp model')
            logger.warning(f'{tabulate_module_summary(self.model_cost)}')
            logger.warning(f'summary rev gp model')
            logger.warning(f'({tabulate_module_summary(self.model_rev)})')

    # def _add_obs(self, gp, X_new, Y_new):
    #     logger.info('entering _add_obs')
    #     X, Y = gp.data
    #     X = self.__input_scaler.transform_back(X)
    #     Y = self.__output_scaler.transform_back(Y)

    #     X = np.append(X, X_new).reshape(-1, 1)
    #     Y = np.append(Y, Y_new).reshape(-1, 1)

    #     logger.info(f'before data:\n{gp.data}')

    #     X = self.__input_scaler.fit(X)
    #     Y = self.__output_scaler.fit(Y)

    #     gp.data = (X, Y)

    #     logger.info(f'after data:\n{gp.data}')

    def _add_obs(self, gp, output_scaler, X_new, Y_new):
        if logger.isEnabledFor(logging.INFO):
            logger.info('entering _add_obs')
        X, Y = gp.data
        X = self.__input_scaler.transform_back(X)
        # Y = self.__output_scaler.transform_back(Y)

        Y = output_scaler.transform_back(Y)

        X = np.append(X, X_new).reshape(-1, 1)
        Y = np.append(Y, Y_new).reshape(-1, 1)

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'before data:\n{gp.data}')

        #self.last_obs = X[-1]

        X = self.__input_scaler.fit(X)
        Y = output_scaler.fit(Y)

        gp.data = (X, Y)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'after data:\n{gp.data}')

    def _add_obs_del_old(self, gp, X_new, Y_new):
        if logger.isEnabledFor(logging.INFO):
            logger.info('entering _add_obs')
        # X, Y = gp.data
        # X = self.__input_scaler.transform_back(X)
        # Y = self.__output_scaler.transform_back(Y)
        # X = np.append(X, X_new).reshape(-1, 1)
        # Y = np.append(Y, Y_new).reshape(-1, 1)

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'before data:\n{gp.data}')

        X = self.__input_scaler.fit(X_new.reshape(-1, 1))
        Y = self.__output_scaler.fit(Y_new.reshape(-1, 1))

        gp.data = (X, Y)

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'after data:\n{gp.data}')

    def get_last_bid(self):
        if logger.isEnabledFor(logging.INFO):
            logger.info('entering get last bid')
        #X, _ = self.model_cost.data
        #X = self.__input_scaler.transform_back(X)
        # Y = self.__output_scaler.transform_back(Y)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'last X-bid value: {self.last_obs}')

        #return X[-1]
        return self.last_obs

    def update(self, X_bid, Y_cost, Y_rev):
        if logger.isEnabledFor(logging.INFO):
            logger.info('entering update method')
        self.last_obs = np.array([X_bid])[-1]

        kernel_cost = gpflow.kernels.SquaredExponential() # + gpflow.kernels.Linear() # gpflow.kernels.Matern32()  # + gpflow.kernels.Linear()  # gpflow.kernels.SquaredExponential(lengthscales=0.5) * gpflow.kernels.Linear() # lengthscales=2.0)  # gpflow.kernels.Constant() * gpflow.kernels.Matern52(lengthscales=2.0)  # gpflow.kernels.SquaredExponential(lengthscales=2.0)  # * gpflow.kernels.Linear()  # gpflow.kernels.Constant() * gpflow.kernels.Matern32(lengthscales=2.0) * gpflow.kernels.Linear()  # gpflow.kernels.RationalQuadratic()
        # kernel_cost = gpflow.kernels.Constant() + gpflow.kernels.Matern32(lengthscales=2.0) * gpflow.kernels.Linear()  # gpflow.kernels.RationalQuadratic()
        # gpflow.kernels.Constant() * gpflow.kernels.Matern52()
        # gpflow.kernels.Constant() * gpflow.kernels.SquaredExponential()
        kernel_rev = gpflow.kernels.SquaredExponential()  # + gpflow.kernels.Linear() # gpflow.kernels.Matern32()  # + gpflow.kernels.Linear()  # gpflow.kernels.SquaredExponential(lengthscales=0.5) * gpflow.kernels.Linear() # gpflow.kernels.Constant() * gpflow.kernels.Matern52(lengthscales=2.0)  # gpflow.kernels.SquaredExponential(lengthscales=2.0)  # * gpflow.kernels.Linear()   # * gpflow.kernels.Matern32(lengthscales=2.0) * gpflow.kernels.Linear()  # gpflow.kernels.RationalQuadratic()
        # kernel_rev = gpflow.kernels.Constant() + gpflow.kernels.Matern32(lengthscales=2.0) * gpflow.kernels.Linear()  # gpflow.kernels.RationalQuadratic()
        # gpflow.kernels.Constant() * gpflow.kernels.SquaredExponential()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'call add-obs with X_bid:\n{X_bid} and Y_cost:\n{Y_cost}')
        self._add_obs(self.model_cost, self.__output_cost_scaler, X_bid, Y_cost)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'call add-obs with X_bid:\n{X_bid} and Y_rev:\n{Y_rev}')
        self._add_obs(self.model_rev, self.__output_rev_scaler, X_bid, Y_rev)

        # self._add_obs_del_old(self.model_cost, X_bid, Y_cost)
        # self._add_obs_del_old(self.model_rev, X_bid, Y_rev)

        self.model_cost = gpflow.models.GPR(data=self.model_cost.data,
                                            kernel=kernel_cost,  # self.model_cost.kernel,
                                            mean_function=self.__mean_cost,  # None,  # self.model_cost.mean_function,
                                            noise_variance=1.0)
        self.model_rev = gpflow.models.GPR(data=self.model_rev.data,
                                           kernel=kernel_rev,  # self.model_rev.kernel,
                                           mean_function=self.__mean_rev,  # None,  # self.model_rev.mean_function,
                                           noise_variance=1.0)

        if logger.isEnabledFor(logging.WARNING):
            logger.warning(f'after update self.model_rev.data:\n'
                    f'{self.model_rev.data}')
            logger.warning(f'after update model_cost.data:\n{self.model_cost.data}')


        # # fix parameters
        # self.model_cost.kernel.variance.assign(0.356734)  # 0.031256)
        # self.model_cost.kernel.lengthscales.assign(0.342836)  # 0.0956548)
        # self.model_cost.likelihood.variance.assign(0.00124288)  #0.000756103)

        # self.model_rev.kernel.variance.assign(0.316219)  # 0.195807)
        # self.model_rev.kernel.lengthscales.assign(0.127569)  # 0.0709042)
        # self.model_rev.likelihood.variance.assign(0.00234378)  #0.00118543)

        # set_trainable(self.model_rev.kernel.variance, False)
        # set_trainable(self.model_cost.kernel.variance, False)
        # set_trainable(self.model_rev.kernel.lengthscales, False)
        # set_trainable(self.model_cost.kernel.lengthscales, False)
        # set_trainable(self.model_rev.likelihood.variance, False)
        # set_trainable(self.model_cost.likelihood.variance, False)

        # self.OPT = False
        if self.FIX_HP:
            self._set_hp(self.hp)
            return

        # priors
        self.model_cost.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
        self.model_rev.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
        self.model_cost.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
        self.model_rev.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
        self.model_cost.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
        self.model_rev.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))

        if self.__mean_cost:
            self.model_cost.mean_function.A.prior = tfd.Normal(f64(0.0), f64(1.))
            self.model_cost.mean_function.b.prior = tfd.Normal(f64(0.0), f64(1.))
        if self.__mean_rev:
            self.model_rev.mean_function.A.prior = tfd.Normal(f64(0.0), f64(1.))
            self.model_rev.mean_function.b.prior = tfd.Normal(f64(0.0), f64(1.))
        
        self.model_cost.kernel.lengthscales.prior = tfd.LogNormal(loc=0., scale=np.float64(1.))
        self.model_rev.kernel.lengthscales.prior = tfd.LogNormal(loc=0., scale=np.float64(1.))
        self.model_cost.kernel.variance.prior = tfd.LogNormal(loc=0., scale=np.float64(1.))
        self.model_rev.kernel.variance.prior = tfd.LogNormal(loc=0., scale=np.float64(1.))
        self.model_cost.likelihood.variance.prior = tfd.LogNormal(loc=0., scale=np.float64(1.))
        self.model_rev.likelihood.variance.prior = tfd.LogNormal(loc=0., scale=np.float64(1.))

        if self.OPT:
            self._optimize()

        if self.MCMC:
            self.hmcmc(self.model_cost)
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f'summary cost gp model after hmcmc')
                logger.warning(f'{tabulate_module_summary(self.model_cost)}')
                logger.warning(f'after hmcmc self.model_cost.data:\n'
                            f'{self.model_cost.data}')
            self.hmcmc(self.model_rev)
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f'summary rev gp model after hmcmc')
                logger.warning(f'{tabulate_module_summary(self.model_rev)}')
                logger.warning(f'after hmcmc self.model_rev.data:\n'
                            f'{self.model_rev.data}')

    def hmcmc(self, model):

        if logger.isEnabledFor(logging.INFO):
            logger.info('here in the hmcmc method')
        #we add priors to the hyperparameters.

        # tfp.distributions dtype is inferred from parameters - so convert to 64-bit
        model.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
        model.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
        model.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
        #model.mean_function.A.prior = tfd.Normal(f64(0.0), f64(10.0))
        #model.mean_function.b.prior = tfd.Normal(f64(0.0), f64(10.0))


        num_burnin_steps = ci_niter(500)
        num_samples = ci_niter(1000)

        # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
        hmc_helper = gpflow.optimizers.SamplingHelper(
            model.log_posterior_density, model.trainable_parameters
        )

        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
        )
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
        )


        @tf.function
        def run_chain_fn():
            return tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_burnin_steps,
                current_state=hmc_helper.current_state,
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            )


        samples, traces = run_chain_fn()
        #parameter_samples = hmc_helper.convert_to_constrained_values(samples)

        #param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(gp.items()}


class TFP_GPModel:
    def __init__(self, min_bid=0.0, max_bid=MAX_BID,
                 min_cost=0.0, max_cost=MAX_EXP):
        self.min_bid = min_bid
        self.max_bid = max_bid
        self.min_cost = min_cost
        self.max_cost = max_cost

        kernel_cost = gpflow.kernels.SquaredExponential() #  * gpflow.kernels.Constant()  # * gpflow.kernels.SquaredExponential()
        kernel_rev = gpflow.kernels.SquaredExponential()  # * gpflow.kernels.Constant()  # * gpflow.kernels.SquaredExponential()

        self.__mean_cost = None  # gpflow.mean_functions.Linear()  # None  # 0.0
        self.__mean_rev = None   # gpflow.mean_functions.Linear()  # None  # 0.0

        self.__input_scaler = preprocess.Preprocess(with_scaler=True, with_mean=False, with_std=False)
        self.__output_cost_scaler = preprocess.Preprocess(scale_min=0.0, scale_max=50.0, with_scaler=True,  with_mean=True, with_std=False)
        self.__output_rev_scaler = preprocess.Preprocess(scale_min=0.0, scale_max=50.0, with_scaler=True,  with_mean=True, with_std=False)

        # data already knew
        self.X = np.array(0.0).reshape(-1, 1)  #  self.__input_scaler.fit(np.array(0.0).reshape(-1, 1))
        self.Y_cost = np.array(0.0).reshape(-1, 1)  # self.__output_scaler.fit(np.array(0.0).reshape(-1, 1))
        self.Y_rev = np.array(0.0).reshape(-1, 1)  # self.__output_scaler.fit(np.array(0.0).reshape(-1, 1))
        self.transformed_X = self.__input_scaler.fit(np.array(0.0).reshape(-1, 1))
        self.transformed_Y_cost = self.__output_cost_scaler.fit(np.array(0.0).reshape(-1,))
        self.transformed_Y_rev = self.__output_rev_scaler.fit(np.array(0.0).reshape(-1,))

        self._optimize()

    # Use `tf.function` to trace the loss for more efficient evaluation.
    @tf.function(autograph=False, experimental_compile=False)
    def __target_log_prob(self, gp_joint_model, amplitude, length_scale, observation_noise_variance, observations_):
      return gp_joint_model.log_prob({
          'amplitude': amplitude,
          'length_scale': length_scale,
          'observation_noise_variance': observation_noise_variance,
          'observations': observations_
      })

    def __trainable_variables(self):

        # Create the trainable model parameters, which we'll subsequently optimize.
        # Note that we constrain them to be strictly positive.

        constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

        amplitude_var = tfp.util.TransformedVariable(
            initial_value=1.,
            bijector=constrain_positive,
            name='amplitude',
            dtype=np.float64)

        length_scale_var = tfp.util.TransformedVariable(
            initial_value=1.,
            bijector=constrain_positive,
            name='length_scale',
            dtype=np.float64)

        observation_noise_variance_var = tfp.util.TransformedVariable(
            initial_value=1.,
            bijector=constrain_positive,
            name='observation_noise_variance_var',
            dtype=np.float64)

        # trainable_variables = [v.trainable_variables[0] for v in
        return [amplitude_var, length_scale_var,
                observation_noise_variance_var]

    def _optimize(self):

        def _build_gp_(amplitude, length_scale, observation_noise_variance):

          """Defines the conditional dist. of GP outputs, given kernel parameters."""

          # Create the covariance kernel, which will be shared between the prior (which we
          # use for maximum likelihood training) and the posterior (which we use for
          # posterior predictive sampling)
          kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

          # Create the GP prior distribution, which we will use to train the model
          # parameters.
          return tfd.GaussianProcess(
              kernel=kernel,
              index_points=self.transformed_X.reshape(-1, 1),  # observation_index_points_,
              observation_noise_variance=observation_noise_variance)
        #print(_build_gp_

        self.gp_cost_joint_model = tfd.JointDistributionNamed({
            'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'observations': _build_gp_,
            })

        self.gp_rev_joint_model = tfd.JointDistributionNamed({
            'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'observations': _build_gp_,
            })

        trainable_rev_variables = self.__trainable_variables()
        trainable_cost_variables = self.__trainable_variables()


        # Now we optimize the model parameters.
        num_iters = 1000
        optimizer = tf.optimizers.Adam(learning_rate=.01)

        # # avoid to use gradient tape and use the minimize method instead
        #loss = -target_log_prob(amplitude_var, length_scale_var,
        #                             observation_noise_variance_var)

        #for i in range(num_iters):
        #    optimizer.minimize(loss, trainable_variables)

        # Store the likelihood values during training, so we can plot the progress
        trainable_variables = [v.trainable_variables[0] for v in trainable_cost_variables]
        # lls_ = np.zeros(num_iters, np.float64)
        for i in range(num_iters):
           with tf.GradientTape() as tape:
             loss = -self.__target_log_prob(self.gp_cost_joint_model, *trainable_cost_variables, self.transformed_Y_cost)
           grads = tape.gradient(loss, trainable_variables)
           optimizer.apply_gradients(zip(grads, trainable_variables))

        predictive_index_points_ = np.linspace(0,MAX_BID/2, N_BID, dtype=np.float64)
        # Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
        predictive_index_points_ = predictive_index_points_[..., np.newaxis]


        #optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
        optimized_kernel = tfk.ExponentiatedQuadratic(trainable_cost_variables[0], trainable_cost_variables[1])
        self.cost_gprm = tfd.GaussianProcessRegressionModel(
            kernel=optimized_kernel,
            index_points=predictive_index_points_,
            observation_index_points=self.transformed_X,
            observations=self.transformed_Y_cost,
            observation_noise_variance=trainable_cost_variables[2],
            predictive_noise_variance=0.)

        # Now we optimize the model parameters.
        num_iters = 1000
        optimizer = tf.optimizers.Adam(learning_rate=.01)

        # # avoid to use gradient tape and use the minimize method instead
        #loss = -target_log_prob(amplitude_var, length_scale_var,
        #                             observation_noise_variance_var)

        #for i in range(num_iters):
        #    optimizer.minimize(loss, trainable_variables)

        # Store the likelihood values during training, so we can plot the progress
        trainable_variables = [v.trainable_variables[0] for v in trainable_rev_variables]
        # lls_ = np.zeros(num_iters, np.float64)
        for i in range(num_iters):
           with tf.GradientTape() as tape:
             loss = -self.__target_log_prob(self.gp_rev_joint_model, *trainable_rev_variables, self.transformed_Y_rev)
           grads = tape.gradient(loss, trainable_variables)
           optimizer.apply_gradients(zip(grads, trainable_variables))

        predictive_index_points_ = np.linspace(0,MAX_BID/2, N_BID, dtype=np.float64)
        # Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
        predictive_index_points_ = predictive_index_points_[..., np.newaxis]


        # optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
        optimized_kernel = tfk.ExponentiatedQuadratic(trainable_rev_variables[0], trainable_rev_variables[1])
        self.rev_gprm = tfd.GaussianProcessRegressionModel(
            kernel=optimized_kernel,
            index_points=predictive_index_points_,
            observation_index_points=self.transformed_X,
            observations=self.transformed_Y_rev,
            observation_noise_variance=trainable_rev_variables[2],
            predictive_noise_variance=0.)

    def update(self, X_bid, Y_cost, Y_rev):
        if logger.isEnabledFor(logging.INFO):
            logger.info('entering update method')
        # X = self.__input_scaler.transform_back(X)
        # Y = self.__output_scaler.transform_back(Y)

        self.X = np.append(self.X, X_bid).reshape(-1, 1)
        self.Y_cost = np.append(self.Y_cost, Y_cost).reshape(-1, 1)
        self.Y_rev = np.append(self.Y_rev, Y_rev).reshape(-1, 1)

        self.transformed_X = self.__input_scaler.fit(self.X).reshape(-1, 1)
        self.transformed_Y_cost = self.__output_cost_scaler.fit(self.Y_cost).reshape(-1,)
        self.transformed_Y_rev = self.__output_rev_scaler.fit(self.Y_rev).reshape(-1,)

        self._optimize()

    def revenue(self, bid, return_var=False, sample_y=False):
        if logger.isEnabledFor(logging.INFO):
            logger.info('entering revenue method')

        bid = self.__input_scaler.transform(bid)

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'bid shape: {bid.shape}')

        if sample_y:
            if logger.isEnabledFor(logging.INFO):
                logger.info('sampling from revenue gp')
            return self.__output_rev_scaler.transform_back(
                            self.rev_gprm.sample()  # sample_shape=bid.shape)
                            )

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'predict revenue gp with {return_var}')

        mean = self.rev_gprm.mean()
        var = self.rev_gprm.variance()

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'mean shape: {mean.shape}')
            logger.info(f'var shape: {var.shape}')

        return (self.__output_rev_scaler.transform_back(mean) if not return_var
                else (self.__output_rev_scaler.transform_back(mean),
                self.__output_rev_scaler.transform_var_back(var)))

    def cost(self, bid, return_var=False, sample_y=False):
        if logger.isEnabledFor(logging.INFO):
            logger.info('entering cost method')

        bid = self.__input_scaler.transform(bid)

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'bid shape: {bid.shape}')

        if sample_y:
            if logger.isEnabledFor(logging.INFO):
                logger.info('sampling from cost gp')
            return self.__output_cost_scaler.transform_back(
                            self.cost_gprm.sample()  # sample_shape=bid.shape)
                            )

        if logger.isEnabledFor(logging.INFO):
            logger.info(f'predict cost gp with {return_var}')

        mean = self.cost_gprm.mean()
        var = self.cost_gprm.variance()
        return (self.__output_cost_scaler.transform_back(mean) if not return_var
                else (self.__output_cost_scaler.transform_back(mean),
                self.__output_cost_scaler.transform_var_back(var)))


if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.ERROR,
                        # format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        # datefmt='%m-%d %H:%M',
                        filename='/tmp/gpmodijfilapp.log',
                        filemode='w')
    import time
    bid = np.linspace(0, MAX_BID, N_BID)
    gp = GPModel()  # TFP_GPModel()
    plot.plot(bid, gp.revenue(bid))
    X = np.random.rand(1, 30)*MAX_BID
    #Y_cost = np.random.rand(10, 1)*MAX_BID
    #Y_rev = np.random.rand(1, 1)*MAX_BID
    
    c = sc.Subcampaign(59, 1.3, 63, 1.0)
    Y_cost = c.cost(X)
    Y_rev = c.revenue(X)
    print('oi', X, Y_cost, Y_rev)
    start = time.time()
    gp.update(X, Y_cost, Y_rev)
    end = time.time()
    print('elapsed time:', end - start)
    plot.plot(bid, c.revenue(bid, noise=False))

    print('gp revenue after:\n', gp.revenue(bid))
    plot.plot(bid, gp.revenue(bid))
    mean, var = gp.revenue(bid, return_var=True)
    logging.info(f'mean\n{mean}')
    logging.info(f'variance\n{var}')
    plot.plot_gp(bid, mean, var, c.revenue(bid, noise=False), X, Y_rev, Show=True)
    print('real function revenue:\n', c.revenue(bid, noise=False))
    print('gp cost after\n', gp.cost(bid))
    print('real function cost:\n', c.cost(bid, noise=False))
    print('sample gp cost\n', gp.cost(bid, sample_y=True))
    plot.plot(bid, gp.revenue(bid, sample_y=True))
