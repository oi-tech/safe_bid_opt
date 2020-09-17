from config import *
import gp_model as gp
import logging
import numpy as np
import subcampaign as sc
from optimize import bid_maxrevenue_given_cost, bid_max_lb_revenue_given_cost, optimize, safe_optimize_rev, safe_optimize_roi, safe_optimize_delta, safe_optimize, safe_opt, bid_max_ub_revenue_given_lb_rev_and_cost# , safe_optimize_roi_2

logger = logging.getLogger(__name__)


class Agent:

    def __init__(self, subcampaigns_number, hps=None):  # , time_period=3):
        self.subcampaigns = [gp.GPModel()
                             for x in range(subcampaigns_number)]
        if hps:
            # set hyper parameters
            for i, s in enumerate(self.subcampaigns):
                s._set_hp(hps[i])
#        self._opt = self._optimize
#        self.T = time_period

    def _optimize(self):
        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'enter _optimize')
        self.exp_revenue()
        # bid = np.linspace(0, MAX_BID, SAMPLES_INVERSE).reshape(-1, 1)
        # logger.info(f'bid array\n{bid}')

        # scl = [bid_maxrevenue_given_cost(bid, s.revenue(bid), s.cost(bid))
        #        for s in self.subcampaigns]
        # scl = np.array(scl)
        # return optimize(scl)
    
    # expected revenue
    def exp_revenue(self):
        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'enter revenue')
        bid = np.linspace(0, MAX_BID, SAMPLES_INVERSE).reshape(-1, 1)
        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'bid array\n{bid}')

        scl = [bid_maxrevenue_given_cost(bid, s.revenue(bid), s.cost(bid))
               for s in self.subcampaigns]
        scl = np.array(scl)

        r, bid_mix, rev_mix, cost_mix = optimize(scl)

        cost_array = np.linspace(0., MAX_EXP, N_COST)
        # r = np.where(np.isnan(r), 0., r)
        roi = r/cost_array
        #bid_mix = np.where(np.isnan(bid_mix), 0. , bid_mix)
        #cost_mix = np.where(np.isnan(cost_mix), 0. , cost_mix)


        return r, bid_mix, rev_mix, cost_mix

    def bid_choice(self, ret_roi_revenue_cost=False):
        if logger.isEnabledFor(logging.WARNING):
            logger.info('entering bid_choice')
        r, bid_mix, rev_mix, cost_mix = self._optimize()

        cost_array = np.linspace(0., MAX_EXP, N_COST)
        roi = r / cost_array

        if np.all(np.isnan(r[1:])) or np.all(r[~np.isnan(r)] <= 0.):
            if logger.isEnabledFor(logging.WARNING):
                logger.error(f'no feasible value found for the revenue of {type(self).__name__};\n'
                             f'the last row of the optimization matrix contains only NaN values\n{r}\n')
                         #f'use the same bids of the previous round\n')
            
            if self.delta:
                bid = np.linspace(0., MAX_BID, N_BID)
                bid_mix = [0. for s in self.subcampaigns]  # [s.get_last_bid() for s in self.subcampaigns]
                for i, s in enumerate(self.subcampaigns):
                    last_bid = s.get_last_bid()
                    bound_down = last_bid - self.delta_bid 
                    bound_down = bound_down if bound_down > 0 else 0.
                    bound_up = last_bid + self.delta_bid 
                    bound_up = bound_up if bound_up < MAX_BID else MAX_BID 
                    if logger.isEnabledFor(logging.WARNING):
                        logger.error(f'restrict the possible bids with the upper bound\n:{bound_up}\n')
                        logger.error(f'restrict the possible bids with the lower bound\n:{bound_down}')

                    bid_rng = bid[(bid >= bound_down) & (bid <= bound_up)]
                    if logger.isEnabledFor(logging.WARNING):
                        logger.error(f'restrict the possible bids with the mask\n:{bid_rng}')
                    bid_mix[i] = np.random.choice(bid_rng)
                    if logger.isEnabledFor(logging.WARNING):
                        logger.error(f'the chosen bid for the current sc\n:{bid_mix[i]}')
                if logger.isEnabledFor(logging.WARNING):
                    logger.error(f'bids chosen randomly\n:{bid_mix}')

                rev_mix = [s.revenue(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
                cost_mix = [s.cost(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
                roi = np.sum(np.array(rev_mix), axis=1) / np.sum(np.array(cost_mix), axis=1)
                return (bid_mix if not ret_roi_revenue_cost
                        else (bid_mix, r, roi, rev_mix, cost_mix, -1))


            bid_mix = [s.get_last_bid() for s in self.subcampaigns]
            rev_mix = [s.revenue(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
            cost_mix = [s.cost(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
            roi = np.sum(np.array(rev_mix), axis=1) / np.sum(np.array(cost_mix), axis=1)
            # roi = np.sum(np.array(rev_mix), axis=1) / cost_array
            return (bid_mix if not ret_roi_revenue_cost
                    else (bid_mix, r, roi, rev_mix, cost_mix, -1))

        if logger.isEnabledFor(logging.WARNING):
            logger.debug(f'r:\n{r}')
            logger.debug(f'bid_mix:\n{bid_mix}')
            logger.debug(f'cost_mix:\n{cost_mix}')

        #bid_mix = np.where(np.isnan(bid_mix), MAX_BID/N_BID, bid_mix)  # min bid > 0
        #logger.error(f'bid_mix removing nan:\n{bid_mix}')


        if logger.isEnabledFor(logging.WARNING):
            logger.debug(f'roi before masking on:\n{roi}')
        # mask = np.where(roi < MIN_ROI, np.nan, roi)
        mask = np.where(np.isnan(roi) | (roi < MIN_ROI), False, True)   # mask = np.where(roi < MIN_ROI, np.nan, True)
        mask[0] = False # no null cost
        if logger.isEnabledFor(logging.WARNING):
            logger.debug(f'mask with {False} roi lesser than {MIN_ROI}:\n{mask}')

        # if np.all(np.isnan(mask)) or np.all(np.isnan(r*mask)):
        if np.all(~mask) or np.all(np.isnan(r[mask])):  #(r*mask)):
            #idx = np.array(np.where(~np.isnan(r))).ravel()[0]  # -1 take the last not NaN element
            if np.all(np.isnan(roi)):
                idx = np.nanargmax(r)
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f'no ROI feasible\n'
                                   f'choose the bid:\n{bid_mix[idx]}\n'
                                   f'for {type(self).__name__}\n'
                                   f'that maximize the revenue\n'
                                   f'{r[idx]}\n'
                                   f'while the roi\n'
                                   f'{roi[idx]}\n')
                return (bid_mix[idx] if not ret_roi_revenue_cost
                        else (bid_mix, r, roi, rev_mix, cost_mix, idx))

                 

            #idx = np.nanargmax(roi)
            
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f'no ROI greater than:{MIN_ROI}\n')
            # ROI can't be all zero: eventually the condition will be satisfied
            min_roi = np.nanmax(roi)
            while np.all(~mask):
                min_roi *= 0.9
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f'Consider the new ROI target:{min_roi}\n')
                mask = np.where(np.isnan(roi) | (roi<min_roi), False, True)

            idx = np.nanargmax(r*mask)
            if logger.isEnabledFor(logging.ERROR):
                logger.warning(f'the chosen index:{idx}\n'
                           f'the chosen bids:\n{bid_mix[idx]}\n'
                           f'for {type(self).__name__}\n'
                           f'that maximize the new target roi\n'  #revenue\n'
                           f'the chosen revenue\n{r[idx]}\n'
                           f'while the roi\n'
                           f'{roi[idx]}\n'
                           #f'the possible revenues were:\n{r[mask]}'
                           f'while r was:\n{r}\n'
                           f'and roi was:\n{roi}\n'
                           f'chosen cost_mix:\n{cost_mix[idx]}\n')
            # idx = np.nanargmax(roi)
            # logger.warning(f'no ROI greater than:{MIN_ROI}\n'
            #                f'choose the bid:\n{bid_mix[idx]}\n'
            #                f'for {type(self).__name__}\n'
            #                f'that maximize the roi\n'  #revenue\n'
            #                f'{r[idx]}\n'
            #                f'while the roi\n'
            #                f'{roi[idx]}\n'
            #                f'the possible revenues were:\n{r[mask]}'
            #                f'while r was:\n{r}'
            #                f'and roi was:\n{roi}')
            # logger.warning(f'r:\n{r}')
            #logger.warning(f'chosen bid_mix:\n{bid_mix[idx]}')
            #if logger.isEnabledFor(logging.WARNING):
            #    logger.warning(f'chosen cost_mix:\n{cost_mix[idx]}')

            # rts_roi = 0
            # rts_rev = 0  # true_revenue_campaign[idx_true_max_roi_campaign+1]
            
            # r = np.where(np.isnan(r), 0., r)
            # roi = r/cost_array

            # return (bid_mix[-1] if not ret_roi_revenue_cost
            #         else (bid_mix[-1], roi[-1], rev_mix[-1], cost_mix[-1]))
            return (bid_mix[idx] if not ret_roi_revenue_cost
                    else (bid_mix, r, roi, rev_mix, cost_mix, idx))

        # idx = np.nanargmax(np.where(~np.isnan(mask), r[-1], np.nan))
        idx = np.nanargmax(r*mask)  #r * mask)
        # np.nanargmax(mask)

        if logger.isEnabledFor(logging.ERROR):
            logger.error(f'index that maximize the revenue - given a minimum ROI:\n'
                         f'{idx}\n'
                         f'the revenue values that respect the ROI constraint:'
                         f'\n{r[mask]}\n'
                         f'the ROI:'
                         f'\n{roi}\n'
                         f'the chosen values for {type(self).__name__}:\n'
                         f'the chosen revenue value:\n{r[idx]}\n'
                         f'the associated roi value:\n{roi[idx]}\n'
                         f'the chosen bid mix:\n{bid_mix[idx]}\n'
                         f'the chosen cost mix:\n{cost_mix[idx]}\n')

        # r = np.where(np.isnan(r), 0., r)
        # roi = r/cost_array
        # return (bid_mix[idx] if not ret_roi_revenue_cost
        #         else (bid_mix[idx], roi[idx], rev_mix[idx], cost_mix[idx]))
        return (bid_mix[idx] if not ret_roi_revenue_cost
                else (bid_mix, r, roi, rev_mix, cost_mix, idx))
        # rts_roi = roi_rts[idx_roi_max_rev]
        # rev_ts = rts[-1][idx_roi_max_rev]

        # print('rev_ts', rev_ts)
        # print('rts roi', rts_roi)

        # X = np.append(X, np.array(
        #     bid_mix_ts[-1][idx_roi_max_rev]).reshape(-1, 1), axis=1)

    def update(self, X_bid, Y_cost, Y_rev):
        logging.info('entering update')
        for i, m in enumerate(self.subcampaigns):
            m.update(X_bid[i], Y_cost[i], Y_rev[i])

    def get_model(self, gp, bid):
        # bid = np.linspace(0, MAX_BID, SAMPLES)
        sc = self.subcampaigns[gp]

        return sc.cost(bid, return_var=True), sc.revenue(bid, return_var=True)


class TS_Agent(Agent):
    def __init__(self, subcampaigns_number, hps=None):
        self._optimize = self._optimize
        super().__init__(subcampaigns_number, hps)

    def __str__(self):
        return f'free exploration agent'

    def _optimize(self):
        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'enter TS_Agent _optimize')
        bid = np.linspace(0., MAX_BID, SAMPLES_INVERSE)

        scl = [bid_maxrevenue_given_cost(bid, s.revenue(bid, sample_y=True),
               s.cost(bid, sample_y=True))
               for s in self.subcampaigns]
        scl = np.array(scl)
        return optimize(scl)

class TS_Conservative_Agent(Agent):
    def __init__(self, subcampaigns_number, hps=None, delta_bid=3.):
        self._optimize = self._optimize
        self.delta = delta_bid
        self.delta_bid = (MAX_BID/100.0) * delta_bid 
        super().__init__(subcampaigns_number, hps)

    def __str__(self):
        return f'{self.delta:.0f}% exploration agent'

    def _optimize(self):
        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'enter {type(self).__name__} _optimize')
        bid = np.linspace(0., MAX_BID, SAMPLES_INVERSE)
        # delta_bid = (MAX_BID/100.0) * 10.0 

        delta_bid = self.delta_bid

        scl = [bid_maxrevenue_given_cost(bid, s.revenue(bid, sample_y=True), s.cost(bid, sample_y=True), min_bid=s.get_last_bid() - self.delta_bid if (s.get_last_bid() - self.delta_bid) > 0.0 else 0.0, max_bid=s.get_last_bid() + self.delta_bid if (s.get_last_bid() + self.delta_bid) < MAX_BID else MAX_BID)
               for s in self.subcampaigns]
        scl = np.array(scl)
        # logger.error(f'scl:\n{scl}')
        # i = 0
        # if np.all(np.isnan(scl[:, :, 1])):
        #     logger.error(f'all nan revenues {type(self).__name__}\n')
        # while np.all(np.isnan(scl[1])) and delta_bid <= MAX_BID and i < 5:
        #     i += 1
        #     logger.error(f'no feasible value found for the current bid range of {type(self).__name__};\n'
        #                  f'Increment the bid-range of {MAX_BID/100.0}\n')
        #     delta_bid += MAX_BID/100.0
        #     scl = [bid_maxrevenue_given_cost(bid, s.revenue(bid, sample_y=True), s.cost(bid, sample_y=True), min_bid=s.get_last_bid() - delta_bid if (s.get_last_bid() - delta_bid) >= 0.0 else 0.0, max_bid=s.get_last_bid() + delta_bid if (s.get_last_bid() + delta_bid) <= MAX_BID else MAX_BID)
        #            for s in self.subcampaigns]
        # 
        # scl = np.array(scl)
        # return optimize(scl)
        i = 0
        r, _, _, _ = optimize(scl)
        if np.all(np.isnan(r[1:])):
            if logger.isEnabledFor(logging.WARNING):
                logger.error(f'all nan revenues {type(self).__name__}\n')
        while np.all(np.isnan(r[:])) and delta_bid <= MAX_BID and i < 3:
            i += 1
            if logger.isEnabledFor(logging.WARNING):
                logger.error(f'no feasible value found for the current bid range of {type(self).__name__};\n'
                         f'Increment the bid-range of {MAX_BID/100.0}\n')
            delta_bid += MAX_BID/100.0
            scl = [bid_maxrevenue_given_cost(bid, s.revenue(bid, sample_y=True), s.cost(bid, sample_y=True), min_bid=s.get_last_bid() - delta_bid if (s.get_last_bid() - delta_bid) >= 0.0 else 0.0, max_bid=s.get_last_bid() + delta_bid if (s.get_last_bid() + delta_bid) <= MAX_BID else MAX_BID)
                   for s in self.subcampaigns]
            scl = np.array(scl)
            r, _, _, _ = optimize(scl)
        
        #scl = np.array(scl)
        return optimize(scl)


class TS_Conservative_Agent_1(TS_Conservative_Agent):
    def __init__(self, subcampaigns_number, hps=None, delta_bid=1.):
        super().__init__(subcampaigns_number, delta_bid)
        self.delta = 1.0
        self.delta_bid = (MAX_BID/100.0) * self.delta
    def __str__(self):
        return f'{self.delta:.0f}% exploration agent'

class TS_Conservative_Agent_5(TS_Conservative_Agent):
    def __init__(self, subcampaigns_number, hps=None, delta_bid=5.):
        super().__init__(subcampaigns_number, delta_bid)
        self.delta = 5.0
        self.delta_bid = (MAX_BID/100.0) * self.delta

class TS_Conservative_Agent_10(TS_Conservative_Agent):
    def __init__(self, subcampaigns_number, hps=None, delta_bid=10.):
        super().__init__(subcampaigns_number, delta_bid)
        self.delta = 10.0
        self.delta_bid = (MAX_BID/100.0) * self.delta

class TS_TFP_Agent(TS_Agent):
    def __init__(self, subcampaigns_number, hps=None):
        super().__init__(subcampaigns_number)
        self.subcampaigns = [gp.TFP_GPModel()
                             for x in range(subcampaigns_number)]


class TS_MCMC_Agent(TS_Agent):
    def __init__(self, subcampaigns_number, hps=None):
        super().__init__(subcampaigns_number)
        self.subcampaigns = [gp.GPModel(OPT=False, MCMC=True)
                             for x in range(subcampaigns_number)]




class Safe_Opt_Agent(Agent):
    def __init__(self, subcampaigns_number, hps=None, ci=0.75, ci_cost=0.5, ci_rev=0.5):  # , delta_bid=3.):
        super().__init__(subcampaigns_number, hps)
        self._optimize = self._optimize
        self.bid_choice = self.bid_choice
        self.ci_cost = ci_cost
        self.ci_rev = ci_rev
        self._opt_func = safe_opt
        self._roi_mask = self._build_roi_mask()

    def _build_roi_mask(self):
        rev_array = np.linspace(0., MAX_REV, N_REV)
        cost_array = np.linspace(0., MAX_EXP, N_COST)
        roi_mask = np.empty((N_REV,N_COST))

        for i, rev in enumerate(rev_array):
            for j, cost in enumerate(cost_array):
                # if (~np.isnan(rev/cost)) & (~np.isinf(rev/cost)) & (rev/cost >= MIN_ROI):
                if (np.isfinite(rev/cost)) & (rev/cost >= MIN_ROI):
                    roi_mask[i,j] = 1.
                else:
                    roi_mask[i,j] = np.nan
        return roi_mask


    def __str__(self):
        return f'safe ucb agent'

    def _optimize(self):
        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'enter {type(self).__name__} _optimize')
        bid = np.linspace(0., MAX_BID, SAMPLES_INVERSE)

        #rev = [(mean, var) for mean, var in s.revenue(bid, return_var=True, sample_y=False) for s in self.subcampaigns]
        mean_rev = np.empty((len(self.subcampaigns), N_BID, 1))
        sigma_rev = np.empty((len(self.subcampaigns), N_BID, 1))
        mean_cost = np.empty((len(self.subcampaigns), N_BID, 1))
        sigma_cost = np.empty((len(self.subcampaigns), N_BID, 1))
        for i, s in enumerate(self.subcampaigns):
          #  for b in range(N_BID):
            mean_rev[i, :], sigma_rev[i, :] = s.revenue(bid, return_var=True, sample_y=False)
            mean_cost[i, :], sigma_cost[i, :] = s.cost(bid, return_var=True, sample_y=False)
        sigma_rev = np.sqrt(sigma_rev)
        sigma_cost = np.sqrt(sigma_cost)

        ci_rev = self.ci_rev  # 1.  # 0.  # 1. #0.75  # 1.96  #0.5  # 1.  # 1.645 # 1.96
        ci_cost = self.ci_cost  # 1.  # 0.  # 1. #0.75  # 1.96  #0.5  # 1.  # 1.645 # 1.96

        scl = [bid_max_ub_revenue_given_lb_rev_and_cost(bid, lb_rev=mean_rev[s] - ci_rev*sigma_rev[s],
                                             ub_rev=mean_rev[s] + ci_rev*sigma_rev[s], ub_cost=mean_cost[s] + ci_cost*sigma_cost[s])
                for s, _ in enumerate(self.subcampaigns)
                ]

        scl = np.array(scl)
        return self._opt_func(scl)


    def bid_choice(self, ret_roi_revenue_cost=False):
        if logger.isEnabledFor(logging.WARNING):
            logger.info('entering bid_choice')
        r, bid_mix, rev_mix, cost_mix = self._optimize()

        # r = r[:, -1] # the last ROI

        # cost_array = np.linspace(0., MAX_EXP, N_COST)
        # roi = r / cost_array
        cost_array = np.linspace(0., MAX_EXP, N_COST)
        try:
            idx = np.unravel_index(np.nanargmax(r*self._roi_mask, axis=None), shape=(r*self._roi_mask).shape)
        except ValueError:
            #bid_mix = [FIRST_BID for s in self.subcampaigns]
            bid_mix = [s.get_last_bid() for s in self.subcampaigns]
            check_bid_mix = np.array(bid_mix)
            if not check_bid_mix.any():
                bid_mix = [FIRST_BID for s in self.subcampaigns]
            rev_mix = [s.revenue(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
            cost_mix = [s.cost(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
            roi = np.sum(np.array(rev_mix), axis=1) / np.sum(np.array(cost_mix), axis=1)
            bid_mix = np.array(bid_mix).reshape((1, len(self.subcampaigns)))
            if logger.isEnabledFor(logging.ERROR):
                logger.error(f'no feasible value found for the revenue of {type(self).__name__};\n'
                             f'the last row of the optimization matrix contains only NaN values\n{r*self._roi_mask}\n'
                             f'use the same bids of the last round\n:{bid_mix}\n')
            # roi = np.sum(np.array(rev_mix), axis=1) / cost_array
            return (bid_mix if not ret_roi_revenue_cost
                    else (bid_mix, r, roi, rev_mix, cost_mix, -1))

        #roi = r[idx[0], :]/cost_array
        roi = r/cost_array
        rev_array = np.linspace(0., MAX_REV, N_REV)
        logger.error(f'roi.shape:{roi.shape}')
        logger.error(f'r.shape:{r.shape}')

        if logger.isEnabledFor(logging.ERROR):
            logger.error(f'index that maximize the revenue - given a minimum ROI:\n'
                           f'{idx}\n')
                           #f'and the roi:\n{roi_array[roi_thr]}\n')

            #logger.error(f'the last row of the optimization matrix: \n{r[:, idx[1]]}')
            logger.error(f'the revenue values that respect the ROI constraint:'
                         f'\n{r[idx]}\n'
                         f'with ci_rev:{self.ci_rev}\n'
                         f'with ci_cost:{self.ci_cost}\n'
                        f'\n{r[idx]}\n')
            logger.error(f'the ROI:'
                        f'\n{roi[idx]}\n')
            logger.error(f'the chosen values for {type(self).__name__}:\n'
                           f'the chosen revenue value:\n{r[idx]}\n'
                           f'the associated roi value:\n{roi[idx]}\n'
                           f'the chosen bid mix:\n{bid_mix[idx]}\n'
                           f'the chosen cost mix:\n{cost_mix[idx]}')

        return (bid_mix[idx] if not ret_roi_revenue_cost
                else (bid_mix, r, roi, rev_mix, cost_mix, idx))


class Safe_Opt_eps_greedy_Agent(Safe_Opt_Agent):
    def __init__(self, subcampaigns_number, hps=None):  # , delta_bid=3.):
        super().__init__(subcampaigns_number, hps)
        self.__str__ = self.__str__
        self.bid_choice = self.bid_choice
        self._rng = np.random.default_rng()
        self._roi_lbrev_mask = self._build_roi_lbrev_mask()

    def _build_roi_lbrev_mask(self):
        rev_array = np.linspace(0., MAX_REV, N_REV)
        cost_array = np.linspace(0., MAX_EXP, N_COST)
        roi_mask = np.empty((N_REV,N_COST))

        for i, rev in enumerate(rev_array):
            for j, cost in enumerate(cost_array):
                if ~np.isnan(rev/cost) & (rev/cost >= MIN_ROI):
                    roi_mask[i,j] = rev
                else:
                    roi_mask[i,j] = np.nan 

        return roi_mask

    def __str__(self):
        return f'safe opt ε-greedy agent'

    def bid_choice(self, ret_roi_revenue_cost=False):
        if logger.isEnabledFor(logging.WARNING):
            logger.info('entering bid_choice')
        r, bid_mix, rev_mix, cost_mix = self._optimize()
        val = self._rng.binomial(1, 0.05)

        cost_array = np.linspace(0., MAX_EXP, N_COST)

        if val:
            # maximize variance
            
            try:
                idx = np.unravel_index(np.nanargmax(r-self._roi_lbrev_mask, axis=None), shape=r.shape)
            except ValueError:
                #bid_mix = [FIRST_BID for s in self.subcampaigns]
                bid_mix = [s.get_last_bid() for s in self.subcampaigns]
                check_bid_mix = np.array(bid_mix)
                if not check_bid_mix.any():
                    bid_mix = [FIRST_BID for s in self.subcampaigns]
                rev_mix = [s.revenue(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
                cost_mix = [s.cost(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
                roi = np.sum(np.array(rev_mix), axis=1) / np.sum(np.array(cost_mix), axis=1)
                if logger.isEnabledFor(logging.ERROR):
                    logger.error(f'no feasible value found for the revenue of {type(self).__name__};\n'
                                 f'the last row of the optimization matrix contains only NaN values\n{r}\n'
                                 f'use the same bids of the last round\n:{bid_mix}\n')
                # roi = np.sum(np.array(rev_mix), axis=1) / cost_array
                return (bid_mix if not ret_roi_revenue_cost
                        else (bid_mix, r, roi, rev_mix, cost_mix, -1))

            #roi = r[idx[0], :]/cost_array
            roi = r/cost_array

            if logger.isEnabledFor(logging.ERROR):
                logger.error(f'In this round {type(self).__name__} maximizes the variance')
                logger.error(f'index that maximize the variance (the difference between upper and lower bound for the revenue):\n'
                               f'{idx}\n')
                               #f'and the roi:\n{roi_array[roi_thr]}\n')

                #logger.error(f'the last row of the optimization matrix: \n{r[:, idx[1]]}')
                logger.error(f'the revenue values that respect the ROI constraint:'
                             f'\n{r[idx]}\n'
                             f'with ci_rev:{self.ci_rev}\n'
                             f'with ci_cost:{self.ci_cost}\n'
                            f'\n{r[idx]}\n')
                logger.error(f'the ROI:'
                            f'\n{roi[idx[1]]}\n')
                logger.error(f'the chosen values for {type(self).__name__}:\n'
                               f'the chosen revenue value:\n{r[idx]}\n'
                               f'the associated roi value:\n{roi[idx[1]]}\n'
                               f'the chosen bid mix:\n{bid_mix[idx]}\n'
                               f'the chosen cost mix:\n{cost_mix[idx]}')

            return (bid_mix[idx] if not ret_roi_revenue_cost
                    else (bid_mix, r, roi, rev_mix, cost_mix, idx))
        else:
            # maximize revenue
            try:
                idx = np.unravel_index(np.nanargmax(r*self._roi_mask, axis=None), shape=r.shape)
            except ValueError:
                #bid_mix = [FIRST_BID for s in self.subcampaigns]
                bid_mix = [s.get_last_bid() for s in self.subcampaigns]
                check_bid_mix = np.array(bid_mix)
                if not check_bid_mix.any():
                    bid_mix = [FIRST_BID for s in self.subcampaigns]

                rev_mix = [s.revenue(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
                cost_mix = [s.cost(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
                roi = np.sum(np.array(rev_mix), axis=1) / np.sum(np.array(cost_mix), axis=1)
                if logger.isEnabledFor(logging.ERROR):
                    logger.error(f'no feasible value found for the revenue of {type(self).__name__};\n'
                                 f'the last row of the optimization matrix contains only NaN values\n{r}\n'
                                 f'use the same bids of the last round\n:{bid_mix}\n')
                # roi = np.sum(np.array(rev_mix), axis=1) / cost_array
                return (bid_mix if not ret_roi_revenue_cost
                        else (bid_mix, r, roi, rev_mix, cost_mix, -1))

            #roi = r[idx[0], :]/cost_array
            roi = r/cost_array

            if logger.isEnabledFor(logging.ERROR):
                logger.error(f'In this round {type(self).__name__} maximizes the revenue')
                logger.error(f'index that maximize the revenue - given a minimum ROI:\n'
                               f'{idx}\n')
                               #f'and the roi:\n{roi_array[roi_thr]}\n')

                #logger.error(f'the last row of the optimization matrix: \n{r[:, idx[1]]}')
                logger.error(f'the revenue values that respect the ROI constraint:'
                             f'\n{r[idx]}\n'
                             f'with ci_rev:{self.ci_rev}\n'
                             f'with ci_cost:{self.ci_cost}\n'
                            f'\n{r[idx]}\n')
                logger.error(f'the ROI:'
                            f'\n{roi[idx[1]]}\n')
                logger.error(f'the chosen values for {type(self).__name__}:\n'
                               f'the chosen revenue value:\n{r[idx]}\n'
                               f'the associated roi value:\n{roi[idx[1]]}\n'
                               f'the chosen bid mix:\n{bid_mix[idx]}\n'
                               f'the chosen cost mix:\n{cost_mix[idx]}')

            return (bid_mix[idx] if not ret_roi_revenue_cost
                    else (bid_mix, r, roi, rev_mix, cost_mix, idx))


class GCB_Agent(Safe_Opt_Agent):

    def __init__(self, subcampaigns_number, T, hps=None):
        super().__init__(subcampaigns_number, hps)
        self._optimize = self._optimize
        self.__str__ = self.__str__
        self.T = T
        self.time = 0

    def __str__(self):
        return f'GCB agent'


    def _optimize(self):
        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'enter {type(self).__name__} _optimize')
        bid = np.linspace(0., MAX_BID, SAMPLES_INVERSE)

        #rev = [(mean, var) for mean, var in s.revenue(bid, return_var=True, sample_y=False) for s in self.subcampaigns]
        mean_rev = np.empty((len(self.subcampaigns), N_BID, 1))
        sigma_rev = np.empty((len(self.subcampaigns), N_BID, 1))
        mean_cost = np.empty((len(self.subcampaigns), N_BID, 1))
        sigma_cost = np.empty((len(self.subcampaigns), N_BID, 1))
        for i, s in enumerate(self.subcampaigns):
          #  for b in range(N_BID):
            mean_rev[i, :], sigma_rev[i, :] = s.revenue(bid, return_var=True, sample_y=False)
            mean_cost[i, :], sigma_cost[i, :] = s.cost(bid, return_var=True, sample_y=False)
        sigma_rev = np.sqrt(sigma_rev)
        sigma_cost = np.sqrt(sigma_cost)

        X, _ = self.subcampaigns[0].model_cost.data
        # t = len(X) - 1
        t = len(X) - 1 - INITIAL_BIDS
        t = t if t>0 else 1

        # coeff = np.sqrt(
        #             2 * 
        #             np.log((12 * len(self.subcampaigns) * N_COST * self.T * t**2)
        #                     /
        #                     np.pi**2)
        #         )
        coeff = np.sqrt(
                    # 2 * 
                    # np.log((12 * len(self.subcampaigns) * t**2)
                    np.log((len(self.subcampaigns) * t**2)
                            /
                            (.2 * np.pi**2))
                            # (.05 * np.pi**2))
                )
        logger.error(f'the coefficient is: {coeff}\n time is equal to:{t}\n')

        ci_rev = self.ci_rev = coeff # self.ci_rev  # 1.  # 0.  # 1. #0.75  # 1.96  #0.5  # 1.  # 1.645 # 1.96
        ci_cost = self.ci_cost = coeff  #self.ci_cost  # 1.  # 0.  # 1. #0.75  # 1.96  #0.5  # 1.  # 1.645 # 1.96

        scl = [bid_max_ub_revenue_given_lb_rev_and_cost(bid, lb_rev=mean_rev[s] + ci_rev*sigma_rev[s],
                                             ub_rev=mean_rev[s] + ci_rev*sigma_rev[s], ub_cost=mean_cost[s] - ci_cost*sigma_cost[s])
                for s, _ in enumerate(self.subcampaigns)
                ]

        scl = np.array(scl)

        return self._opt_func(scl)

class GCBsafe_Agent(GCB_Agent):
    def __init__(self, subcampaigns_number, T, hps=None, eps=1.):
        self.eps = eps
        super().__init__(subcampaigns_number, T, hps)
        self._optimize = self._optimize
        self.__str__ = self.__str__
        self._build_roi_mask = self._build_roi_mask
        self._roi_mask = self._build_roi_mask()


    def __str__(self):
        return f"GCB {f'{self.eps*100}%' if self.eps != 1. else ''} safe Agent"

    def _build_roi_mask(self):
        rev_array = np.linspace(0., MAX_REV, N_REV)
        cost_array = np.linspace(0., MAX_EXP, N_COST)
        roi_mask = np.empty((N_REV,N_COST))
        min_roi = MIN_ROI*self.eps

        for i, rev in enumerate(rev_array):
            for j, cost in enumerate(cost_array):
                # if (~np.isnan(rev/cost)) & (~np.isinf(rev/cost)) & (rev/cost >= MIN_ROI):
                if (np.isfinite(rev/cost)) & (rev/cost >= min_roi):
                    roi_mask[i,j] = 1.
                else:
                    roi_mask[i,j] = np.nan
        return roi_mask

    def _optimize(self):
        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'enter {type(self).__name__} _optimize')
        bid = np.linspace(0., MAX_BID, SAMPLES_INVERSE)

        mean_rev = np.empty((len(self.subcampaigns), N_BID, 1))
        sigma_rev = np.empty((len(self.subcampaigns), N_BID, 1))
        mean_cost = np.empty((len(self.subcampaigns), N_BID, 1))
        sigma_cost = np.empty((len(self.subcampaigns), N_BID, 1))
        for i, s in enumerate(self.subcampaigns):
            mean_rev[i, :], sigma_rev[i, :] = s.revenue(bid, return_var=True, sample_y=False)
            mean_cost[i, :], sigma_cost[i, :] = s.cost(bid, return_var=True, sample_y=False)
        sigma_rev = np.sqrt(sigma_rev)
        sigma_cost = np.sqrt(sigma_cost)

        X, _ = self.subcampaigns[0].model_cost.data
        # t = len(X) - 1
        t = len(X) - 1 - INITIAL_BIDS
        t = t if t>0 else 1

        # coeff = np.sqrt(
        #             2 * 
        #             np.log((12 * len(self.subcampaigns) * N_COST * self.T * t**2)
        #                     /
        #                     np.pi**2)
        #         )
        coeff = np.sqrt(
                    # 2 * 
                    # np.log((12 * len(self.subcampaigns) * t**2)
                    np.log((len(self.subcampaigns) * t**2)
                    # np.log((12 * len(self.subcampaigns) * t**2)
                            /
                            (.2 * np.pi**2))
                            # (.05 * np.pi**2))
                )

        ci_rev = self.ci_rev = coeff # self.ci_rev  # 1.  # 0.  # 1. #0.75  # 1.96  #0.5  # 1.  # 1.645 # 1.96
        ci_cost = self.ci_cost = coeff  #self.ci_cost  # 1.  # 0.  # 1. #0.75  # 1.96  #0.5  # 1.  # 1.645 # 1.96

        scl = [bid_max_ub_revenue_given_lb_rev_and_cost(bid, lb_rev=mean_rev[s] - ci_rev*sigma_rev[s],
                                             ub_rev=mean_rev[s] + ci_rev*sigma_rev[s], ub_cost=mean_cost[s] + ci_cost*sigma_cost[s])
                for s, _ in enumerate(self.subcampaigns)
                ]

        scl = np.array(scl)

        return self._opt_func(scl)


class GCBsafe95_Agent(GCBsafe_Agent):
    def __init__(self, subcampaigns_number, T, hps=None, eps=.95):
        self.eps = eps
        super().__init__(subcampaigns_number, T, hps, eps)


class GCBsafe90_Agent(GCBsafe_Agent):
    def __init__(self, subcampaigns_number, T, hps=None, eps=.90):
        self.eps = eps
        super().__init__(subcampaigns_number, T, hps, eps)


class GCBsafe85_Agent(GCBsafe_Agent):
    def __init__(self, subcampaigns_number, T, hps=None, eps=.85):
        self.eps = eps
        super().__init__(subcampaigns_number, T, hps, eps)


#class Clairvoyant_Agent(Agent):
class Clairvoyant_Agent(Safe_Opt_Agent):
    def __init__(self, subcampaigns, hps=None):
        self._optimize = self._optimize
        self.subcampaigns = subcampaigns
        #self.bid_choice = self.bid_choice
        self.ci_cost = 0. #ci_cost
        self.ci_rev = 0.  #ci_rev

        self._roi_mask = self._build_roi_mask()
        # self.delta = delta_bid
        # self.delta_bid = (MAX_BID/100.0) * delta_bid 

    def _build_roi_mask(self):
        rev_array = np.linspace(0., MAX_REV, N_REV)
        cost_array = np.linspace(0., MAX_EXP, N_COST)
        roi_mask = np.empty((N_REV,N_COST))

        for i, rev in enumerate(rev_array):
            for j, cost in enumerate(cost_array):
                if ~np.isnan(rev/cost) & (rev/cost >= MIN_ROI):
                    roi_mask[i,j] = 1.
                else:
                    roi_mask[i,j] = np.nan
        return roi_mask

    def _optimize(self):
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(f'enter Clairvoyant _optimize')
            logger.warning(f'self.subcampaigns:\n{self.subcampaigns}')

        bid = np.linspace(0, MAX_BID, SAMPLES_INVERSE)

        scl = [bid_max_ub_revenue_given_lb_rev_and_cost(bid, s.revenue(bid, noise=False), s.revenue(bid, noise=False),
               s.cost(bid, noise=False))
               for s in self.subcampaigns]
        scl = np.array(scl)

        if logger.isEnabledFor(logging.WARNING):
            logger.warning(f'bid_maxrevenue_given_cost of all the subc:\n{scl}')

        return safe_opt(scl)

    def update(self, X_bid, Y_cost, Y_rev):
        if logger.isEnabledFor(logging.WARNING):
            logger.warning('here we are')
        # nothing to do here
        pass

    def get_model(self, gp, bid):
        #        bid = np.linspace(0, MAX_BID, SAMPLES)
        sc = self.subcampaigns[gp]
        return (sc.cost(bid, noise=False), np.zeros(bid.shape)), (sc.revenue(bid, noise=False), np.zeros(bid.shape))

#class Clairvoyant_Agent(Agent):
class Clairvoyant_B_Agent(Agent):
    def __init__(self, subcampaigns, hps=None):
        self._optimize = self._optimize
        self.subcampaigns = subcampaigns
        #self.bid_choice = self.bid_choice
        #self.ci_cost = 0. #ci_cost
        #self.ci_rev = 0.  #ci_rev

        #self._roi_mask = self._build_roi_mask()
        # self.delta = delta_bid
        # self.delta_bid = (MAX_BID/100.0) * delta_bid 

    # def _build_roi_mask(self):
    #     rev_array = np.linspace(0., MAX_REV, N_REV)
    #     cost_array = np.linspace(0., MAX_EXP, N_COST)
    #     roi_mask = np.empty((N_REV,N_COST))

    #     for i, rev in enumerate(rev_array):
    #         for j, cost in enumerate(cost_array):
    #             if ~np.isnan(rev/cost) & (rev/cost >= MIN_ROI):
    #                 roi_mask[i,j] = 1.
    #             else:
    #                 roi_mask[i,j] = np.nan
    #     return roi_mask

    def _optimize(self):
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(f'enter Clairvoyant _optimize')
            logger.warning(f'self.subcampaigns:\n{self.subcampaigns}')

        bid = np.linspace(0, MAX_BID, SAMPLES_INVERSE)

        scl = [bid_maxrevenue_given_cost(bid, s.revenue(bid, noise=False),
               s.cost(bid, noise=False))
               for s in self.subcampaigns]
        scl = np.array(scl)

        # bid = np.linspace(0, MAX_BID, SAMPLES_INVERSE)

        # scl = [bid_max_lb_revenue_given_cost(bid, s.revenue(bid, noise=False),
        #        s.cost(bid, noise=False), s.revenue(bid, noise=False), s.cost(bid, noise=False))
        #        for s in self.subcampaigns]
        # scl = np.array(scl)
        # scl = [bid_max_ub_revenue_given_lb_rev_and_cost(bid, s.revenue(bid, noise=False), s.revenue(bid, noise=False),
        #        s.cost(bid, noise=False))
        #        for s in self.subcampaigns]
        # scl = np.array(scl)

        if logger.isEnabledFor(logging.WARNING):
            logger.warning(f'bid_maxrevenue_given_cost of all the subc:\n{scl}')
        return optimize(scl)
        #return safe_optimize_roi(scl)
        #return safe_opt(scl)

    def update(self, X_bid, Y_cost, Y_rev):
        if logger.isEnabledFor(logging.WARNING):
            logger.warning('here we are')
        # nothing to do here
        pass

    def get_model(self, gp, bid):
        #        bid = np.linspace(0, MAX_BID, SAMPLES)
        sc = self.subcampaigns[gp]
        return (sc.cost(bid, noise=False), np.zeros(bid.shape)), (sc.revenue(bid, noise=False), np.zeros(bid.shape))


class Safe_Agent(Agent):
    # def bid_choice(self, ret_roi_revenue_cost=False):
    #     super()
    def __init__(self, subcampaigns_number, ci=0.75, ci_cost=1., ci_rev=1.):  # , delta_bid=3.):
        self._optimize = self._optimize
        self.bid_choice = self.bid_choice
        self.ci_cost = ci_cost
        self.ci_rev = ci_rev
        self._opt_func = safe_optimize_roi
        # self.delta = delta_bid
        # self.delta_bid = (MAX_BID/100.0) * delta_bid 
        super().__init__(subcampaigns_number)

    def __str__(self):
        return f'safe optimization ucb agent'

    def _optimize(self):
        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'enter {type(self).__name__} _optimize')
        bid = np.linspace(0., MAX_BID, SAMPLES_INVERSE)

        #rev = [(mean, var) for mean, var in s.revenue(bid, return_var=True, sample_y=False) for s in self.subcampaigns]
        mean_rev = np.empty((len(self.subcampaigns), N_BID, 1))
        sigma_rev = np.empty((len(self.subcampaigns), N_BID, 1))
        mean_cost = np.empty((len(self.subcampaigns), N_BID, 1))
        sigma_cost = np.empty((len(self.subcampaigns), N_BID, 1))
        for i, s in enumerate(self.subcampaigns):
          #  for b in range(N_BID):
            mean_rev[i, :], sigma_rev[i, :] = s.revenue(bid, return_var=True, sample_y=False)
            mean_cost[i, :], sigma_cost[i, :] = s.cost(bid, return_var=True, sample_y=False)
        sigma_rev = np.sqrt(sigma_rev)
        sigma_cost = np.sqrt(sigma_cost)
        #rev = [s.revenue(bid, return_var=True, sample_y=False) for s in self.subcampaigns]
        # print(mean_rev)
        # print(sigma_rev)
        # mean_rev,var_rev = rev
        # var_rev = rev[:, 1]
        # cost = [s.cost(bid, return_var=True, sample_y=False) for s in self.subcampaigns]
        # mean_cost = rev[:, 0]
        # var_cost = rev[:, 1]
        ci_rev = self.ci_rev  # 1.  # 0.  # 1. #0.75  # 1.96  #0.5  # 1.  # 1.645 # 1.96
        ci_cost = self.ci_cost  # 1.  # 0.  # 1. #0.75  # 1.96  #0.5  # 1.  # 1.645 # 1.96

        scl = [bid_max_lb_revenue_given_cost(bid, mean_rev[s] - ci_rev*sigma_rev[s], mean_cost[s] - ci_cost*sigma_cost[s],
                                             mean_rev[s] + ci_rev*sigma_rev[s], mean_cost[s] + ci_cost*sigma_cost[s])
                for s, _ in enumerate(self.subcampaigns)
                # mean_rev, var_rev = s.revenue(bid, return_var=True, sample_y=False)
                # mean_cost, var_cost = s.cost(bid, return_var=True, sample_y=False)
                ]
                #for s in self.subcampaigns]
        #print(scl)

        # scl = [bid_max_lb_revenue_given_cost(bid, s.revenue(bid, sample_y=True),
        #        s.cost(bid, sample_y=True))
        #        for s in self.subcampaigns]
        scl = np.array(scl)
        #return safe_optimize_roi(scl)
        return self._opt_func(scl)

    def bid_choice(self, ret_roi_revenue_cost=False):
        if logger.isEnabledFor(logging.WARNING):
            logger.info('entering bid_choice')
        r, bid_mix, rev_mix, cost_mix = self._optimize()

        # r = r[:, -1] # the last ROI

        # cost_array = np.linspace(0., MAX_EXP, N_COST)
        # roi = r / cost_array
        cost_array = np.linspace(0., MAX_EXP, N_COST)
        roi_array = np.linspace(LOW_ROI, MAX_ROI, N_ROI)
        roi = (r.T / cost_array).T
        roi_thr = np.argmax(roi_array >= MIN_ROI)

        if np.all(np.isnan(r[1:])) or np.all(r[~np.isnan(r)] <= 0.):

            # safe previous value
            ci_cost = self.ci_cost
            self.ci_cost = 0.
            r, bid_mix, rev_mix, cost_mix = self._optimize()
            roi = (r.T / cost_array).T
            if np.all(np.isnan(r[1:])) or np.all(r[~np.isnan(r)] <= 0.):
                # no solution even with ci_cost = 0.
                #bid_mix = [s.get_last_bid() for s in self.subcampaigns]
                bid_mix = [FIRST_BID for s in self.subcampaigns]
                rev_mix = [s.revenue(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
                cost_mix = [s.cost(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
                roi = np.sum(np.array(rev_mix), axis=1) / np.sum(np.array(cost_mix), axis=1)
                if logger.isEnabledFor(logging.ERROR):
                    logger.error(f'no feasible value found for the revenue of {type(self).__name__};\n'
                                 f'the last row of the optimization matrix contains only NaN values\n{r}\n'
                                 f'even with no constraint on the cost (ci=0.)'
                                 f'use the same bids of the first round\n:{bid_mix}\n')
                # roi = np.sum(np.array(rev_mix), axis=1) / cost_array
                self.ci_cost = ci_cost
                return (bid_mix if not ret_roi_revenue_cost
                        else (bid_mix, r, roi, rev_mix, cost_mix, -1))

            try:
                idx = np.unravel_index(np.nanargmax(r[1:, roi_thr:], axis=None), r[1:, roi_thr:].shape)
            except ValueError:
                # no ROI greater than MIN_ROI
                # make the threashold lower
                for k in range(roi_thr - 1, -1, -1):  #roi_array[:roi_thr]:
                    try:
                        idx = np.nanargmax(r[1:, k])
                    except ValueError:
                        continue
                    # value found
                    idx = idx+1, k
                    if logger.isEnabledFor(logging.ERROR):
                        logger.error(f'no feasible value found for the revenue of {type(self).__name__};\n'
                                     f'even with no constraint on the cost (ci_cost=0.)\n'
                                     f'while ci_rev is: {self.ci_rev}\n'
                                     f'index that maximize the revenue - given the new minimum ROI:\n'
                                     f'{idx}\n'
                                     f'and the roi'
                                     f'\n{roi_array[k]}\n')

                        logger.error(f'the chosen values for {type(self).__name__}:\n'
                                       f'the chosen revenue value:\n{r[idx]}\n'
                                       f'the revenue values were:\n{r}\n'
                                       f'the associated roi value:\n{roi[idx]}\n'
                                       f'the considered ROI constraint:'
                                       f'\n{roi_array[idx[1]]}\n'
                                       f'the chosen bid mix:\n{bid_mix[idx]}\n'
                                       f'the chosen cost mix:\n{cost_mix[idx]}\n')
                    self.ci_cost = ci_cost
                    return (bid_mix[idx] if not ret_roi_revenue_cost
                            else (bid_mix, r, roi, rev_mix, cost_mix, idx))
                if logger.isEnabledFor(logging.ERROR):
                    logger.error(f'something went wrong; no value found for {type(self).__name__}; while it should have been')
            idx = idx[0]+1, idx[1] + roi_thr

            #if logger.isEnabledFor(logging.WARNING):
            #    logger.warning(f'the chosen index:{idx}\n'
            #                   f'the chosen bids:\n{bid_mix[idx]}\n'
            #                   f'for {type(self).__name__}\n'
            #                   f'that maximize the new target roi\n'  #revenue\n'
            #                   f'the chosen revenue\n{r[idx]}\n'
            #                   f'while the roi\n'
            #                   f'{roi[idx]}\n'
            #                   f'the possible revenues were:\n{r[mask]}'
            #                   f'while r was:\n{r}'
            #                   f'and roi was:\n{roi}'
            #                   f'chosen cost_mix:\n{cost_mix[idx]}')

            if logger.isEnabledFor(logging.ERROR):
                logger.error(f'index that maximize the revenue - given a minimum ROI:\n'
                             f'with no constraint on the cost (ci=0.)\n'
                             f'while ci_rev is: {self.ci_rev}\n'
                             f'{idx}\n'
                             f'and the roi:\n{roi_array[roi_thr]}\n')

                #logger.error(f'the last row of the optimization matrix: \n{r[:, idx[1]]}')
                logger.error(f'the revenue values that respect the ROI constraint:'
                            f'\n{r[idx]}\n')
                logger.error(f'the ROI:'
                            f'\n{roi[idx]}\n')
                logger.error(f'the considered ROI constraint:'
                            f'\n{roi_array[idx[1]]}\n')
                logger.error(f'the chosen values for {type(self).__name__}:\n'
                               f'the chosen revenue value:\n{r[idx]}\n'
                               f'the associated roi value:\n{roi[idx]}\n'
                               f'the chosen bid mix:\n{bid_mix[idx]}\n'
                               f'the chosen cost mix:\n{cost_mix[idx]}')

            self.ci_cost = ci_cost
            return (bid_mix[idx] if not ret_roi_revenue_cost
                    else (bid_mix, r, roi, rev_mix, cost_mix, idx))

        # cost_array = np.linspace(0., MAX_EXP, N_COST)
        # roi_array = np.linspace(LOW_ROI, MAX_ROI, N_ROI)
        # roi = (r.T / cost_array).T

        # roi_thr = np.argmax(roi_array >= MIN_ROI)
        try:
            idx = np.unravel_index(np.nanargmax(r[1:, roi_thr:], axis=None), r[1:, roi_thr:].shape)
        except ValueError:
            # no ROI greater than MIN_ROI
            # make the threashold lower
            for k in range(roi_thr - 1, -1, -1):  #roi_array[:roi_thr]:
                try:
                    idx = np.nanargmax(r[1:, k])
                except ValueError:
                    continue
                # value found
                idx = idx+1, k
                if logger.isEnabledFor(logging.ERROR):
                    logger.error(f'no feasible value found for the revenue of {type(self).__name__};\n'
                                 f'with ci_rev:{self.ci_rev}\n'
                                 f'with ci_cost:{self.ci_cost}\n'
                                 f'index that maximize the revenue - given the new minimum ROI:\n'
                                 f'{idx}\n'
                                 f'and the roi'
                                 f'\n{roi_array[k]}\n')

                    logger.error(f'the chosen values for {type(self).__name__}:\n'
                                   f'the chosen revenue value:\n{r[idx]}\n'
                                   f'the revenue values were:\n{r}\n'
                                   f'the associated roi value:\n{roi[idx]}\n'
                                   f'the considered ROI constraint:'
                                   f'\n{roi_array[idx[1]]}\n'
                                   f'the chosen bid mix:\n{bid_mix[idx]}\n'
                                   f'the chosen cost mix:\n{cost_mix[idx]}\n')
                return (bid_mix[idx] if not ret_roi_revenue_cost
                        else (bid_mix, r, roi, rev_mix, cost_mix, idx))
            if logger.isEnabledFor(logging.ERROR):
                logger.error(f'something went wrong; no value found for {type(self).__name__}; while it should have been')
        # value found
        # idx[1] += roi_thr
        idx = idx[0]+1, idx[1] + roi_thr

        #if logger.isEnabledFor(logging.WARNING):
        #    logger.warning(f'the chosen index:{idx}\n'
        #                   f'the chosen bids:\n{bid_mix[idx]}\n'
        #                   f'for {type(self).__name__}\n'
        #                   f'that maximize the new target roi\n'  #revenue\n'
        #                   f'the chosen revenue\n{r[idx]}\n'
        #                   f'while the roi\n'
        #                   f'{roi[idx]}\n'
        #                   f'the possible revenues were:\n{r[mask]}'
        #                   f'while r was:\n{r}'
        #                   f'and roi was:\n{roi}'
        #                   f'chosen cost_mix:\n{cost_mix[idx]}')

        if logger.isEnabledFor(logging.ERROR):
            logger.error(f'index that maximize the revenue - given a minimum ROI:\n'
                           f'{idx}\n'
                           f'and the roi:\n{roi_array[roi_thr]}\n')

            #logger.error(f'the last row of the optimization matrix: \n{r[:, idx[1]]}')
            logger.error(f'the revenue values that respect the ROI constraint:'
                         f'\n{r[idx]}\n'
                         f'with ci_rev:{self.ci_rev}\n'
                         f'with ci_cost:{self.ci_cost}\n'
                        f'\n{r[idx]}\n')
            logger.error(f'the ROI:'
                        f'\n{roi[idx]}\n')
            logger.error(f'the considered ROI constraint:'
                        f'\n{roi_array[idx[1]]}\n')
            logger.error(f'the chosen values for {type(self).__name__}:\n'
                           f'the chosen revenue value:\n{r[idx]}\n'
                           f'the associated roi value:\n{roi[idx]}\n'
                           f'the chosen bid mix:\n{bid_mix[idx]}\n'
                           f'the chosen cost mix:\n{cost_mix[idx]}')

        return (bid_mix[idx] if not ret_roi_revenue_cost
                else (bid_mix, r, roi, rev_mix, cost_mix, idx))

class Safe_Agent_rev(Safe_Agent):
    def __init__(self, subcampaigns_number):  # , delta_bid=3.):
        super().__init__(subcampaigns_number)
        self._opt_func = safe_optimize_rev
        self.__str__ = self.__str__

    def __str__(self):
        return f'safe optimization ucb agent on rev'


class Safe_Agent_delta(Safe_Agent):
    def __init__(self, subcampaigns_number):  # , delta_bid=3.):
        super().__init__(subcampaigns_number)
        self._opt_func = safe_optimize_delta
        self.bid_choice = self.bid_choice
        self.__str__ = self.__str__

    def __str__(self):
        return f'safe optimization delta agent'

class Safe_Agent_sample_cost(Safe_Agent):
    def __init__(self, subcampaigns_number):  # , delta_bid=3.):
        super().__init__(subcampaigns_number)
        self._optimize = self._optimize
        self._opt_func = safe_optimize_roi
        self.__str__ = self.__str__
        self.ci_cost = 0. # sampling on cost

    def __str__(self):
        return f'safe optimization sample cost agent'

    def _optimize(self):
        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'enter {type(self).__name__} _optimize')
        bid = np.linspace(0., MAX_BID, SAMPLES_INVERSE)

        #rev = [(mean, var) for mean, var in s.revenue(bid, return_var=True, sample_y=False) for s in self.subcampaigns]
        mean_rev = np.empty((len(self.subcampaigns), N_BID, 1))
        sigma_rev = np.empty((len(self.subcampaigns), N_BID, 1))
        sample_cost = np.empty((len(self.subcampaigns), N_BID, 1))

        for i, s in enumerate(self.subcampaigns):
          #  for b in range(N_BID):
            mean_rev[i, :], sigma_rev[i, :] = s.revenue(bid, return_var=True, sample_y=False)
            sample_cost[i, :] = s.cost(bid, sample_y=True)
        sigma_rev = np.sqrt(sigma_rev)

        ci_rev = self.ci_rev  # 1.  # 0.  # 1. #0.75  # 1.96  #0.5  # 1.  # 1.645 # 1.96

        scl = [bid_max_lb_revenue_given_cost(bid, mean_rev[s] - ci_rev*sigma_rev[s], sample_cost[s],
                                             mean_rev[s] + ci_rev*sigma_rev[s], sample_cost[s])
                for s, _ in enumerate(self.subcampaigns)
                ]

        scl = np.array(scl)

        return self._opt_func(scl)

class Safe_Agent_lb_cost(Safe_Agent):
    def __init__(self, subcampaigns_number):  # , delta_bid=3.):
        super().__init__(subcampaigns_number)
        self._optimize = self._optimize
        self._opt_func = safe_optimize_roi
        self.__str__ = self.__str__
        self.ci_cost = 1.0

    def __str__(self):
        return f'safe optimization lower bound on cost agent'

    def _optimize(self):
        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'enter {type(self).__name__} _optimize')
        bid = np.linspace(0., MAX_BID, SAMPLES_INVERSE)

        #rev = [(mean, var) for mean, var in s.revenue(bid, return_var=True, sample_y=False) for s in self.subcampaigns]
        mean_rev = np.empty((len(self.subcampaigns), N_BID, 1))
        sigma_rev = np.empty((len(self.subcampaigns), N_BID, 1))
        mean_cost = np.empty((len(self.subcampaigns), N_BID, 1))
        sigma_cost = np.empty((len(self.subcampaigns), N_BID, 1))

        for i, s in enumerate(self.subcampaigns):
            mean_rev[i, :], sigma_rev[i, :] = s.revenue(bid, return_var=True, sample_y=False)
            mean_cost[i, :], sigma_cost[i, :] = s.cost(bid, return_var=True, sample_y=False)

        sigma_rev = np.sqrt(sigma_rev)
        sigma_cost = np.sqrt(sigma_cost)

        ci_rev = self.ci_rev  # 1.  # 0.  # 1. #0.75  # 1.96  #0.5  # 1.  # 1.645 # 1.96
        ci_cost = self.ci_cost  # 1.  # 0.  # 1. #0.75  # 1.96  #0.5  # 1.  # 1.645 # 1.96

        # use lower-bound for the cost
        scl = [bid_max_lb_revenue_given_cost(bid, mean_rev[s] - ci_rev*sigma_rev[s], mean_cost[s] - ci_cost*sigma_cost[s],
                                             mean_rev[s] + ci_rev*sigma_rev[s], mean_cost[s] - ci_cost*sigma_cost[s])
                for s, _ in enumerate(self.subcampaigns)
                # mean_rev, var_rev = s.revenue(bid, return_var=True, sample_y=False)
                # mean_cost, var_cost = s.cost(bid, return_var=True, sample_y=False)
                ]

        scl = np.array(scl)

        return self._opt_func(scl)


class Safe_eps_greedy_Agent(Safe_Agent):
    def __init__(self, subcampaigns_number):  # , delta_bid=3.):
        super().__init__(subcampaigns_number)
        self._opt_func = safe_optimize_delta
        self.__str__ = self.__str__
        self._optimize = self._optimize
        self.bid_choice = self.bid_choice
        self._rng = np.random.default_rng()

    def __str__(self):
        return f'safe ε-greedy agent'

    def _optimize(self):
        val = self._rng.binomial(1, 0.05)
        if val:
            self._opt_func = safe_optimize_delta
        else:
            self._opt_func = safe_optimize_roi
        self.val = val
        return super()._optimize()


    def bid_choice(self, ret_roi_revenue_cost=False):
        if logger.isEnabledFor(logging.WARNING):
            logger.info('entering bid_choice')
        r, bid_mix, rev_mix, cost_mix = self._optimize()

        # r = r[:, -1] # the last ROI

        # cost_array = np.linspace(0., MAX_EXP, N_COST)
        # roi = r / cost_array

        if np.all(np.isnan(r[1:])) or np.all(r[~np.isnan(r)] <= 0.):
            

            #bid_mix = [s.get_last_bid() for s in self.subcampaigns]
            bid_mix = [FIRST_BID for s in self.subcampaigns]
            rev_mix = [s.revenue(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
            cost_mix = [s.cost(bid_mix[i]) for i, s in enumerate(self.subcampaigns)]
            roi = np.sum(np.array(rev_mix), axis=1) / np.sum(np.array(cost_mix), axis=1)
            if logger.isEnabledFor(logging.ERROR):
                logger.error(f'no feasible value found for the revenue of {type(self).__name__};\n'
                             f'the last row of the optimization matrix contains only NaN values\n{r}\n'
                             f'use the same bids of the first round\n:{bid_mix}\n')
            # roi = np.sum(np.array(rev_mix), axis=1) / cost_array
            return (bid_mix if not ret_roi_revenue_cost
                    else (bid_mix, r, roi, rev_mix, cost_mix, -1))

        cost_array = np.linspace(0., MAX_EXP, N_COST)
        roi_array = np.linspace(LOW_ROI, MAX_ROI, N_ROI)
        roi = (r.T / cost_array).T

        roi_thr = np.argmax(roi_array >= MIN_ROI)
        try:
            if self.val:
                idx = np.unravel_index(np.nanargmax(r[1:, roi_thr:] - rev_mix[1:, roi_thr:], axis=None), (r[1:, roi_thr:]- rev_mix[1:, roi_thr:]).shape)
            else:
                idx = np.unravel_index(np.nanargmax(r[1:, roi_thr:], axis=None), r[1:, roi_thr:].shape)
        except ValueError:
            # no ROI greater than MIN_ROI
            # make the threashold lower
            for k in range(roi_thr - 1, -1, -1):  #roi_array[:roi_thr]:
                try:
                    if self.val:
                        idx = np.nanargmax(r[1:, k] - rev_mix[1:, k])
                    else:
                        idx = np.nanargmax(r[1:, k])
                except ValueError:
                    continue
                # value found
                idx = idx+1, k
                if logger.isEnabledFor(logging.ERROR):
                    if self.val:
                        logger.error(f'no feasible value found for {type(self).__name__};\n'
                                     f'index that maximize the variance - given the new minimum ROI:\n'
                                     f'{idx}\n'
                                     f'and the roi'
                                     f'\n{roi_array[k]}\n')
                    else:
                        logger.error(f'no feasible value found for the revenue of {type(self).__name__};\n'
                                     f'index that maximize the revenue - given the new minimum ROI:\n'
                                     f'{idx}\n'
                                     f'and the roi'
                                     f'\n{roi_array[k]}\n')

                    logger.error(f'the chosen values for {type(self).__name__}:\n'
                                   f'the chosen revenue value:\n{r[idx]}\n'
                                   f'the revenue values were:\n{r}\n'
                                   f'the associated roi value:\n{roi[idx]}\n'
                                   f'the considered ROI constraint:'
                                   f'\n{roi_array[idx[1]]}\n'
                                   f'the chosen bid mix:\n{bid_mix[idx]}\n'
                                   f'the chosen cost mix:\n{cost_mix[idx]}\n')
                return (bid_mix[idx] if not ret_roi_revenue_cost
                        else (bid_mix, r, roi, rev_mix, cost_mix, idx))
            if logger.isEnabledFor(logging.ERROR):
                logger.error(f'something went wrong; no value found for {type(self).__name__}; while it should have been')
        # value found
        # idx[1] += roi_thr
        idx = idx[0]+1, idx[1] + roi_thr

        #if logger.isEnabledFor(logging.WARNING):
        #    logger.warning(f'the chosen index:{idx}\n'
        #                   f'the chosen bids:\n{bid_mix[idx]}\n'
        #                   f'for {type(self).__name__}\n'
        #                   f'that maximize the new target roi\n'  #revenue\n'
        #                   f'the chosen revenue\n{r[idx]}\n'
        #                   f'while the roi\n'
        #                   f'{roi[idx]}\n'
        #                   f'the possible revenues were:\n{r[mask]}'
        #                   f'while r was:\n{r}'
        #                   f'and roi was:\n{roi}'
        #                   f'chosen cost_mix:\n{cost_mix[idx]}')

        if logger.isEnabledFor(logging.ERROR):
            if self.val:
                logger.error(f'In this round {type(self).__name__} maximizes the variance')
                logger.error(f'index that maximize the variance (the difference between upper and lower bound for the revenue):\n'
                             f'{idx}\n'
                             f'and the roi:\n{roi_array[roi_thr]}\n')
            else:        
                logger.error(f'In this round {type(self).__name__} maximizes the revenue')
                logger.error(f'index that maximize the revenue:\n'
                           f'{idx}\n'
                           f'and the roi:\n{roi_array[roi_thr]}\n')

            #logger.error(f'the last row of the optimization matrix: \n{r[:, idx[1]]}')
            logger.error(f'the revenue values that respect the ROI constraint:'
                        f'\n{r[idx]}\n')
            logger.error(f'the ROI:'
                        f'\n{roi[idx]}\n')
            logger.error(f'the considered ROI constraint:'
                        f'\n{roi_array[idx[1]]}\n')
            logger.error(f'the chosen values for {type(self).__name__}:\n'
                           f'the chosen revenue value:\n{r[idx]}\n'
                           f'the associated roi value:\n{roi[idx]}\n'
                           f'the chosen bid mix:\n{bid_mix[idx]}\n'
                           f'the chosen cost mix:\n{cost_mix[idx]}')

        return (bid_mix[idx] if not ret_roi_revenue_cost
                else (bid_mix, r, roi, rev_mix, cost_mix, idx))

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    num_threads = 1
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
    )

    #start = time.time()
    print(tf.config.threading.get_inter_op_parallelism_threads())
    tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
    )
    print(tf.config.threading.get_intra_op_parallelism_threads())

    # campaign_a = [sc.Subcampaign(59, 1.3, 25, 1.0),
    #               sc.Subcampaign(20, 2.4, 30, 2.2),
    #               sc.Subcampaign(20, 1.3, 35, 1.2)]
    import time
    campaign_a = [sc.Subcampaign(64, 0.3, 265, 0.2),
                  sc.Subcampaign(70, 0.4, 355, 0.2),
                  sc.Subcampaign(70, 0.3, 326, 0.1),
                  sc.Subcampaign(50, 0.3, 185, 0.2),
                  sc.Subcampaign(55, 0.4, 208, 0.1),
                  sc.Subcampaign(120, 0.4, 724, 0.2),
                  sc.Subcampaign(100, 0.3, 669, 0.25),
                  sc.Subcampaign(90, 0.34, 595, 0.15),
                  sc.Subcampaign(95, 0.38, 616, 0.19),
                  sc.Subcampaign(110, 0.4, 675, 0.12)]
    campaign_f = [sc.Subcampaign(160, 0.65, 847, .45),
                  sc.Subcampaign(170, .62, 895, 0.42),
                  sc.Subcampaign(170, 0.69, 886, 0.49)] #,
    # campaign_g = [sc.Subcampaign(60, 0.65, 497, .45, snr=50),
    #               sc.Subcampaign(77, .62, 665, 0.42, snr=50),
    #               sc.Subcampaign(75, .67, 573, 0.43, snr=50),
    #               sc.Subcampaign(65, .68, 503, 0.47, snr=50),
    #               sc.Subcampaign(70, 0.69, 536, 0.49, snr=50)] #,

    campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=50),
                  sc.Subcampaign(77, .62, 565, 0.48, snr=50),
                  sc.Subcampaign(75, .67, 573, 0.43, snr=50),
                  sc.Subcampaign(65, .68, 503, 0.47, snr=50),
                  sc.Subcampaign(70, 0.69, 536, 0.40, snr=50)] #,
                  # sc.Subcampaign(150, 0.57, 845, 0.47),
                  # sc.Subcampaign(155, 0.59, 848, 0.49)]
    # campaign_g = [sc.Subcampaign(60, 0.65, 497, .35, snr=50),
    #               sc.Subcampaign(77, .62, 565, 0.38, snr=50),
    #               sc.Subcampaign(75, .67, 573, 0.39, snr=50),
    #               sc.Subcampaign(65, .68, 503, 0.37, snr=50),
    #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=50)] #,

    #   sc.Subcampaign(148.0, 1.0377934628741592, 507.0, 0.44027797731395735, snr=30),
    #   sc.Subcampaign(145.0, 0.6551456696651985, 599.0, 0.3396719957535973, snr=33),
    #   sc.Subcampaign(77.0, 1.0485820099112553, 483.0, 0.34777072779804974, snr=31),
    #   sc.Subcampaign(144.0, 0.49874734951433286, 520.0, 0.49292800934489955, snr=31),
    #   sc.Subcampaign(110.0, 0.5739913810818937, 445.0, 0.37589871130889635, snr=30)
    # noise cost 0: 2.8360804317842248
    # noise rev 0: 13.158048836966795
    # noise cost 1: 2.3827528382392344
    # noise rev 1: 11.574172771261757
    # noise cost 2: 1.3083519534997488
    # noise rev 2: 11.70277047352294
    # noise cost 3: 3.231984818435723
    # noise rev 3: 11.706289473299163
    # noise cost 4: 2.6637152009028995
    # noise rev 4: 11.930595053542682
    # ROI >= 10

    rnd_campaign_011 =  [
                        sc.Subcampaign(148.0, 1.0377934628741592, 507.0, 0.44027797731395735, snr=30),
                        sc.Subcampaign(145.0, 0.6551456696651985, 599.0, 0.3396719957535973, snr=33),
                        sc.Subcampaign(77.0, 1.0485820099112553, 483.0, 0.34777072779804974, snr=31),
                        sc.Subcampaign(144.0, 0.49874734951433286, 520.0, 0.49292800934489955, snr=31),
                        sc.Subcampaign(110.0, 0.5739913810818937, 445.0, 0.37589871130889635, snr=30)
                       ]


    # [
    # sc.Subcampaign(116.0, 0.49023808983119915, 559.0, 0.45957526625947004, snr=35),
    # sc.Subcampaign(145.0, 0.30520020069613246, 325.0, 0.26532994162420004, snr=33),
    # sc.Subcampaign(148.0, 0.3500111080251961, 344.0, 0.26173224352782093, snr=33),
    # sc.Subcampaign(136.0, 0.28176893303144557, 327.0, 0.2114547059187422, snr=31),
    # sc.Subcampaign(66.0, 1.0268515219909764, 580.0, 1.0177294822744638, snr=35)
    # ]
    # noise cost 0: 1.6499795015575118
    # noise rev 0: 8.07793643366628
    # noise cost 1: 2.848808352039576
    # noise rev 1: 6.5062601289358515
    # noise cost 2: 2.8452838738168875
    # noise rev 2: 6.898111796323972
    # noise cost 3: 3.4013746833748266
    # noise rev 3: 8.445330865231421
    # noise cost 4: 0.7149294340016205
    # noise rev 4: 6.310145465005205
    rnd_campaign_021 = [

                    sc.Subcampaign(116.0, 0.49023808983119915, 559.0, 0.45957526625947004, snr=35),
                    sc.Subcampaign(145.0, 0.30520020069613246, 325.0, 0.26532994162420004, snr=33),
                    sc.Subcampaign(148.0, 0.3500111080251961, 344.0, 0.26173224352782093, snr=33),
                    sc.Subcampaign(136.0, 0.28176893303144557, 327.0, 0.2114547059187422, snr=31),
                    sc.Subcampaign(66.0, 1.0268515219909764, 580.0, 1.0177294822744638, snr=35)
                    ]

    # [
    # sc.Subcampaign(142.0, 0.9725389734513807, 587.0, 0.9604834530657442, snr=31),
    # sc.Subcampaign(78.0, 0.8044043407707389, 448.0, 0.22918637045900483, snr=31),
    # sc.Subcampaign(50.0, 0.2821595248908533, 480.0, 0.2663193928010516, snr=30),
    # sc.Subcampaign(102.0, 1.0469733476524765, 479.0, 0.4961937553590189, snr=35),
    # sc.Subcampaign(102.0, 0.8693305044075821, 486.0, 0.28373421923392844, snr=31)
    # ]
    # noise cost 0: 2.5023797509453707
    # noise rev 0: 10.405119773727183
    # noise cost 1: 1.4942810694482864
    # noise rev 1: 11.478975445367047
    # noise cost 2: 1.402832719136793
    # noise rev 2: 13.567189861770549
    # noise cost 3: 1.0943720895758995
    # noise rev 3: 6.792301364795966
    # noise cost 4: 1.8911953948196434
    # noise rev 4: 12.143689819160766
    rnd_campaign_031 = [
                    sc.Subcampaign(142.0, 0.9725389734513807, 587.0, 0.9604834530657442, snr=31),
                    sc.Subcampaign(78.0, 0.8044043407707389, 448.0, 0.22918637045900483, snr=31),
                    sc.Subcampaign(50.0, 0.2821595248908533, 480.0, 0.2663193928010516, snr=30),
                    sc.Subcampaign(102.0, 1.0469733476524765, 479.0, 0.4961937553590189, snr=35),
                    sc.Subcampaign(102.0, 0.8693305044075821, 486.0, 0.28373421923392844, snr=31)
                    ]

      #           sc.Subcampaign(83.0, 0.9390275741377971, 530.0, 0.35653961965235176, snr=34),
      #           sc.Subcampaign(97.0, 0.8565310891376972, 417.0, 0.6893952328071604, snr=30),
      #           sc.Subcampaign(72.0, 0.4845001118567374, 548.0, 0.29997157236918714, snr=35),
      #           sc.Subcampaign(100.0, 0.6618767352251798, 571.0, 0.5709120360333635, snr=34),
      #           sc.Subcampaign(96.0, 0.24623017989617788, 550.0, 0.24553470046274206, snr=31)
      #   noise cost 0: 1.052537761979174
      #   noise rev 0: 9.051996755602742
      #   noise cost 1: 2.0309038431474433
      #   noise rev 1: 9.508677060463024
      #   noise cost 2: 1.0271687490408083
      #   noise rev 2: 8.573505008292507
      #   noise cost 3: 1.4594531211207673
      #   noise rev 3: 8.738346068981958
      #   noise cost 4: 2.4408327998747
      #   noise rev 4: 13.988380646734504
      #   ROI >= 10. or 12.
    rnd_campaign_01 = [
                sc.Subcampaign(83.0, 0.9390275741377971, 530.0, 0.35653961965235176, snr=34),
                sc.Subcampaign(97.0, 0.8565310891376972, 417.0, 0.6893952328071604, snr=30),
                sc.Subcampaign(72.0, 0.4845001118567374, 548.0, 0.29997157236918714, snr=35),
                sc.Subcampaign(100.0, 0.6618767352251798, 571.0, 0.5709120360333635, snr=34),
                sc.Subcampaign(96.0, 0.24623017989617788, 550.0, 0.24553470046274206, snr=31)
                ]

#                 sc.Subcampaign(83.0, 0.22405720652229322, 597.0, 0.2022227936494456, snr=30),
#                 sc.Subcampaign(98.0, 0.8499901193552486, 682.0, 0.5208800234776618, snr=31),
#                 sc.Subcampaign(56.0, 0.7267762997014247, 698.0, 0.3670485687338533, snr=31),
#                 sc.Subcampaign(60.0, 0.559426504706663, 456.0, 0.39340047292135694, snr=32),
#                 sc.Subcampaign(51.0, 0.7831724596643996, 444.0, 0.6895966545162417, snr=32)
#         noise cost 0: 2.391688637572843
#         noise rev 0: 17.370646759573905
#         noise cost 1: 1.834715198886506
#         noise rev 1: 15.132168560115462
#         noise cost 2: 1.116293518680102
#         noise rev 2: 16.752028796346515
#         noise cost 3: 1.1629130940535077
#         noise rev 3: 9.626443994149284
#         noise cost 4: 0.8802366586767133
#         noise rev 4: 8.041214809229146
#         ROI >=14.
    rnd_campaign_02 = [
                sc.Subcampaign(83.0, 0.22405720652229322, 597.0, 0.2022227936494456, snr=30),
                sc.Subcampaign(98.0, 0.8499901193552486, 682.0, 0.5208800234776618, snr=31),
                sc.Subcampaign(56.0, 0.7267762997014247, 698.0, 0.3670485687338533, snr=31),
                sc.Subcampaign(60.0, 0.559426504706663, 456.0, 0.39340047292135694, snr=32),
                sc.Subcampaign(51.0, 0.7831724596643996, 444.0, 0.6895966545162417, snr=32)
                ]

#                 sc.Subcampaign(97.0, 0.2254717323719329, 570.0, 0.21712317683804871, snr=30),
#                 sc.Subcampaign(78.0, 0.6809006218688995, 514.0, 0.638902808265124, snr=31),
#                 sc.Subcampaign(53.0, 1.0518056176361688, 426.0, 0.6940683038810456, snr=34),
#                 sc.Subcampaign(80.0, 0.4129454844360104, 469.0, 0.39195310889540247, snr=32),
#                 sc.Subcampaign(82.0, 0.9188239551808393, 548.0, 0.34500566310456804, snr=30)
#         noise cost 0: 2.7933318460913106
#         noise rev 0: 16.475872439014033
#         noise cost 1: 1.5921802697614682
#         noise rev 1: 10.723778001484668
#         noise cost 2: 0.6365712276896207
#         noise rev 2: 6.114230875627782
#         noise cost 3: 1.672289879051099
#         noise rev 3: 9.908077332799381
#         noise cost 4: 1.6645033179533408
#         noise rev 4: 14.918007073945152
#         ROI >= 10.5
    rnd_campaign_03 = [
                sc.Subcampaign(97.0, 0.2254717323719329, 570.0, 0.21712317683804871, snr=30),
                sc.Subcampaign(78.0, 0.6809006218688995, 514.0, 0.638902808265124, snr=31),
                sc.Subcampaign(53.0, 1.0518056176361688, 426.0, 0.6940683038810456, snr=34),
                sc.Subcampaign(80.0, 0.4129454844360104, 469.0, 0.39195310889540247, snr=32),
                sc.Subcampaign(82.0, 0.9188239551808393, 548.0, 0.34500566310456804, snr=30)
                ]

#                 sc.Subcampaign(62.0, 0.460443097667253, 487.0, 0.34862416791066253, snr=32),
#                 sc.Subcampaign(79.0, 1.0217507506678771, 494.0, 0.4244629979053927, snr=30),
#                 sc.Subcampaign(76.0, 0.5159200609033661, 467.0, 0.32645838278958483, snr=33),
#                 sc.Subcampaign(69.0, 0.8948304325148888, 684.0, 0.7226499482261013, snr=34),
#                 sc.Subcampaign(99.0, 1.0568307954024765, 494.0, 0.26571147363891007, snr=34)
#         noise cost 0: 1.2649886220831275
#         noise rev 0: 10.512079296630048
#         noise cost 1: 1.5254700525111253
#         noise rev 1: 12.924425391861316
#         noise cost 2: 1.3429196771862904
#         noise rev 2: 9.081780930867424
#         noise cost 3: 0.8942772030668069
#         noise rev 3: 9.673210440580613
#         noise cost 4: 1.1862455613178984
#         noise rev 4: 8.81247994676349
#         ROI >= 12.
    rnd_campaign_04 = [
                sc.Subcampaign(62.0, 0.460443097667253, 487.0, 0.34862416791066253, snr=32),
                sc.Subcampaign(79.0, 1.0217507506678771, 494.0, 0.4244629979053927, snr=30),
                sc.Subcampaign(76.0, 0.5159200609033661, 467.0, 0.32645838278958483, snr=33),
                sc.Subcampaign(69.0, 0.8948304325148888, 684.0, 0.7226499482261013, snr=34),
                sc.Subcampaign(99.0, 1.0568307954024765, 494.0, 0.26571147363891007, snr=34)
                ]


       #          sc.Subcampaign(52.0, 0.7236376036521779, 525.0, 0.2581539194710056, snr=31),
       #          sc.Subcampaign(87.0, 0.8347532352531297, 643.0, 0.607729693610427, snr=35),
       #          sc.Subcampaign(68.0, 1.0545161819092699, 455.0, 0.39019889273979436, snr=33),
       #          sc.Subcampaign(99.0, 1.0715069147863183, 440.0, 0.7409946097511, snr=33),
       #          sc.Subcampaign(94.0, 0.9434443599935285, 600.0, 0.3880212001600485, snr=33)
       #  noise cost 0: 1.0382372825242678
       #  noise rev 0: 13.27543911770621
       #  noise cost 1: 1.035595972838408
       #  noise rev 1: 8.603141236567959
       #  noise cost 2: 0.9152154198839922
       #  noise rev 2: 8.57452471140731
       #  noise cost 3: 1.3218160270008468
       #  noise rev 3: 6.916137860073078
       #  noise cost 4: 1.3345914074676068
       #  noise rev 4: 11.319411525186174
       # ROI >= 14 with R101 1400
    rnd_campaign_05 = [
                sc.Subcampaign(52.0, 0.7236376036521779, 525.0, 0.2581539194710056, snr=31),
                sc.Subcampaign(87.0, 0.8347532352531297, 643.0, 0.607729693610427, snr=35),
                sc.Subcampaign(68.0, 1.0545161819092699, 455.0, 0.39019889273979436, snr=33),
                sc.Subcampaign(99.0, 1.0715069147863183, 440.0, 0.7409946097511, snr=33),
                sc.Subcampaign(94.0, 0.9434443599935285, 600.0, 0.3880212001600485, snr=33)
                ]

#                 sc.Subcampaign(71.0, 0.8750247925943013, 617.0, 0.8440732817094825, snr=31),
#                 sc.Subcampaign(53.0, 0.8411068818594563, 518.0, 0.6772664432914361, snr=33),
#                 sc.Subcampaign(87.0, 1.0703614987489847, 547.0, 0.8667712665875702, snr=32),
#                 sc.Subcampaign(98.0, 0.6310736950229558, 567.0, 0.2521616349302094, snr=31),
#                 sc.Subcampaign(59.0, 0.28867007783272997, 576.0, 0.24763783882935328, snr=32)
#         noise cost 0: 1.3126839542858937
#         noise rev 0: 11.585606645864749
#         noise cost 1: 0.7916942053910707
#         noise rev 1: 8.41485596756745
#         noise cost 2: 1.304034180201852
#         noise rev 2: 9.050632923219277
#         noise cost 3: 2.0529724164451726
#         noise rev 3: 14.37706218287495
#         noise cost 4: 1.3108599538271641
#         noise rev 4: 13.043970921305917
#   ROI>11. or 11.5
    rnd_campaign_06 = [
                sc.Subcampaign(71.0, 0.8750247925943013, 617.0, 0.8440732817094825, snr=31),
                sc.Subcampaign(53.0, 0.8411068818594563, 518.0, 0.6772664432914361, snr=33),
                sc.Subcampaign(87.0, 1.0703614987489847, 547.0, 0.8667712665875702, snr=32),
                sc.Subcampaign(98.0, 0.6310736950229558, 567.0, 0.2521616349302094, snr=31),
                sc.Subcampaign(59.0, 0.28867007783272997, 576.0, 0.24763783882935328, snr=32)
                ]


#                 sc.Subcampaign(77.0, 0.8109246960855101, 409.0, 0.5079367092300731, snr=31),
#                 sc.Subcampaign(78.0, 0.2469886328248737, 592.0, 0.23086355578873, snr=31),
#                 sc.Subcampaign(91.0, 0.7744431310645019, 628.0, 0.5712510728701953, snr=35),
#                 sc.Subcampaign(50.0, 0.5162810922290018, 613.0, 0.359321181934275, snr=34),
#                 sc.Subcampaign(71.0, 0.3794531987004781, 513.0, 0.3077863622385426, snr=35)
#         noise cost 0: 1.4702530998654688
#         noise rev 0: 9.136088501551532
#         noise cost 1: 1.9824893640410592
#         noise rev 1: 15.157180860242853
#         noise cost 2: 1.1168782500459453
#         noise rev 2: 8.563985546834195
#         noise cost 3: 0.7872723082789037
#         noise rev 3: 10.455214694677672
#         noise cost 4: 1.0685405517885862
#         noise rev 4: 7.9960531730263575
#   ROI>=11.5 or 12.
    rnd_campaign_07 = [
                sc.Subcampaign(77.0, 0.8109246960855101, 409.0, 0.5079367092300731, snr=31),
                sc.Subcampaign(78.0, 0.2469886328248737, 592.0, 0.23086355578873, snr=31),
                sc.Subcampaign(91.0, 0.7744431310645019, 628.0, 0.5712510728701953, snr=35),
                sc.Subcampaign(50.0, 0.5162810922290018, 613.0, 0.359321181934275, snr=34),
                sc.Subcampaign(71.0, 0.3794531987004781, 513.0, 0.3077863622385426, snr=35)
                ]

#                 sc.Subcampaign(67.0, 0.6715707414871959, 602.0, 0.3266023723522325, snr=30),
#                 sc.Subcampaign(80.0, 0.7751757702430537, 605.0, 0.2658241604853398, snr=32),
#                 sc.Subcampaign(99.0, 0.44063783263446604, 618.0, 0.2013497671860729, snr=31),
#                 sc.Subcampaign(77.0, 0.31016111269832836, 505.0, 0.21944833868391506, snr=31),
#                 sc.Subcampaign(99.0, 0.4059724156582363, 588.0, 0.2913060845388664, snr=32)
#         noise cost 0: 1.5419719600895856
#         noise rev 0: 16.535614599436503
#         noise cost 1: 1.386410621090635
#         noise rev 1: 13.586383198552335
#         noise cost 2: 2.2894905861318913
#         noise rev 2: 16.03234527287414
#         noise cost 3: 1.9000068585160585
#         noise rev 3: 12.996132353361428
#         noise cost 4: 2.0767656872154405
#         noise rev 4: 13.047901411614037
# ROI>=13. Maxrev 1350
    rnd_campaign_08 = [
                sc.Subcampaign(67.0, 0.6715707414871959, 602.0, 0.3266023723522325, snr=30),
                sc.Subcampaign(80.0, 0.7751757702430537, 605.0, 0.2658241604853398, snr=32),
                sc.Subcampaign(99.0, 0.44063783263446604, 618.0, 0.2013497671860729, snr=31),
                sc.Subcampaign(77.0, 0.31016111269832836, 505.0, 0.21944833868391506, snr=31),
                sc.Subcampaign(99.0, 0.4059724156582363, 588.0, 0.2913060845388664, snr=32)
                ]

#                 sc.Subcampaign(53.0, 0.6181790644241923, 486.0, 0.4181144656709901, snr=35),
#                 sc.Subcampaign(82.0, 0.8639181969833853, 684.0, 0.3300544004489604, snr=34),
#                 sc.Subcampaign(58.0, 0.6695803386210537, 547.0, 0.5291694460820753, snr=33),
#                 sc.Subcampaign(96.0, 0.8661364638643831, 419.0, 0.7291640886632436, snr=35),
#                 sc.Subcampaign(100.0, 0.8319412598597624, 453.0, 0.67917291744797, snr=30)
#         noise cost 0: 0.7052673189388399
#         noise rev 0: 7.17332439563639
#         noise cost 1: 1.0792575077034179
#         noise rev 1: 11.834582267235822
#         noise cost 2: 0.9459731857323016
#         noise rev 2: 9.599118910805634
#         noise cost 3: 1.1248657093307093
#         noise rev 3: 5.2634428273400005
#         noise cost 4: 2.119760316683622
#         noise rev 4: 10.384492750720524
#   ROI>=13 or 13.5
    rnd_campaign_09 = [
                sc.Subcampaign(53.0, 0.6181790644241923, 486.0, 0.4181144656709901, snr=35),
                sc.Subcampaign(82.0, 0.8639181969833853, 684.0, 0.3300544004489604, snr=34),
                sc.Subcampaign(58.0, 0.6695803386210537, 547.0, 0.5291694460820753, snr=33),
                sc.Subcampaign(96.0, 0.8661364638643831, 419.0, 0.7291640886632436, snr=35),
                sc.Subcampaign(100.0, 0.8319412598597624, 453.0, 0.67917291744797, snr=30)
                ]

#                 sc.Subcampaign(51.0, 1.049310026718121, 617.0, 0.2051759576328354, snr=30),
#                 sc.Subcampaign(86.0, 0.7797153486111319, 520.0, 0.539937043644524, snr=35),
#                 sc.Subcampaign(93.0, 0.23347185518099078, 422.0, 0.217994206233266, snr=30),
#                 sc.Subcampaign(61.0, 0.5780205490681166, 559.0, 0.49006423739237187, snr=30),
#                 sc.Subcampaign(84.0, 0.562291455390848, 457.0, 0.22447254161352354, snr=30)
#         noise cost 0: 0.9719732402639386
#         noise rev 0: 17.92921408590325
#         noise cost 1: 1.0526735068232538
#         noise rev 1: 7.207958956291181
#         noise cost 2: 2.668505313654968
#         noise rev 2: 12.19318711170443
#         noise cost 3: 1.4740476542752188
#         noise rev 3: 14.140717160855376
#         noise cost 4: 2.0465678730719907
#         noise rev 4: 13.166241468628666
#   ROI>=14.0  Max_rev 1400
    rnd_campaign_10 = [
                sc.Subcampaign(51.0, 1.049310026718121, 617.0, 0.2051759576328354, snr=30),
                sc.Subcampaign(86.0, 0.7797153486111319, 520.0, 0.539937043644524, snr=35),
                sc.Subcampaign(93.0, 0.23347185518099078, 422.0, 0.217994206233266, snr=30),
                sc.Subcampaign(61.0, 0.5780205490681166, 559.0, 0.49006423739237187, snr=30),
                sc.Subcampaign(84.0, 0.562291455390848, 457.0, 0.22447254161352354, snr=30)
                ]




    campaign = rnd_campaign_09



    # a = Agent(len(campaign_a))
    # b = TS_Agent(len(campaign_a))
    c = Clairvoyant_Agent(campaign)
    # a = Agent(len(campaign_a))
    # b = TS_Agent(len(campaign_a))
    # e = TS_Conservative_Agent_1(len(campaign_a))
    # f = TS_Conservative_Agent_5(len(campaign_a))
    # g = TS_Conservative_Agent_10(len(campaign_a))
    # n = Safe_Agent(len(campaign_a))
    import matplotlib.pyplot as plt

    def _build_another_roi_mask():
        rev_array = np.linspace(0., MAX_REV, N_REV)
        cost_array = np.linspace(0., MAX_EXP, N_COST)
        roi_mask = np.empty((N_REV,N_COST))

        max_rev = 0.
        max_rev_idx = (0, 0)

        for i, rev in enumerate(rev_array):
            for j, cost in enumerate(cost_array):
                if ~np.isnan(rev/cost) & (rev/cost >= MIN_ROI):
                    roi_mask[i,j] = rev/cost
                    max_rev = rev if rev > max_rev else max_rev
                    max_rev_idx = (i, j) if rev > max_rev else max_rev_idx
                else:
                    roi_mask[i,j] = np.nan
        print(roi_mask[(~np.isnan(roi_mask)) & ~np.isinf(roi_mask)])
        print('max_rev', max_rev)
        print('max_rev_idx', max_rev_idx)
        return roi_mask
    import sys
    np.set_printoptions(threshold=sys.maxsize)

    # for i in range(len(campaign_g)):
    #     plt.plot(np.linspace(0.0, 2.0, 201), campaign_g[i].cost(np.linspace(0.0, 2.0, 201), noise=False))  #, xlabel='bid', ylabel='cost', title=f'sc_{i}')
    #     x_label = 'bid'
    #     y_label = 'cost'
    #     plt.xlabel(x_label)
    #     plt.ylabel(y_label)
    #     plt.title(f'sc_{i} cost')
    #     plt.gca().legend()
    #     plt.show()
    #     plt.plot(np.linspace(0.0, 2.0, 201), campaign_g[i].revenue(np.linspace(0.0, 2.0, 201), noise=False))
    #     ylabel='revenue'
    #     plt.xlabel(x_label)
    #     plt.ylabel(y_label)
    #     plt.title(f'sc_{i} revenue')
    #     plt.gca().legend()
    #     plt.show()

     
    _, r, roi, _, cost_mix,idx = c.bid_choice(ret_roi_revenue_cost=True) 
    
    cost_array = np.linspace(0., MAX_EXP, N_COST)

    plt.plot(np.linspace(0., MAX_EXP, N_COST), roi[0,:])
    plt.show()
    plt.plot(np.linspace(0., MAX_EXP, N_COST), r[0,:]/cost_array)
    plt.show()
    print('rev 0.01:', campaign_f[0].revenue(np.array([0.01]), noise=False))
    print('cost 0.01:', campaign_f[0].cost(np.array([0.01]), noise=False))
    print('rev/cost 0.01:', campaign_f[0].revenue(np.array([0.01]), noise=False)/campaign_f[0].cost(np.array([0.01]), noise=False))

    print('rev 1.51:', campaign_f[0].revenue(np.array([1.51]), noise=False))
    print('cost 1.51:', campaign_f[0].cost(np.array([1.51]), noise=False))
    print('rev/cost 1.51:', campaign_f[0].revenue(np.array([1.51]), noise=False)/campaign_f[0].cost(np.array([1.51]), noise=False))
    import sys
    sys.exit()

    X = np.random.rand(1, 30)*MAX_BID
    #Y_cost = np.random.rand(10, 1)*MAX_BID
    #Y_rev = np.random.rand(1, 1)*MAX_BID
    
    c = sc.Subcampaign(59, 1.3, 63, 1.0)
    Y_cost = c.cost(X)
    Y_rev = c.revenue(X)
    X = X.reshape((-1, 1))
    Y_cost = Y_cost.reshape((-1, 1))
    Y_rev = Y_rev.reshape((-1, 1))
    #n._optimize()
    n.update(X, Y_cost, Y_rev)
    n.bid_choice()
    sys.exit()
    # c = agent.TS_TFP_Agent(len(campaign_a))
    # d = agent.TS_MCMC_Agent(len(campaign_a))
   # c = agent.Clairvoyant_Agent(campaign_a)
    agents = [b, e, f, g]  # , c, d]  # [a, b, c, d]

    X = np.random.rand(1, 30)*MAX_BID
    #Y_cost = np.random.rand(10, 1)*MAX_BID
    #Y_rev = np.random.rand(1, 1)*MAX_BID
    
    c = sc.Subcampaign(59, 1.3, 63, 1.0)
    Y_cost = c.cost(X)
    Y_rev = c.revenue(X)
    X = X.reshape((-1, 1))
    Y_cost = Y_cost.reshape((-1, 1))
    Y_rev = Y_rev.reshape((-1, 1))


    start = time.time()
    for i, a in enumerate(agents):
        a.update(X, Y_cost, Y_rev)
    end = time.time()
    print('elapsed time:', end - start)

    for i, a in enumerate(agents):
        r, exp_bid_mix, exp_rev_mix, exp_cost_mix = a.exp_revenue()

        bid_mix, _, roi, rev_mix, cost_mix, idx = (
                        a.bid_choice(ret_roi_revenue_cost=True)
                        )
    end = time.time()

    print('elapsed time:', end - start)
