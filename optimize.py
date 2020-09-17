import logging
import numpy as np
from config import *

logger = logging.getLogger(__name__)


def optimize(scl):  #, LOG=False):  # , min_bid=0.0):
    """
    Optimization on the revenue using dynamic programming

    Args:
        scl (List[Subcampaign]): subcampaigns' list,
                                a list of matrices with columns:
                                bid-revenue-cost
    Returns:
        r (list[][]): the revenue of the campaign in function of the cost
    """
    print('entering optimize')
    if logger.isEnabledFor(logging.INFO):
        logger.info('entering optimize')

    # the result matrix
    r = np.full((len(scl), N_COST), 0.0)  # np.nan)
    cost_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]
    bid_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]
    rev_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]

    if logger.isEnabledFor(logging.DEBUG): 
        logger.debug(f'scl shape: {scl.shape}')
        logger.info(f'the bids maximizing the revenues'
                    f' given the costs of the first subcampaign:\n{scl[0]}')
        logger.debug(f'scl[0] shape: {scl[0].shape}')

    r[0] = scl[0, :, 1]

    bid_mix[0] = [[x] for x in scl[0, :, 0]]
    cost_mix[0] = [[x] for x in scl[0, :, 2]]
    rev_mix[0] = [[x] for x in scl[0, :, 1]]

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'the first row of the optimization matrix:\n{r[0]}')
        logger.info(f'the first row of the bid_mix:\n{bid_mix[0]}')
        logger.info(f'the first row of the cost_mix:\n{cost_mix[0]}')

    for i in range(1, len(scl)):
        cbr = scl[i]
        cost_array = np.linspace(0., MAX_EXP, N_COST)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'the bids that maximize the revenues'
                        f'given the costs of the subcampaign {i+1}:\n{cbr}')
            logger.debug(f'the previous row of the optimization matrix:\n{r[i-1]}')

        for j, c in enumerate(cost_array):  # np.linspace(0., MAX_EXP, N_COST)):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'the current column/cost: {j}')
                logger.debug(f'current bids:\n{cbr[:j+1, 0]}')
                logger.debug(f'current revenues:\n{cbr[:j+1, 1]}')
                logger.debug(f'current costs:\n{cbr[:j+1, 2]}')
                logger.debug(f'the previous row of the resulting matrix'
                            f'- until the current cost/column {j} - reversed:\n'
                            f'{r[i-1, j::-1]}')

            mix = cbr[:j+1, 1] + r[i-1, j::-1]

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'the resulting possible mixes'
                            f' respecting the current cost:\n{mix}')

            try:
                opt_idx = np.nanargmax(mix)
            except ValueError:  # no feasible value
                r[i][j] = np.nan
                bid_mix[i][j] = [np.nan for x in range(i+1)]  # bid_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 0]]
                rev_mix[i][j] = [np.nan for x in range(i+1)]  # rev_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 1]]
                cost_mix[i][j] = [np.nan for x in range(i+1)]  #cost_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 2]]
# 
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f'no feasible value found, set r[{i}][{j}] to '
                                f'{r[i][j]})')
                    logger.warning(f'no feasible value found, set bid_mix[{i}][{j}] to '
                                f'{bid_mix[i][j]})')

                continue

            # mix = mix.reshape(-1, 1)
            if logger.isEnabledFor(logging.DEBUG):

                logger.debug(f'mix shape is: {mix.shape}')
                logger.debug(f'the mix index that maximize the revenue is:'
                            f'{opt_idx}\n'
                            f'the corrisponding revenue is:{mix[opt_idx]}')

            r[i][j] = mix[opt_idx] if mix[opt_idx] > 0. else 0.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'the revenue in {i}-{j} is: {r[i][j]}')

                logger.debug(f'the bid-mix in {i}-{j} is:\n {bid_mix[i][j]}')

            bid_mix[i][j] = bid_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 0]]
            rev_mix[i][j] = rev_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 1]]
            cost_mix[i][j] = cost_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 2]]

            if logger.isEnabledFor(logging.INFO):
                logger.info(f'the revenue in {i}-{j} is: {r[i][j]}')
                logger.info(f'the bid-mix in {i}-{j} is:\n {bid_mix[i][j]}')
                logger.info(f'the cost-mix in {i}-{j} is:\n {cost_mix[i][j]}')

    print('exit optimize')

    return r[-1], bid_mix[-1], rev_mix[-1], cost_mix[-1]


def bid_maxrevenue_given_cost(bid, revenue, cost,
                              max_cost=MAX_EXP, max_bid=MAX_BID,
                              min_cost=0., min_bid=0.0,
                              size=N_COST):
    if logger.isEnabledFor(logging.DEBUG):
        logger.info('Entering bid_maxrevenue_given_cost')
        logger.debug(f'bid shape is {bid.shape} and bid is: \n{bid}')
        logger.debug(f'cost shape is {cost.shape} and cost is: \n{cost}')
        logger.debug(f'revenue shape is {revenue.shape} and revenue is:'
                f'\n{revenue}')

    revenue = revenue[(bid >= min_bid) & (bid <= max_bid)]
    cost = cost[(bid >= min_bid) & (bid <= max_bid)]
    bid = bid[(bid >= min_bid) & (bid <= max_bid)]

    if logger.isEnabledFor(logging.DEBUG):
        logger.info(f'cost bid revenue\n{np.column_stack((cost, bid, revenue))}')

    b_r_c = np.full((size, 3), np.nan)
    b_r_c[:, 2] = np.linspace(0., max_cost, size)
    if min_bid == 0.:
        # revenue in zero is the least possible
        min_rev = np.nanmin(revenue)
        min_rev = min_rev - 1. if min_rev < 0. else 0.
        #b_r_c[0] = np.array([0., 0., 0.])
        b_r_c[0] = np.array([0., min_rev, 0.])

    for i, c in enumerate(np.linspace(min_cost, max_cost, size)):
        if c == 0.:
            continue
        mask = np.where(np.isnan(cost) | (cost > c), np.nan, cost)
        if logger.isEnabledFor(logging.DEBUG):
            logger.warning(f'iteration {i}, cost {c}')
            logger.warning(f'mask the values greater than {c} to nan\n {mask}')
        if np.all(np.isnan(mask)):
            b_r_c[i] = np.nan, np.nan, c
            if logger.isEnabledFor(logging.DEBUG):
                logger.warning(f"no value smaller than {c} found,"
                            f" set row {i} to the bid_revenue_cost:"
                            f" {b_r_c[i]}")
            continue
        idx = np.nanargmax(np.where(~np.isnan(mask), revenue, np.nan))
        b_r_c[i] = bid[idx], revenue[idx], c  # ost[idx]  # or only c??
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f"the value at index {idx} has the highest revenue"
                        f" given the cost {c}")
            logger.debug(f'set row {i} to the bid_revenue_cost {b_r_c[i]}')
    if logger.isEnabledFor(logging.DEBUG):
        logger.info(f'the resulting bids maximizing the revenue'
                    f' given a cost\n {b_r_c}')
        logger.debug(f'the result shape is: {b_r_c.shape}')
    return b_r_c

def bid_max_lb_revenue_given_cost_b(bid, lb_rev, lb_cost,
                                  ub_rev, ub_cost,
                                  max_cost=MAX_EXP, max_bid=MAX_BID,
                                  min_cost=0., min_bid=0.0,
                                  size=N_COST):
    b_r_c = bid_maxrevenue_given_cost(bid, lb_rev, ub_cost)
    # lb_cost = np.array([ lb_cost[bid==b][0] for b in b_r_c[:, 0]]).reshape(-1, 1)
    # ub_rev = np.array([ ub_rev[bid==b][0] for b in b_r_c[:, 0]]).reshape(-1, 1)
    print('ub_rev before', ub_rev)
    lb_cost = np.array([ lb_cost[bid==b] for b in b_r_c[:, 0]]).reshape(-1, 1)
    ub_rev = np.array([ ub_rev[bid==b] for b in b_r_c[:, 0]]).reshape(-1, 1)
    print('ub_rev after', ub_rev)
    return np.hstack((b_r_c, ub_rev, lb_cost)) # , axis=1)


def bid_max_lb_revenue_given_cost(bid, lb_rev, lb_cost,
                                  ub_rev, ub_cost,
                                  max_cost=MAX_EXP, max_bid=MAX_BID,
                                  min_cost=0., min_bid=0.0,
                                  size=N_COST):
    if logger.isEnabledFor(logging.DEBUG):
        logger.info('Entering bid_maxrevenue_given_cost')
        logger.debug(f'bid shape is {bid.shape} and bid is: \n{bid}')
        logger.debug(f'cost shape is {cost.shape} and cost is: \n{cost}')
        logger.debug(f'revenue shape is {revenue.shape} and revenue is:'
                f'\n{revenue}')

    lb_rev = lb_rev[(bid >= min_bid) & (bid <= max_bid)]
    ub_rev = ub_rev[(bid >= min_bid) & (bid <= max_bid)]
    lb_cost = lb_cost[(bid >= min_bid) & (bid <= max_bid)]
    ub_cost = ub_cost[(bid >= min_bid) & (bid <= max_bid)]
    bid = bid[(bid >= min_bid) & (bid <= max_bid)]

    if logger.isEnabledFor(logging.DEBUG):
        logger.info(f'cost bid revenue\n{np.column_stack((cost, bid, revenue))}')

    b_r_c = np.full((size, 5), np.nan)
    b_r_c[:, 2] = np.linspace(0, max_cost, size)
    if min_bid == 0.:
        min_lb_rev = np.nanmin(lb_rev)
        min_lb_rev = min_lb_rev - 1. if min_lb_rev < 0. else 0.
        min_ub_rev = np.nanmin(ub_rev)
        min_ub_rev = min_ub_rev - 1. if min_ub_rev < 0. else 0.
        b_r_c[0] = np.array([0., min_lb_rev, 0., min_ub_rev, 0.])
        #b_r_c[0] = np.array([0., 0., 0., 0., 0.])

    for i, c in enumerate(np.linspace(min_cost, max_cost, size)):
        if c == 0.:
            continue
        mask = np.where(np.isnan(ub_cost) | (ub_cost > c), np.nan, ub_cost)
        if logger.isEnabledFor(logging.DEBUG):
            logger.warning(f'iteration {i}, cost {c}')
            logger.warning(f'mask the values greater than {c} to nan\n {mask}')
        if np.all(np.isnan(mask)):
            b_r_c[i] = np.nan, np.nan, c, np.nan, np.nan
            if logger.isEnabledFor(logging.DEBUG):
                logger.warning(f"no value smaller than {c} found,"
                            f" set row {i} to the bid_revenue_cost:"
                            f" {b_r_c[i]}")
            continue
        idx = np.nanargmax(np.where(~np.isnan(mask), lb_rev, np.nan))
        b_r_c[i] = bid[idx], lb_rev[idx], c, ub_rev[idx], lb_cost[idx]  # ost[idx]  # or only c??
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f"the value at index {idx} has the highest revenue"
                        f" given the cost {c}")
            logger.debug(f'set row {i} to the bid_revenue_cost {b_r_c[i]}')
    if logger.isEnabledFor(logging.DEBUG):
        logger.info(f'the resulting bids maximizing the revenue'
                    f' given a cost\n {b_r_c}')
        logger.debug(f'the result shape is: {b_r_c.shape}')
    return b_r_c


def bid_max_ub_revenue_given_lb_rev_and_cost(bid, lb_rev,
                                             ub_rev, ub_cost,
                                             max_cost=MAX_EXP, max_bid=MAX_BID,
                                             min_cost=0., min_bid=0.0,
                                             # size=N_COST):
                                             ):
    if logger.isEnabledFor(logging.DEBUG):
        logger.info('Entering bid_max_ub_revenue_given_lb_rev_and_cost')
        logger.debug(f'bid shape is {bid.shape} and bid is: \n{bid}')
        logger.debug(f'cost shape is {ub_cost.shape} and cost is: \n{ub_cost}')
        logger.debug(f'revenue shape is {ub_rev.shape} and revenue is:'
                f'\n{ub_rev}')

    lb_rev = lb_rev[(bid >= min_bid) & (bid <= max_bid)]
    ub_rev = ub_rev[(bid >= min_bid) & (bid <= max_bid)]
    ub_cost = ub_cost[(bid >= min_bid) & (bid <= max_bid)]
    bid = bid[(bid >= min_bid) & (bid <= max_bid)]

    # print('ub_rev', ub_rev)
    # print('lb_rev', lb_rev)
    # print('ub_cost', ub_cost)

    if logger.isEnabledFor(logging.DEBUG):
        logger.info(f'cost bid revenue\n{np.column_stack((ub_cost, bid, lb_rev, ub_rev))}')

    # bid and revenue upper bound for each lower bound revenue and cost values
    b_ubr = np.full((N_REV, N_COST, 2), np.nan)

    if min_bid == 0.:
        # min_lb_rev = np.nanmin(lb_rev)
        # min_lb_rev = min_lb_rev - 1. if min_lb_rev < 0. else 0.
        min_ub_rev = np.nanmin(ub_rev)
        min_ub_rev = min_ub_rev - 1. if min_ub_rev < 0. else 0.
        # min_ub_rev = 0.
        
        b_ubr[0, 0] = np.array([0., min_ub_rev])
        # b_ubr[1:, 0] = np.array([0., np.nan])  # min_ub_rev])

    cost_array = np.linspace(0., MAX_EXP, N_COST)
    rev_array = np.linspace(0., MAX_REV, N_REV)

    max_lb_rev = np.nanmax(lb_rev)
    rev_thr = np.nanargmax(rev_array > max_lb_rev) if max_lb_rev < rev_array[-1] else N_REV
    # # max_ub_cost = np.nanmax(ub_cost)
    # # cost_thr = np.nanargmax(cost_array > max_ub_cost) if max_ub_cost < cost_array[-1] else N_COST

    for i, rev in enumerate(rev_array[:rev_thr]):
    # for i, rev in enumerate(rev_array):  #[:rev_thr]):
        for j, cost in enumerate(cost_array):  # [:cost_thr]):
            if cost == 0.:  #  and rev == 0.:
                continue
            #mask = np.where(np.isnan(ub_cost) | (ub_cost > cost) | (ub_cost < 0.0) | np.isnan(lb_rev) | (lb_rev < rev), False, True)
            mask = np.where(np.isnan(ub_cost) | (ub_cost > cost) | np.isnan(lb_rev) | (lb_rev < rev), False, True)
            # print('cost', cost)
            # print('rev', rev)
            # print('mask', mask)
            # print('maskshape', mask.shape)

            if logger.isEnabledFor(logging.DEBUG):
                logger.warning(f'iteration {i}, rev {rev}')
                logger.warning(f'iteration {j}, cost {cost}')
                logger.warning(f'mask the values greater than {cost} and lesser than {rev} to nan\n {mask}')
            #if np.all(np.isnan(mask)):
            if np.all(~(mask)):
                b_ubr[i,j] = np.array([np.nan, np.nan])
                if logger.isEnabledFor(logging.DEBUG):
                    logger.warning(f"no value smaller than {cost} found, or no value greater than {rev} found"
                                f" set {i}-{j} to the bid-upperbound revenue value:"
                                f" {b_ubr[i,j]}")
                continue

            #idx = np.nanargmax(np.where(~np.isnan(mask), ub_rev, np.nan))
            idx = np.nanargmax(np.where(mask, ub_rev, np.nan))
            b_ubr[i,j] = np.array([bid[idx], ub_rev[idx] if ub_rev[idx] > 0. else 0.])
            if logger.isEnabledFor(logging.DEBUG):
                logger.info(f"the value at index {idx} has the highest revenue"
                            f" given the cost {cost}"
                            f" and the revenue lower bound {rev}")
                logger.debug(f'set {i}-{j} to the bid_upperbound revenue {b_ubr[i,j]}')
        # if cost_thr < N_COST:
        #     for j, cost in enumerate(cost_array[cost_thr:]):
        #             b_ubr[i, j] = b_ubr[i, cost_thr -1]
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f'the resulting bids maximizing the revenue'
                        f' given a cost\n {b_ubr}')
            logger.debug(f'the result shape is: {b_ubr.shape}')
    return b_ubr


def optimize2(scl):  # , min_bid=0.0):
    """
    Optimization on the revenue using dynamic programming

    Args:
        scl (List[Subcampaign]): subcampaigns' list,
                                a list of matrices with columns:
                                bid-revenue-cost
    Returns:
        r (list[][]): the revenue of the campaign in function of the cost
    """
    print('entering optimize')
    #logger.info('entering optimize')

    # the result matrix
    r = np.full((2, N_COST), 0.0)  # np.nan)
    cost_mix = np.full((2, N_COST), 0.0) #[[[0.] for c in range(N_COST)] for i in range(2)]
    bid_mix =  np.full((2, N_COST), 0.0)#[[[0.] for c in range(N_COST)] for i in range(2)]
    rev_mix =  np.full((2, N_COST), 0.0)#[[[0.] for c in range(N_COST)] for i in range(2)]

    logger.debug(f'scl shape: {scl.shape}')

    logger.info(f'the bids maximizing the revenues'
                f' given the costs of the first subcampaign:\n{scl[0]}')
    logger.debug(f'scl[0] shape: {scl[0].shape}')

    r[0] = scl[0, :, 1]

    bid_mix[0] = scl[0, :, 0]  # [[x] for x in scl[0, :, 0]]
    cost_mix[0] = scl[0, :, 2]  # [[x] for x in scl[0, :, 2]]
    rev_mix[0] =  scl[0, :, 1] # [[x] for x in scl[0, :, 1]]

    logger.info(f'the first row of the optimization matrix:\n{r[0]}')
    logger.info(f'the first row of the bid_mix:\n{bid_mix[0]}')
    logger.info(f'the first row of the cost_mix:\n{cost_mix[0]}')

            # the best combination of cumulative cost for the previous subcompaigns, r[s-1], and the current considered, csr  
            #r[s][j] = max([r[s-1][x] + csr[j-x] for x in range(j+1)])


    for i in range(1, len(scl)):
        cbr = scl[i]
        cost_array = np.linspace(0., MAX_EXP, N_COST)

        logger.debug(f'the bids that maximize the revenues'
                    f'given the costs of the subcampaign {i+1}:\n{cbr}')
        logger.debug(f'the previous row of the optimization matrix:\n{r[0]}')

        for j, c in enumerate(cost_array):  # np.linspace(0., MAX_EXP, N_COST)):
            logger.debug(f'the current column/cost: {j}')
    #        logger.debug(f'current bids:\n{cbr[:j+1, 0]}')
    #        logger.debug(f'current revenues:\n{cbr[:j+1, 1]}')
    #        logger.debug(f'current costs:\n{cbr[:j+1, 2]}')
    #        logger.debug(f'the previous row of the resulting matrix'
    #                    f'- until the current cost/column {j} - reversed:\n'
    #                    f'{r[0, j::-1]}')

            mix = [r[0][x] + cbr[j-x, 1] for x in range(j+1)]  # cbr[:j+1, 1] + r[0, j::-1]

    #        logger.debug(f'the resulting possible mixes'
    #                    f' respecting the current cost:\n{mix}')

            try:
                opt_idx = np.nanargmax(mix)
            except ValueError:  # no feasible value
                r[1][j] = np.nan
                bid_mix[1][j] = [np.nan for x in range(i+1)]  # bid_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 0]]
                rev_mix[1][j] = [np.nan for x in range(i+1)]  # rev_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 1]]
                cost_mix[1][j] = [np.nan for x in range(i+1)]  #cost_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 2]]
# 
    #            logger.warning(f'no feasible value found, set r[{i}][{j}] to '
    #                        f'{r[i][j]})')
    #            logger.warning(f'no feasible value found, set bid_mix[{i}][{j}] to '
    #                        f'{bid_mix[i][j]})')

                continue

            # mix = mix.reshape(-1, 1)

            mix = np.array(mix)
    #        logger.debug(f'mix shape is: {mix.shape}')
    #        logger.debug(f'the mix index that maximize the revenue is:'
    #                    f'{opt_idx}\n'
    #                    f'the corrisponding revenue is:{mix[opt_idx]}')

            r[1][j] = mix[opt_idx] if mix[opt_idx] > 0. else 0.
    #        logger.debug(f'the revenue in {i}-{j} is: {r[1][j]}')

    #        logger.debug(f'the bid-mix in {i}-{j} is:\n {bid_mix[1][j]}')

            bid_mix[1][j] = bid_mix[0][j-opt_idx] + [cbr[opt_idx, 0]]
            rev_mix[1][j] = rev_mix[0][j-opt_idx] + [cbr[opt_idx, 1]]
            cost_mix[1][j] = cost_mix[0][j-opt_idx] + [cbr[opt_idx, 2]]

    #        logger.info(f'the revenue in {i}-{j} is: {r[1][j]}')
    #        logger.info(f'the bid-mix in {i}-{j} is:\n {bid_mix[1][j]}')
    #        logger.info(f'the cost-mix in {i}-{j} is:\n {cost_mix[1][j]}')
        r[0] = r[1]
        bid_mix[0] = bid_mix[1]
        cost_mix[0] = cost_mix[1]
        rev_mix[0] = rev_mix[1]

    print('exit optimize')

    return r[-1], bid_mix[-1], rev_mix[-1], cost_mix[-1]


# shuffle input
def optimize_3(scl):  #, LOG=False):  # , min_bid=0.0):
    """
    Optimization on the revenue using dynamic programming

    Args:
        scl (List[Subcampaign]): subcampaigns' list,
                                a list of matrices with columns:
                                bid-revenue-cost
    Returns:
        r (list[][]): the revenue of the campaign in function of the cost
    """
    print('entering optimize')
    if logger.isEnabledFor(logging.INFO):
        logger.info('entering optimize')

    # the result matrix
    r = np.full((len(scl), N_COST), 0.0)  # np.nan)
    cost_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]
    bid_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]
    rev_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]

    # shuffle input
    shuffle_idx = np.arange(len(scl))
    np.random.shuffle(shuffle_idx)


    if logger.isEnabledFor(logging.DEBUG): 
        logger.debug(f'scl shape: {scl.shape}')
        logger.info(f'the bids maximizing the revenues'
                    f' given the costs of the first subcampaign:\n{scl[0]}')
        logger.debug(f'scl[0] shape: {scl[0].shape}')

    r[0] = scl[shuffle_idx[0], :, 1]

    bid_mix[0] = [[x] for x in scl[shuffle_idx[0], :, 0]]
    cost_mix[0] = [[x] for x in scl[shuffle_idx[0], :, 2]]
    rev_mix[0] = [[x] for x in scl[shuffle_idx[0], :, 1]]

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'the first row of the optimization matrix:\n{r[0]}')
        logger.info(f'the first row of the bid_mix:\n{bid_mix[0]}')
        logger.info(f'the first row of the cost_mix:\n{cost_mix[0]}')

    for i in range(1, len(scl)):
        cbr = scl[shuffle_idx[i]]
        cost_array = np.linspace(0., MAX_EXP, N_COST)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'the bids that maximize the revenues'
                        f'given the costs of the subcampaign {i+1}:\n{cbr}')
            logger.debug(f'the previous row of the optimization matrix:\n{r[i-1]}')

        for j, c in enumerate(cost_array):  # np.linspace(0., MAX_EXP, N_COST)):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'the current column/cost: {j}')
                logger.debug(f'current bids:\n{cbr[:j+1, 0]}')
                logger.debug(f'current revenues:\n{cbr[:j+1, 1]}')
                logger.debug(f'current costs:\n{cbr[:j+1, 2]}')
                logger.debug(f'the previous row of the resulting matrix'
                            f'- until the current cost/column {j} - reversed:\n'
                            f'{r[i-1, j::-1]}')

            mix = cbr[:j+1, 1] + r[i-1, j::-1]

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'the resulting possible mixes'
                            f' respecting the current cost:\n{mix}')

            try:
                opt_idx = np.nanargmax(mix)
            except ValueError:  # no feasible value
                r[i][j] = np.nan
                bid_mix[i][j] = [np.nan for x in range(i+1)]  # bid_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 0]]
                rev_mix[i][j] = [np.nan for x in range(i+1)]  # rev_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 1]]
                cost_mix[i][j] = [np.nan for x in range(i+1)]  #cost_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 2]]
# 
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f'no feasible value found, set r[{i}][{j}] to '
                                f'{r[i][j]})')
                    logger.warning(f'no feasible value found, set bid_mix[{i}][{j}] to '
                                f'{bid_mix[i][j]})')

                continue

            # mix = mix.reshape(-1, 1)
            if logger.isEnabledFor(logging.DEBUG):

                logger.debug(f'mix shape is: {mix.shape}')
                logger.debug(f'the mix index that maximize the revenue is:'
                            f'{opt_idx}\n'
                            f'the corrisponding revenue is:{mix[opt_idx]}')

            r[i][j] = mix[opt_idx] if mix[opt_idx] > 0. else 0.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'the revenue in {i}-{j} is: {r[i][j]}')

                logger.debug(f'the bid-mix in {i}-{j} is:\n {bid_mix[i][j]}')

            bid_mix[i][j] = bid_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 0]]
            rev_mix[i][j] = rev_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 1]]
            cost_mix[i][j] = cost_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 2]]

            if logger.isEnabledFor(logging.INFO):
                logger.info(f'the revenue in {i}-{j} is: {r[i][j]}')
                logger.info(f'the bid-mix in {i}-{j} is:\n {bid_mix[i][j]}')
                logger.info(f'the cost-mix in {i}-{j} is:\n {cost_mix[i][j]}')

    print('exit optimize')
    bid_mix = bid_mix[-1]
    cost_mix = cost_mix[-1]
    rev_mix = rev_mix[-1]

    # unshuffle mixes
    ordered_bid_mix = np.empty((N_COST, len(scl)))
    ordered_cost_mix = np.empty((N_COST, len(scl)))
    ordered_rev_mix = np.empty((N_COST, len(scl)))


    # unshuffle
    for i, _ in enumerate(cost_array):
        for j, el in enumerate(shuffle_idx):
            ordered_bid_mix[i, el] = bid_mix[i][j]
            ordered_cost_mix[i, el] = cost_mix[i][j]
            ordered_rev_mix[i, el] = rev_mix[i][j]


    return r[-1], ordered_bid_mix, ordered_rev_mix, ordered_cost_mix


# sum low revenue
def safe_optimize_rev(scl):
    print('entering optimize')
    if logger.isEnabledFor(logging.INFO):
        logger.info('entering optimize')

    # the result matrix
    r = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    rev_mix = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    bid_mix = np.full((len(scl), N_COST, N_ROI, len(scl)), 0.0)  # np.nan)
    cost_mix = np.full((len(scl), N_COST, N_ROI, len(scl)), 0.0)  # np.nan)
    # cost_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]
    # bid_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]
    # rev_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]

    if logger.isEnabledFor(logging.DEBUG): 
        logger.debug(f'scl shape: {scl.shape}')
        logger.info(f'the bids maximizing the revenues'
                    f' given the costs of the first subcampaign:\n{scl[0]}')
        logger.debug(f'scl[0] shape: {scl[0].shape}')

    
    roi_array = np.linspace(LOW_ROI, MAX_ROI, N_ROI)
    cost_array = np.linspace(0., MAX_EXP, N_COST)

    # shuffle input
    shuffle_idx = np.arange(len(scl))
    np.random.shuffle(shuffle_idx)

    bid = scl[shuffle_idx[0], :, 0]
    lb_rev = scl[shuffle_idx[0], :, 1]
    ub_cost = scl[shuffle_idx[0], :, 2]
    ub_rev = scl[shuffle_idx[0], :, 3]
    lb_cost = scl[shuffle_idx[0], :, 4]
    #print('ub_cost', ub_cost)
    #print('lb_rev', lb_rev)
    #print('ub_rev', ub_rev)

    for k, roi in enumerate(roi_array):
        r[0, :, k] = ub_rev
        rev_mix[0, :, k] = lb_rev
        cost_mix[0, :, k, shuffle_idx[0]] = ub_cost
        bid_mix[0, :, k, shuffle_idx[0]] = bid
    # print('r', r[0])
    # print('r-reshape', r[0].reshape((3, -1)))
    # print('r-rereshape', r[0].T)
    # print('c_a*r_a', cost_array[j::-1]*roi_array.reshape((-1,1)))

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'the first row of the optimization matrix:\n{r[0]}')
        logger.info(f'the first row of the bid_mix:\n{bid_mix[0]}')
        logger.info(f'the first row of the cost_mix:\n{cost_mix[0]}')

    for i in range(1, len(scl)):
        bid = scl[shuffle_idx[i], :, 0]
        lb_rev = scl[shuffle_idx[i], :, 1]
        ub_cost = scl[shuffle_idx[i], :, 2]
        ub_rev = scl[shuffle_idx[i], :, 3]
        lb_cost = scl[shuffle_idx[i], :, 4]
        # cbr = scl[i]
        # cost_array = np.linspace(0., MAX_EXP, N_COST)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'the bids that maximize the revenues'
                        f'given the costs of the subcampaign {i+1}:\n{cbr}')
            logger.debug(f'the previous row of the optimization matrix:\n{r[i-1]}')

        for j, c in enumerate(cost_array):  # np.linspace(0., MAX_EXP, N_COST)):
            if c == 0. and (bid[0] == 0. or r[i-1][0][0] == 0.):
                continue
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'the current column/cost: {j}')

            for k, roi in enumerate(roi_array):
                #mask = (~np.isnan(lb_rev/ub_cost)) & (ub_cost <= c) & (>= roi)
                #print('mask', mask)
                #print('ub_rev', ub_rev)
                #s = np.where(mask, ub_rev, np.nan)  # lb_rev[s])
                #print('s', s)
                #mix = (lb_rev[:j+1] + cost_mix[i-1, j::-1]*roi_array.reshape((-1,1))) / c
                mix = (lb_rev[:j+1] + rev_mix[i-1, j::-1, :].T) / c
                #mix = (lb_rev[:j+1] + cost_array[j::-1]*roi_array.reshape((-1,1))) / c
                # print('mixshape', mix.shape)
                mask = (~np.isnan(mix)) & (~np.isnan(r[i-1,j::-1,:].T)) & (mix>=roi) # & (~np.isnan(r[i-1,j::-1,:].T)) 
                # print('maskshape', mask.shape)
                # print('rshape', r[i-1, j::-1, :].T.shape)
                # print('sumshape', np.array(ub_rev[:j+1] + r[i-1, j::-1, :].T).shape)
                # print('mix', mix)
                # print('mask', mask)
                # print('lbrev', lb_rev[:j+1])
                # print('lbrevplusc_a*r_a', lb_rev[:j+1] + cost_array[j::-1]*roi_array.reshape((-1,1)))
                # print('c_a*r_a', cost_array[j::-1]*roi_array.reshape((-1,1)))
                # print('rprevious', r[i-1, j::-1, :].T)
                # print('rpreviouszero', r[i-1, j::-1, 0])
                # print('rpreviouszeroreshape', r[i-1, j::-1, 0].reshape((1,-1)))

                #s = np.where(mask, ub_rev[:j+1] + r[i-1, j::-1, :].reshape((3,-1)), np.nan)  # lb_rev[s])
                s = np.where(mask, ub_rev[:j+1] + r[i-1, j::-1, :].T, np.nan)  # lb_rev[s])
                try:
                    a = np.unravel_index(np.nanargmax(s, axis=None), s.shape)
                except ValueError:  # no feasible value
                    # #print('thie')
                    # bid_mix[i,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                    # cost_mix[i,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                    # #bid_mix[i, j, k] = np.nan  # bid[a]
                    # rev_mix[i, j, k] = np.nan  # lb_rev[a]
                    # #cost_mix[i, j, k] = np.nan  # ub_cost[a]
                    # r[i, j, k] = np.nan  #  ub_rev[a]

                    # if logger.isEnabledFor(logging.WARNING):
                    #         logger.warning(f'no feasible value found, set r[0][{j}][{k}] to '
                    #                     f'{r[0][j][k]})')
                    #         logger.warning(f'no feasible value found, set bid_mix[0][{j}][{k}] to '
                    #                     f'{bid_mix[0][j][k]})')
                    # continue

                    for x, _ in enumerate(roi_array[k:]):
                        bid_mix[i,j,k+x] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                        cost_mix[i,j,k+x] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                        rev_mix[i, j, k+x] = np.nan  # lb_rev[a]
                        #cost_mix[i][j][k] = [np.nan for x in range(i+1)]  # np.nan  # ub_cost[a]
                        r[i, j, k+x] = np.nan  #  ub_rev[a]

                        if logger.isEnabledFor(logging.WARNING):
                                logger.warning(f'no feasible value found, set r[0][{j}][{k}] to '
                                            f'{r[0][j][k]})')
                                logger.warning(f'no feasible value found, set bid_mix[0][{j}][{k}] to '
                                            f'{bid_mix[0][j][k]})')
                    break

                #print('k', k)
                # print('j', j)
                # bid_mix[i][j][k] = bid[a[0]]
                # rev_mix[i][j][k] = lb_rev[a[0]]
                # cost_mix[i][j][k] = ub_cost[a[0]]
                # r[i][j][k] = ub_rev[a[0]]
                # print('s', s)
                # print('a', a)
                # print('a[0]', a[0])
                bid_mix[i,j,k,shuffle_idx[i]] = bid[a[1]]
                bid_mix[i,j,k] += bid_mix[i-1][j-a[1]][a[0]]
                cost_mix[i,j,k,shuffle_idx[i]] = ub_cost[a[1]]
                cost_mix[i,j,k] += cost_mix[i-1][j-a[1]][a[0]]
                

                #bid_mix[i, j, k] = bid_mix[i-1][j-a[1]][a[0]] + [bid[a[1]]]
                #rev_mix[i, j, k] = rev_mix[i-1][j-a[0]][l] + [lb_rev[a]] #lb_rev[a]
                rev_mix[i, j, k] = rev_mix[i-1][j-a[1]][a[0]] + lb_rev[a[1]] #lb_rev[a]
                #cost_mix[i, j, k] = cost_mix[i-1][j-a[1]][a[0]] + [ub_cost[a[1]]]
                #r[i, j, k] = r[i-1, j-a[1], a[0]] + ub_rev[a[1]]
                #r[i, j, k] = r[i-1, j::-1, :].T[a] + ub_rev[a[1]]
                r[i, j, k] = (ub_rev[:j+1]+r[i-1, j::-1, :].T)[a] #+ ub_rev[a[1]]

                # print('rijk', r[i, j, k])
                if r[i,j,k] / c < roi:
                    print('something unexpected happened', r[i,j,k] / c )
                    return


    return r[-1], bid_mix[-1], rev_mix[-1], cost_mix[-1]

# 3th dimension on ROI values
# rev as roi*cost 
def safe_optimize_roi_2(scl):
    print('entering optimize')
    if logger.isEnabledFor(logging.INFO):
        logger.info('entering optimize')

    # the result matrix
    r = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    #bid_mix = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    bid_mix = np.full((len(scl), N_COST, N_ROI, len(scl)), 0.0)  # np.nan)
    cost_mix = np.full((len(scl), N_COST, N_ROI, len(scl)), 0.0)  # np.nan)
    rev_mix = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    #cost_mix = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    #cost_mix = [[[0.] for i in range(len(scl)) for c in range(N_COST) for k in range(N_ROI)] ]
    #bid_mix = [[[0.] for i in range(len(scl)) for c in range(N_COST) for k in range(N_ROI)]
    # cost_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]
    #bid_mix = [[[[0.] for k in range(N_ROI)] for c in range(N_COST)] for i in range(len(scl))]
    #bid_mix = [[ [[0.] for i in range(N_ROI)] for c in range(N_COST)] for k in range(len(scl))]
    #cost_mix = [[ [[0.] for i in range(N_ROI)] for c in range(N_COST)] for k in range(len(scl))]
    # rev_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]
    # print(bid_mix[0][0][:])
    # print(bid_mix[-1])
    # return

    if logger.isEnabledFor(logging.DEBUG): 
        logger.debug(f'scl shape: {scl.shape}')
        logger.info(f'the bids maximizing the revenues'
                    f' given the costs of the first subcampaign:\n{scl[0]}')
        logger.debug(f'scl[0] shape: {scl[0].shape}')

    
    roi_array = np.linspace(LOW_ROI, MAX_ROI, N_ROI)
    cost_array = np.linspace(0., MAX_EXP, N_COST)

    bid = scl[0, :, 0]
    lb_rev = scl[0, :, 1]
    ub_cost = scl[0, :, 2]
    ub_rev = scl[0, :, 3]
    lb_cost = scl[0, :, 4]
    # print('ub_cost', ub_cost)
    # print('lb_rev', lb_rev)
    # print('ub_rev', ub_rev)
    # print('bid', bid)
    # return

    for j, c in enumerate(cost_array):
        if c == 0. and bid[0] == 0.:
            continue
        for k, roi in enumerate(roi_array):
            mask = (~np.isnan(lb_rev)) & (~np.isnan(lb_rev/ub_cost)) & (~np.isnan(lb_rev/c)) & (ub_cost <= c) & (lb_rev/c >= roi)  # (lb_rev/ub_cost >= roi)
            #print('mask', mask)
            #print('ub_rev', ub_rev)
            s = np.where(mask, ub_rev, np.nan)  # lb_rev[s])
            #print('s', s)
            a = 0
            try:
                a = np.nanargmax(s)
            except ValueError:  # no feasible value
                #print('thie')
                bid_mix[0,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                cost_mix[0,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                rev_mix[0, j, k] = np.nan  # lb_rev[a]
                #cost_mix[0][j][k] = [np.nan] # for x in range(i+1)]  # np.nan  # ub_cost[a]
                r[0, j, k] = np.nan  #  ub_rev[a]

                if logger.isEnabledFor(logging.WARNING):
                        logger.warning(f'no feasible value found, set r[0][{j}][{k}] to '
                                    f'{r[0][j][k]})')
                        logger.warning(f'no feasible value found, set bid_mix[0][{j}][{k}] to '
                                    f'{bid_mix[0][j][k]})')
                continue

            #print('k', k)
            # print('j', j)
    # bid_mix[0] = [[x] for x in scl[0, :, 0]]
    # cost_mix[0] = [[x] for x in scl[0, :, 2]]
            #bid_mix[0][j][k] = [bid[a]]
            bid_mix[0,j,k,0] = bid[a]
            cost_mix[0,j,k,0] = ub_cost[a]
            rev_mix[0][j][k] = lb_rev[a]
            #cost_mix[0][j][k] = [ub_cost[a]]
            r[0][j][k] = ub_rev[a]
            if r[0,j,k] / c < roi:
                print('something unexpected happened', r[i,j,k] / c )
                return

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'the first row of the optimization matrix:\n{r[0]}')
        logger.info(f'the first row of the bid_mix:\n{bid_mix[0]}')
        logger.info(f'the first row of the cost_mix:\n{cost_mix[0]}')

    for i in range(1, len(scl)):
        bid = scl[i, :, 0]
        lb_rev = scl[i, :, 1]
        ub_cost = scl[i, :, 2]
        ub_rev = scl[i, :, 3]
        lb_cost = scl[i, :, 4]
        # cbr = scl[i]
        # cost_array = np.linspace(0., MAX_EXP, N_COST)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'the bids that maximize the revenues'
                        f'given the costs of the subcampaign {i+1}:\n{cbr}')
            logger.debug(f'the previous row of the optimization matrix:\n{r[i-1]}')

        for j, c in enumerate(cost_array):  # np.linspace(0., MAX_EXP, N_COST)):
            if c == 0. and (bid[0] == 0. or r[i-1][0][0] == 0.):
                # for k in range(N_ROI):
                #     bid_mix[i][j][k] = [0.0 for x in range(i+1)]
                #print('bidmix', bid_mix[i][j][:])
                continue
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'the current column/cost: {j}')

            for k, roi in enumerate(roi_array):
                #mask = (~np.isnan(lb_rev/ub_cost)) & (ub_cost <= c) & (>= roi)
                #print('mask', mask)
                #print('ub_rev', ub_rev)
                #s = np.where(mask, ub_rev, np.nan)  # lb_rev[s])
                #print('s', s)
                #mix = (lb_rev[:j+1] + cost_mix[i-1, j::-1]*roi_array.reshape((-1,1))) / c
                mix = (lb_rev[:j+1] + cost_array[j::-1]*roi_array.reshape((-1,1))) / c
                # print('mixshape', mix.shape)
                mask = (~np.isnan(mix)) & (~np.isnan(r[i-1,j::-1,:].T)) & (mix>=roi) # & (~np.isnan(r[i-1,j::-1,:].T)) 
                # print('maskshape', mask.shape)
                # print('rshape', r[i-1, j::-1, :].T.shape)
                # print('sumshape', np.array(ub_rev[:j+1] + r[i-1, j::-1, :].T).shape)
                # print('mix', mix)
                # print('mask', mask)
                # print('lbrev', lb_rev[:j+1])
                # print('lbrevplusc_a*r_a', lb_rev[:j+1] + cost_array[j::-1]*roi_array.reshape((-1,1)))
                # print('c_a*r_a', cost_array[j::-1]*roi_array.reshape((-1,1)))
                # print('rprevious', r[i-1, j::-1, :].T)
                # print('rpreviouszero', r[i-1, j::-1, 0])
                # print('rpreviouszeroreshape', r[i-1, j::-1, 0].reshape((1,-1)))

                #s = np.where(mask, ub_rev[:j+1] + r[i-1, j::-1, :].reshape((3,-1)), np.nan)  # lb_rev[s])
                s = np.where(mask, ub_rev[:j+1] + r[i-1, j::-1, :].T, np.nan)  # lb_rev[s])
                try:
                    a = np.unravel_index(np.nanargmax(s, axis=None), s.shape)
                except ValueError:  # no feasible value
                    #print('thie')
                    # bid_mix[i][j][k] = [np.nan for x in range(i+1)]  # np.nan  # bid[a]

                    bid_mix[i,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                    cost_mix[i,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                    rev_mix[i, j, k] = np.nan  # lb_rev[a]
                    #cost_mix[i][j][k] = [np.nan for x in range(i+1)]  # np.nan  # ub_cost[a]
                    r[i, j, k] = np.nan  #  ub_rev[a]

                    if logger.isEnabledFor(logging.WARNING):
                            logger.warning(f'no feasible value found, set r[0][{j}][{k}] to '
                                        f'{r[0][j][k]})')
                            logger.warning(f'no feasible value found, set bid_mix[0][{j}][{k}] to '
                                        f'{bid_mix[0][j][k]})')
                    continue

                #print('k', k)
                # print('j', j)
                # bid_mix[i][j][k] = bid[a[0]]
                # rev_mix[i][j][k] = lb_rev[a[0]]
                # cost_mix[i][j][k] = ub_cost[a[0]]
                # r[i][j][k] = ub_rev[a[0]]
                # print('s', s)
                # print('a', a)
                # print('a[0]', a[0])
                
            #bid_mix[i][j] = bid_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 0]]
                # print('bida1', bid[a[1]])
                # print('bid_mix', bid_mix[i-1][j-a[1]][a[0]])
                # print('bidmixnew', bid_mix[i-1][j-a[1]][a[0]] + [bid[a[1]]])
                # return

                bid_mix[i,j,k,i] = bid[a[1]]
                bid_mix[i,j,k] += bid_mix[i-1][j-a[1]][a[0]]
                cost_mix[i,j,k,i] = ub_cost[a[1]]
                cost_mix[i,j,k] += cost_mix[i-1][j-a[1]][a[0]]
                #bid_mix[i][j][k] = (bid_mix[i-1][j::-1, :].T)[a] + [bid[a[1]]]
                #rev_mix[i, j, k] = rev_mix[i-1][j-a[0]][l] + [lb_rev[a]] #lb_rev[a]
                rev_mix[i, j, k] = rev_mix[i-1][j-a[1]][a[0]] + lb_rev[a[1]] #lb_rev[a]
                #cost_mix[i][j][k] = cost_mix[i-1][j-a[1]][a[0]] + [ub_cost[a[1]]]
                #r[i, j, k] = r[i-1, j-a[1], a[0]] + ub_rev[a[1]]
                #r[i, j, k] = r[i-1, j::-1, :].T[a] + ub_rev[a[1]]
                r[i, j, k] = (ub_rev[:j+1]+r[i-1, j::-1, :].T)[a] #+ ub_rev[a[1]]

                # print('rijk', r[i, j, k])
                if r[i,j,k] / c < roi:
                    print('something unexpected happened', r[i,j,k] / c )
                    return

    return r[-1], bid_mix[-1], rev_mix[-1], cost_mix[-1]


def safe_optimize_delta(scl):
    print('entering optimize')
    if logger.isEnabledFor(logging.INFO):
        logger.info('entering optimize')

    # the result matrix
    r = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    #bid_mix = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    bid_mix = np.full((len(scl), N_COST, N_ROI, len(scl)), 0.0)  # np.nan)
    cost_mix = np.full((len(scl), N_COST, N_ROI, len(scl)), 0.0)  # np.nan)
    rev_mix = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    delta_rev_mix = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    #cost_mix = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    #cost_mix = [[[0.] for i in range(len(scl)) for c in range(N_COST) for k in range(N_ROI)] ]
    #bid_mix = [[[0.] for i in range(len(scl)) for c in range(N_COST) for k in range(N_ROI)]
    # cost_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]
    #bid_mix = [[[[0.] for k in range(N_ROI)] for c in range(N_COST)] for i in range(len(scl))]
    #bid_mix = [[ [[0.] for i in range(N_ROI)] for c in range(N_COST)] for k in range(len(scl))]
    #cost_mix = [[ [[0.] for i in range(N_ROI)] for c in range(N_COST)] for k in range(len(scl))]
    # rev_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]
    # print(bid_mix[0][0][:])
    # print(bid_mix[-1])
    # return

    if logger.isEnabledFor(logging.DEBUG): 
        logger.debug(f'scl shape: {scl.shape}')
        logger.info(f'the bids maximizing the revenues'
                    f' given the costs of the first subcampaign:\n{scl[0]}')
        logger.debug(f'scl[0] shape: {scl[0].shape}')

    
    roi_array = np.linspace(LOW_ROI, MAX_ROI, N_ROI)
    cost_array = np.linspace(0., MAX_EXP, N_COST)

    # shuffle input
    shuffle_idx = np.arange(len(scl))
    np.random.shuffle(shuffle_idx)

    bid = scl[shuffle_idx[0], :, 0]
    lb_rev = scl[shuffle_idx[0], :, 1]
    ub_cost = scl[shuffle_idx[0], :, 2]
    ub_rev = scl[shuffle_idx[0], :, 3]
    lb_cost = scl[shuffle_idx[0], :, 4]
    # print('ub_cost', ub_cost)
    # print('lb_rev', lb_rev)
    # print('ub_rev', ub_rev)
    # print('bid', bid)
    # return

    for j, c in enumerate(cost_array):
        if c == 0. and bid[0] == 0.:
            continue
        for k, roi in enumerate(roi_array):
            mask = (~np.isnan(lb_rev)) & (~np.isnan(lb_rev/ub_cost)) & (~np.isnan(lb_rev/c)) & (ub_cost <= c) & (lb_rev/c >= roi)  # (lb_rev/ub_cost >= roi)
            #print('mask', mask)
            #print('ub_rev', ub_rev)
            s = np.where(mask, ub_rev - lb_rev, np.nan)  # lb_rev[s])
            #print('s', s)
            a = 0
            try:
                a = np.nanargmax(s)
            except ValueError:  # no feasible value
                #print('thie')
                bid_mix[0,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                cost_mix[0,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                rev_mix[0, j, k] = np.nan  # lb_rev[a]
                #cost_mix[0][j][k] = [np.nan] # for x in range(i+1)]  # np.nan  # ub_cost[a]
                r[0, j, k] = np.nan  #  ub_rev[a]
                delta_rev_mix[0,j,k] = np.nan

                if logger.isEnabledFor(logging.WARNING):
                        logger.warning(f'no feasible value found, set r[0][{j}][{k}] to '
                                    f'{r[0][j][k]})')
                        logger.warning(f'no feasible value found, set bid_mix[0][{j}][{k}] to '
                                    f'{bid_mix[0][j][k]})')
                continue

            #print('k', k)
            # print('j', j)
    # bid_mix[0] = [[x] for x in scl[0, :, 0]]
    # cost_mix[0] = [[x] for x in scl[0, :, 2]]
            #bid_mix[0][j][k] = [bid[a]]
            bid_mix[0,j,k,shuffle_idx[0]] = bid[a]
            cost_mix[0,j,k,shuffle_idx[0]] = ub_cost[a]
            rev_mix[0][j][k] = lb_rev[a]
            #cost_mix[0][j][k] = [ub_cost[a]]
            r[0][j][k] = ub_rev[a] #- lb_rev[a]
            delta_rev_mix[0,j,k] = ub_rev[a] - lb_rev[a]

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'the first row of the optimization matrix:\n{r[0]}')
        logger.info(f'the first row of the bid_mix:\n{bid_mix[0]}')
        logger.info(f'the first row of the cost_mix:\n{cost_mix[0]}')

    for i in range(1, len(scl)):
        bid = scl[shuffle_idx[i], :, 0]
        lb_rev = scl[shuffle_idx[i], :, 1]
        ub_cost = scl[shuffle_idx[i], :, 2]
        ub_rev = scl[shuffle_idx[i], :, 3]
        lb_cost = scl[shuffle_idx[i], :, 4]
        # cbr = scl[i]
        # cost_array = np.linspace(0., MAX_EXP, N_COST)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'the bids that maximize the revenues'
                        f'given the costs of the subcampaign {i+1}:\n{cbr}')
            logger.debug(f'the previous row of the optimization matrix:\n{r[i-1]}')

        for j, c in enumerate(cost_array):  # np.linspace(0., MAX_EXP, N_COST)):
            if c == 0. and (bid[0] == 0. or r[i-1][0][0] == 0.):
                # for k in range(N_ROI):
                #     bid_mix[i][j][k] = [0.0 for x in range(i+1)]
                #print('bidmix', bid_mix[i][j][:])
                continue
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'the current column/cost: {j}')

            for k, roi in enumerate(roi_array):
                #mask = (~np.isnan(lb_rev/ub_cost)) & (ub_cost <= c) & (>= roi)
                #print('mask', mask)
                #print('ub_rev', ub_rev)
                #s = np.where(mask, ub_rev, np.nan)  # lb_rev[s])
                #print('s', s)
                #mix = (lb_rev[:j+1] + cost_mix[i-1, j::-1]*roi_array.reshape((-1,1))) / c
                mix = (lb_rev[:j+1] + cost_array[j::-1]*roi_array.reshape((-1,1))) / c
                # print('mixshape', mix.shape)
                mask = (~np.isnan(mix)) & (~np.isnan(r[i-1,j::-1,:].T)) & (mix>=roi) # & (~np.isnan(r[i-1,j::-1,:].T)) 
                # print('maskshape', mask.shape)
                # print('rshape', r[i-1, j::-1, :].T.shape)
                # print('sumshape', np.array(ub_rev[:j+1] + r[i-1, j::-1, :].T).shape)
                # print('mix', mix)
                # print('mask', mask)
                # print('lbrev', lb_rev[:j+1])
                # print('lbrevplusc_a*r_a', lb_rev[:j+1] + cost_array[j::-1]*roi_array.reshape((-1,1)))
                # print('c_a*r_a', cost_array[j::-1]*roi_array.reshape((-1,1)))
                # print('rprevious', r[i-1, j::-1, :].T)
                # print('rpreviouszero', r[i-1, j::-1, 0])
                # print('rpreviouszeroreshape', r[i-1, j::-1, 0].reshape((1,-1)))

                #s = np.where(mask, ub_rev[:j+1] + r[i-1, j::-1, :].reshape((3,-1)), np.nan)  # lb_rev[s])
                s = np.where(mask, ub_rev[:j+1] + delta_rev_mix[i-1, j::-1, :].T - lb_rev[:j+1], np.nan)  # lb_rev[s])
                try:
                    a = np.unravel_index(np.nanargmax(s, axis=None), s.shape)
                except ValueError:  # no feasible value

                    # bid_mix[i,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                    # cost_mix[i,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                    # rev_mix[i, j, k] = np.nan  # lb_rev[a]
                    # #cost_mix[i][j][k] = [np.nan for x in range(i+1)]  # np.nan  # ub_cost[a]
                    # r[i, j, k] = np.nan  #  ub_rev[a]
                    # delta_rev_mix[i, j, k] = np.nan

                    # if logger.isEnabledFor(logging.WARNING):
                    #         logger.warning(f'no feasible value found, set r[0][{j}][{k}] to '
                    #                     f'{r[0][j][k]})')
                    #         logger.warning(f'no feasible value found, set bid_mix[0][{j}][{k}] to '
                    #                     f'{bid_mix[0][j][k]})')
                    # continue

                    for x, _ in enumerate(roi_array[k:]):
                        bid_mix[i,j,k+x] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                        cost_mix[i,j,k+x] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                        rev_mix[i, j, k+x] = np.nan  # lb_rev[a]
                        #cost_mix[i][j][k] = [np.nan for x in range(i+1)]  # np.nan  # ub_cost[a]
                        r[i, j, k+x] = np.nan  #  ub_rev[a]

                        if logger.isEnabledFor(logging.WARNING):
                                logger.warning(f'no feasible value found, set r[0][{j}][{k}] to '
                                            f'{r[0][j][k]})')
                                logger.warning(f'no feasible value found, set bid_mix[0][{j}][{k}] to '
                                            f'{bid_mix[0][j][k]})')
                    break

                #print('k', k)
                # print('j', j)
                # bid_mix[i][j][k] = bid[a[0]]
                # rev_mix[i][j][k] = lb_rev[a[0]]
                # cost_mix[i][j][k] = ub_cost[a[0]]
                # r[i][j][k] = ub_rev[a[0]]
                # print('s', s)
                # print('a', a)
                # print('a[0]', a[0])
                
            #bid_mix[i][j] = bid_mix[i - 1][j-opt_idx] + [cbr[opt_idx, 0]]
                # print('bida1', bid[a[1]])
                # print('bid_mix', bid_mix[i-1][j-a[1]][a[0]])
                # print('bidmixnew', bid_mix[i-1][j-a[1]][a[0]] + [bid[a[1]]])
                # return

                bid_mix[i,j,k,shuffle_idx[i]] = bid[a[1]]
                bid_mix[i,j,k] += bid_mix[i-1][j-a[1]][a[0]]
                cost_mix[i,j,k,shuffle_idx[i]] = ub_cost[a[1]]
                cost_mix[i,j,k] += cost_mix[i-1][j-a[1]][a[0]]
                #bid_mix[i][j][k] = (bid_mix[i-1][j::-1, :].T)[a] + [bid[a[1]]]
                #rev_mix[i, j, k] = rev_mix[i-1][j-a[0]][l] + [lb_rev[a]] #lb_rev[a]
                rev_mix[i, j, k] = rev_mix[i-1][j-a[1]][a[0]] + lb_rev[a[1]] #lb_rev[a]
                #cost_mix[i][j][k] = cost_mix[i-1][j-a[1]][a[0]] + [ub_cost[a[1]]]
                #r[i, j, k] = r[i-1, j-a[1], a[0]] + ub_rev[a[1]]
                #r[i, j, k] = r[i-1, j::-1, :].T[a] + ub_rev[a[1]]
                r[i, j, k] = (ub_rev[:j+1]+r[i-1, j::-1, :].T)[a] #+ ub_rev[a[1]]
                delta_rev_mix[i, j, k] = (ub_rev[:j+1]+delta_rev_mix[i-1, j::-1, :].T)[a] - lb_rev[a[1]]#+ ub_rev[a[1]]


    return r[-1], bid_mix[-1], rev_mix[-1], cost_mix[-1]


def safe_optimize_roi(scl):
    print('entering optimize')
    if logger.isEnabledFor(logging.INFO):
        logger.info('entering optimize')

    # the result matrix
    r = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    bid_mix = np.full((len(scl), N_COST, N_ROI, len(scl)), 0.0)  # np.nan)
    cost_mix = np.full((len(scl), N_COST, N_ROI, len(scl)), 0.0)  # np.nan)
    rev_mix = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)

    if logger.isEnabledFor(logging.DEBUG): 
        logger.debug(f'scl shape: {scl.shape}')
        logger.info(f'the bids maximizing the revenues'
                    f' given the costs of the first subcampaign:\n{scl[0]}')
        logger.debug(f'scl[0] shape: {scl[0].shape}')

    # shuffle input
    shuffle_idx = np.arange(len(scl))
    np.random.shuffle(shuffle_idx)
    
    roi_array = np.linspace(LOW_ROI, MAX_ROI, N_ROI)
    cost_array = np.linspace(0., MAX_EXP, N_COST)

    bid = scl[shuffle_idx[0], :, 0]
    lb_rev = scl[shuffle_idx[0], :, 1]
    ub_cost = scl[shuffle_idx[0], :, 2]
    ub_rev = scl[shuffle_idx[0], :, 3]
    lb_cost = scl[shuffle_idx[0], :, 4]

    for j, c in enumerate(cost_array):
        if c == 0. and bid[0] == 0.:
            continue
        for k, roi in enumerate(roi_array):
            mask = (~np.isnan(lb_rev/ub_cost)) & (~np.isnan(lb_rev/c)) & (ub_cost <= c) & (lb_rev/c >= roi)  # (lb_rev/ub_cost >= roi)

            s = np.where(mask, ub_rev, np.nan)  # lb_rev[s])

            a = 0
            try:
                a = np.nanargmax(s)
            except ValueError:  # no feasible value
                bid_mix[0,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                cost_mix[0,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                rev_mix[0, j, k] = np.nan  # lb_rev[a]
                r[0, j, k] = np.nan  #  ub_rev[a]

                if logger.isEnabledFor(logging.WARNING):
                        logger.warning(f'no feasible value found, set r[0][{j}][{k}] to '
                                    f'{r[0][j][k]})')
                        logger.warning(f'no feasible value found, set bid_mix[0][{j}][{k}] to '
                                    f'{bid_mix[0][j][k]})')
                continue

            bid_mix[0,j,k,shuffle_idx[0]] = bid[a]
            cost_mix[0,j,k,shuffle_idx[0]] = ub_cost[a]
            rev_mix[0][j][k] = lb_rev[a]
            r[0][j][k] = ub_rev[a]
            if r[0,j,k] / c < roi:
                print('something unexpected happened', r[i,j,k] / c )
                return

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'the first row of the optimization matrix:\n{r[0]}')
        logger.info(f'the first row of the bid_mix:\n{bid_mix[0]}')
        logger.info(f'the first row of the cost_mix:\n{cost_mix[0]}')

    for i in range(1, len(scl)):
        bid = scl[shuffle_idx[i], :, 0]
        lb_rev = scl[shuffle_idx[i], :, 1]
        ub_cost = scl[shuffle_idx[i], :, 2]
        ub_rev = scl[shuffle_idx[i], :, 3]
        lb_cost = scl[shuffle_idx[i], :, 4]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'the bids that maximize the revenues'
                        f'given the costs of the subcampaign {i+1}:\n{cbr}')
            logger.debug(f'the previous row of the optimization matrix:\n{r[i-1]}')

        for j, c in enumerate(cost_array):  # np.linspace(0., MAX_EXP, N_COST)):
            if c == 0. and (bid[0] == 0. or r[i-1][0][0] == 0.):
                continue

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'the current column/cost: {j}')

            for k, roi in enumerate(roi_array):
                mix = (lb_rev[:j+1] + cost_array[j::-1]*roi_array.reshape((-1,1))) / c

                mask = (~np.isnan(mix)) & (~np.isnan(r[i-1,j::-1,:].T)) & (mix>=roi) # & (~np.isnan(r[i-1,j::-1,:].T)) 

                s = np.where(mask, ub_rev[:j+1] + r[i-1, j::-1, :].T, np.nan)  # lb_rev[s])
                try:
                    a = np.unravel_index(np.nanargmax(s, axis=None), s.shape)
                except ValueError:  # no feasible value

                    for x, _ in enumerate(roi_array[k:]):
                        bid_mix[i,j,k+x] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                        cost_mix[i,j,k+x] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                        rev_mix[i, j, k+x] = np.nan  # lb_rev[a]
                        r[i, j, k+x] = np.nan  #  ub_rev[a]

                        if logger.isEnabledFor(logging.WARNING):
                                logger.warning(f'no feasible value found, set r[0][{j}][{k}] to '
                                            f'{r[0][j][k]})')
                                logger.warning(f'no feasible value found, set bid_mix[0][{j}][{k}] to '
                                            f'{bid_mix[0][j][k]})')
                    break

                bid_mix[i,j,k,shuffle_idx[i]] = bid[a[1]]
                bid_mix[i,j,k] += bid_mix[i-1][j-a[1]][a[0]]
                cost_mix[i,j,k,shuffle_idx[i]] = ub_cost[a[1]]
                cost_mix[i,j,k] += cost_mix[i-1][j-a[1]][a[0]]
                rev_mix[i, j, k] = rev_mix[i-1][j-a[1]][a[0]] + lb_rev[a[1]] #lb_rev[a]
                r[i, j, k] = (ub_rev[:j+1]+r[i-1, j::-1, :].T)[a] #+ ub_rev[a[1]]

                if r[i,j,k] / c < roi:
                    print('something unexpected happened', r[i,j,k] / c )
                    return

    return r[-1], bid_mix[-1], rev_mix[-1], cost_mix[-1]


def safe_optimize(scl):  #, LOG=False):  # , min_bid=0.0):
    """
    Optimization on the revenue using dynamic programming

    Args:
        scl (List[Subcampaign]): subcampaigns' list,
                                a list of matrices with columns:
                                bid-revenue-cost
    Returns:
        r (list[][]): the revenue of the campaign in function of the cost
    """
    print('entering optimize')
    if logger.isEnabledFor(logging.INFO):
        logger.info('entering optimize')

    # the result matrix
    r = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    bid_mix = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    rev_mix = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    cost_mix = np.full((len(scl), N_COST, N_ROI), 0.0)  # np.nan)
    # cost_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]
    # bid_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]
    # rev_mix = [[[0.] for c in range(N_COST)] for i in range(len(scl))]

    if logger.isEnabledFor(logging.DEBUG): 
        logger.debug(f'scl shape: {scl.shape}')
        logger.info(f'the bids maximizing the revenues'
                    f' given the costs of the first subcampaign:\n{scl[0]}')
        logger.debug(f'scl[0] shape: {scl[0].shape}')

    
    roi_array = np.linspace(0., MAX_ROI, N_ROI)
    cost_array = np.linspace(0., MAX_EXP, N_COST)

    ub_cost = scl[0, :, 2]
    ub_rev = scl[0, :, 3]
    bid = scl[0, :, 0]
    lb_cost = scl[0, :, 4]
    lb_rev = scl[0, :, 1]
    #print('ub_cost', ub_cost)
    #print('lb_rev', lb_rev)
    #print('ub_rev', ub_rev)

    for j, c in enumerate(cost_array):
        if c == 0. and bid[0] == 0.:
            continue
        for k, roi in enumerate(roi_array):
            mask = (~np.isnan(lb_rev/ub_cost)) & (ub_cost <= c) & (lb_rev/ub_cost >= roi)
            #print('mask', mask)
            #print('ub_rev', ub_rev)
            s = np.where(mask, ub_rev, np.nan)  # lb_rev[s])
            #print('s', s)
            a = 0
            try:
                a = np.nanargmax(s)
            except ValueError:  # no feasible value
                #print('thie')
                bid_mix[0, j, k] = np.nan  # bid[a]
                rev_mix[0, j, k] = np.nan  # lb_rev[a]
                cost_mix[0, j, k] = np.nan  # ub_cost[a]
                r[0, j, k] = np.nan  #  ub_rev[a]

                if logger.isEnabledFor(logging.WARNING):
                        logger.warning(f'no feasible value found, set r[0][{j}][{k}] to '
                                    f'{r[0][j][k]})')
                        logger.warning(f'no feasible value found, set bid_mix[0][{j}][{k}] to '
                                    f'{bid_mix[0][j][k]})')

            #print('k', k)
            # print('j', j)
            bid_mix[0][j][k] = bid[a]
            rev_mix[0][j][k] = lb_rev[a]
            cost_mix[0][j][k] = ub_cost[a]
            r[0][j][k] = ub_rev[a]

    # r[0] = scl[0, :, 1]

    # bid_mix[0] = [[x] for x in scl[0, :, 0]]
    # cost_mix[0] = [[x] for x in scl[0, :, 2]]
    # rev_mix[0] = [[x] for x in scl[0, :, 1]]
    #print('rev_mix_0', rev_mix[0])
    #print('r_0', r[0,:,0])

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'the first row of the optimization matrix:\n{r[0]}')
        logger.info(f'the first row of the bid_mix:\n{bid_mix[0]}')
        logger.info(f'the first row of the cost_mix:\n{cost_mix[0]}')

    start = time.time()
    for i in range(1, len(scl)):
        ub_cost = scl[i, :, 2]
        ub_rev = scl[i, :, 3]
        bid = scl[i, :, 0]
        lb_cost = scl[i, :, 4]
        lb_rev = scl[i, :, 1]
        # cbr = scl[i]
        # cost_array = np.linspace(0., MAX_EXP, N_COST)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'the bids that maximize the revenues'
                        f'given the costs of the subcampaign {i+1}:\n{cbr}')
            logger.debug(f'the previous row of the optimization matrix:\n{r[i-1]}')

        for j, c in enumerate(cost_array):  # np.linspace(0., MAX_EXP, N_COST)):
            if c == 0. and (bid[0] == 0. or r[i-1][0][0] == 0.):
                continue
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'the current column/cost: {j}')
                # logger.debug(f'current bids:\n{cbr[:j+1, 0]}')
                # logger.debug(f'current revenues:\n{cbr[:j+1, 1]}')
                # logger.debug(f'current costs:\n{cbr[:j+1, 2]}')
                # logger.debug(f'the previous row of the resulting matrix'
                #             f'- until the current cost/column {j} - reversed:\n'
                #             f'{r[i-1, j::-1]}')
            a = -1 
            # for k, roi in enumerate(roi_array):
            k = 0
            roi = 0.
            while k < N_ROI:
                a = -1 
                for l, _ in enumerate(roi_array[:k+1]):
                    mix = (lb_rev[:j+1] + rev_mix[i-1, j::-1, l]) / c
                    mask = (~np.isnan(mix)) & (mix>=roi)

                    s = np.where(mask, ub_rev[:j+1] + r[i-1, j::-1, l], np.nan)  # lb_rev[s])
                    # print('a before is', a)
                    try:
                        aa = np.nanargmax(s)
                    except ValueError:  # no feasible value
                     #    print('l is', l)
                     #    print('no max found')
                     #    print('a is', a)
                        continue
                    a = aa
                    break
                if a<0:
                    # print('no solution found')
                    # no feasible solution
                    # print('s is:', s)
                    # print('mix is:', mix)
                    # print('roi is:', roi)
                    for x, _ in enumerate(roi_array[k:]):
                        bid_mix[i, j, x+k] = np.nan  # bid[a]
                        rev_mix[i, j, x+k] = np.nan  # lb_rev[a]
                        cost_mix[i, j, x+k] = np.nan  # ub_cost[a]
                        r[i, j, x+k] = np.nan  #  ub_rev[a]
                        if logger.isEnabledFor(logging.WARNING):
                            logger.warning(f'no feasible value found, set r[0][{j}][{k}] to '
                                           f'{r[0][j][k]})')
                            logger.warning(f'no feasible value found, set bid_mix[0][{j}][{k}] to '
                                           f'{bid_mix[0][j][k]})')
                    k = N_ROI
                    break
                
                bid_mix[i, j, k] = bid_mix[i-1][j-a][l] + [bid[a]]
                #rev_mix[i, j, k] = rev_mix[i-1][j-a][l] + [lb_rev[a]] #lb_rev[a]
                rev_mix[i, j, k] = rev_mix[i-1][j-a][l] + lb_rev[a] #lb_rev[a]
                cost_mix[i, j, k] = cost_mix[i-1][j-a][l] + [ub_cost[a]]
                r[i, j, k] = r[i-1, j-a, l] + ub_rev[a]
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f'the revenue in {i}-{j}-{k} is: {r[i][j][k]}')
                    logger.info(f'the bid-mix in {i}-{j}-{k} is:\n {bid_mix[i][j][k]}')
                    logger.info(f'the cost-mix in {i}-{j}-{k} is:\n {cost_mix[i][j][k]}')
                k+=1
                for z, roi in enumerate(roi_array[k:]):
                    if mix[a] >= roi:
                        # print('z+k', z+k)
                        # print('z', z)
                        # print('k', k)
                        # print('mix[a]:', mix[a])
                        # print('roi', roi)
                        bid_mix[i, j, k+z] = bid_mix[i-1][j-a][l] + [bid[a]]
                        #rev_mix[i, j, k] = rev_mix[i-1][j-a][l] + [lb_rev[a]] #lb_rev[a]
                        rev_mix[i, j, k+z] = rev_mix[i-1][j-a][l] + lb_rev[a] #lb_rev[a]
                        cost_mix[i, j,k+z] = cost_mix[i-1][j-a][l] + [ub_cost[a]]
                        r[i, j, k+z] = r[i-1, j-a, l] + ub_rev[a]
                        # print('r/c', r[i,j,z]/c)
                        #k = z+1
                        # k+=1
                        # print('k now is', k)
                    else:
                        # print('else z is', z)
                        # print('roi is', roi)
                        break
                k += z

    print('exit optimize')
    end = time.time()
    print('elapsed time:', end - start)

    return r[-1], bid_mix[-1], rev_mix[-1], cost_mix[-1]


def safe_opt(scl):
    print('entering optimize')
    if logger.isEnabledFor(logging.INFO):
        logger.info('entering optimize')

    # the result matrix
    r = np.full((len(scl), N_REV, N_COST), np.nan)
    bid_mix = np.full((len(scl), N_REV, N_COST, len(scl)), 0.0)  #np.nan)
    cost_mix = np.full((len(scl), N_REV, N_COST, len(scl)),0.0)  #np.nan)
    rev_mix = np.full((len(scl), N_REV, N_COST), np.nan)

    if logger.isEnabledFor(logging.DEBUG): 
        logger.debug(f'scl shape: {scl.shape}')
        logger.info(f'the bids maximizing the revenues'
                    f' given the costs of the first subcampaign:\n{scl[0]}')
        logger.debug(f'scl[0] shape: {scl[0].shape}')

    # shuffle input
    shuffle_idx = np.arange(len(scl))
    np.random.shuffle(shuffle_idx)
    
    rev_array = np.linspace(0., MAX_REV, N_REV)
    cost_array = np.linspace(0., MAX_EXP, N_COST)

    bid = scl[shuffle_idx[0], :, :, 0]
    #lb_rev = scl[shuffle_idx[0], :, 1]
    #ub_cost = scl[shuffle_idx[0], :, 2]
    ub_rev = scl[shuffle_idx[0], :, :, 1]
    #lb_cost = scl[shuffle_idx[0], :, 4]

    bid_mix[0,:,:,shuffle_idx[0]] = bid
    r[0,:,:] = ub_rev
    cost_mix[0,:,:, shuffle_idx[0]] = np.where(~np.isnan(bid), cost_array, np.nan)
    #print('cost_mix')
    #return

    if logger.isEnabledFor(logging.INFO):
        logger.info(f'the first row of the optimization matrix:\n{r[0]}')
        logger.info(f'the first row of the bid_mix:\n{bid_mix[0]}')
        logger.info(f'the first row of the cost_mix:\n{cost_mix[0]}')

    for i in range(1, len(scl)):
        bid = scl[shuffle_idx[i], :, :, 0]
        #lb_rev = scl[shuffle_idx[i], :, 1]
        #ub_cost = scl[shuffle_idx[i], :, 2]
        ub_rev = scl[shuffle_idx[i], :, :, 1]
        #lb_cost = scl[shuffle_idx[i], :, 4]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'the bids that maximize the revenues'
                        f'given the costs of the subcampaign {i+1}:\n{cbr}')
            logger.debug(f'the previous row of the optimization matrix:\n{r[i-1]}')

        for j, rev in enumerate(rev_array):  # np.linspace(0., MAX_EXP, N_COST)):

            for k, c in enumerate(cost_array):
                # if (c == 0.) and (bid[j, k] == 0.) and (r[i-1][j][k] <= 0.):
                #     continue

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'the current column/cost: {j}')

                mix = ub_rev[:j+1, :k+1] + r[i-1, j::-1, k::-1]

                try:
                    a = np.unravel_index(np.nanargmax(mix, axis=None), mix.shape)
                except ValueError:  # no feasible value

                    mix = ub_rev[j, :k+1] + r[i-1, j, k::-1]

                    if np.all(np.isnan(mix)):
                        bid_mix[i,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                        cost_mix[i,j,k] = np.full((len(scl)),np.nan)  # for x in range(0+1)]  # np.nan  # bid[a]
                        rev_mix[i, j, k] = np.nan  # lb_rev[a]
                        r[i, j, k] = np.nan  #  ub_rev[a]

                        if logger.isEnabledFor(logging.WARNING):
                                logger.warning(f'no feasible value found, set r[0][{j}][{k}] to '
                                            f'{r[0][j][k]})')
                                logger.warning(f'no feasible value found, set bid_mix[0][{j}][{k}] to '
                                            f'{bid_mix[0][j][k]})')
                        continue
                    a = np.nanargmax(mix)
                    bid_mix[i,j,k,shuffle_idx[i]] = bid[j, a]
                    bid_mix[i,j,k] += bid_mix[i-1, j, k-a]
                    cost_mix[i,j,k,shuffle_idx[i]] = cost_array[a]  # ub_cost[a[1]]
                    cost_mix[i,j,k] += cost_mix[i-1, j, k-a]
                    r[i, j, k] = mix[a]  # (ub_rev[a] + r[i-1, j-a[0], k-a[1]]  # (ub_rev[:j+1]+r[i-1, j::-1, :].T)[a] #+ ub_rev[a[1]]
                    continue

                mix_j = ub_rev[j, :k+1] + r[i-1, j, k::-1]
                
                if np.all(np.isnan(mix_j)):
                    bid_mix[i,j,k,shuffle_idx[i]] = bid[a]
                    bid_mix[i,j,k] += bid_mix[i-1, j-a[0], k-a[1]]
                    cost_mix[i,j,k,shuffle_idx[i]] = cost_array[a[1]]  # ub_cost[a[1]]
                    cost_mix[i,j,k] += cost_mix[i-1, j-a[0], k - a[1]]
                    # rev_mix[i, j, k] = rev_mix[i-1][j-a[1]][a[0]] + lb_rev[a[1]] #lb_rev[a]
                    #r[i, j, k] = ub_rev[a] + r[i-1, j-a[0], k-a[1]]  # (ub_rev[:j+1]+r[i-1, j::-1, :].T)[a] #+ ub_rev[a[1]]
                    r[i, j, k] = mix[a]  # (ub_rev[a] + r[i-1, j-a[0], k-a[1]]  # (ub_rev[:j+1]+r[i-1, j::-1, :].T)[a] #+ ub_rev[a[1]]
                    continue

                max_j = np.nanmax(mix_j)

                if max_j > mix[a]:
                    a = j, np.nanargmax(mix_j)

                bid_mix[i,j,k,shuffle_idx[i]] = bid[a]
                bid_mix[i,j,k] += bid_mix[i-1, j-a[0], k-a[1]]
                cost_mix[i,j,k,shuffle_idx[i]] = cost_array[a[1]]  # ub_cost[a[1]]
                cost_mix[i,j,k] += cost_mix[i-1, j-a[0], k - a[1]]
                # rev_mix[i, j, k] = rev_mix[i-1][j-a[1]][a[0]] + lb_rev[a[1]] #lb_rev[a]
                #r[i, j, k] = ub_rev[a] + r[i-1, j-a[0], k-a[1]]  # (ub_rev[:j+1]+r[i-1, j::-1, :].T)[a] #+ ub_rev[a[1]]
                r[i, j, k] = mix[a]  # (ub_rev[a] + r[i-1, j-a[0], k-a[1]]  # (ub_rev[:j+1]+r[i-1, j::-1, :].T)[a] #+ ub_rev[a[1]]


    return r[-1], bid_mix[-1], rev_mix[-1], cost_mix[-1]


if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG,
    logging.basicConfig(level=logging.ERROR,
                        # format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        # datefmt='%m-%d %H:%M',
                       filename='/tmp/myothergpapp.log',
                       filemode='w')
    #logging.disable(logging.CRITICAL)
    logging.logThreads = 0
    logging.logProcesses = 0

    logging.getLogger().propagate = False
    logging.getLogger().disabled = True

    import sys
    np.set_printoptions(threshold=sys.maxsize)
    # bid = np.linspace(0, 2.0, 30)
    # revenue = np.linspace(1, 3.0, 30)
    # cost = np.linspace(1, 2.3, 30)
    # # revenue[10] = 5.
    # sc = bid_maxrevenue_given_cost(bid, revenue, cost, min_bid=0.0, max_bid = 1.3, min_cost=2., max_cost=30)
    # print('bid', bid)
    # print('revenue', revenue)
    # print('cost', cost)
    # print('sc', sc)
    # sys.exit()

    # scl = [sc for x in range(3)]
    # scl = np.array(scl)

    # r, bid_mix, rev_mix, cost_mix = optimize(scl)
    import subcampaign as sc
    import time

    # campaign_a = [sc.Subcampaign(64, 0.3, 69, 0.2),
    #               sc.Subcampaign(70, 0.4, 78, 0.2),
    #               sc.Subcampaign(70, 0.3, 73, 0.1)]
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

    bid = np.linspace(0, MAX_BID, SAMPLES_INVERSE).reshape(-1, 1)
    #scl = np.array([bid_max_lb_revenue_given_cost(bid, s.revenue(bid, noise=False), s.cost(bid, noise=False), s.revenue(bid, noise=False), s.cost(bid, noise=False)) for s in campaign_a])
    start = time.time()
    scl = np.array([bid_max_ub_revenue_given_lb_rev_and_cost(bid, s.revenue(bid, noise=False), s.revenue(bid, noise=False), s.cost(bid, noise=False)) for s in campaign_f])
    #scl = np.array(scl)
    end = time.time()
    print('elapsed time build scl', end - start)

    print('scl 5th rev, first sc', scl[0, 13, :])
    sys.exit()

    scl2 = scl
    print(scl.shape)
    scl = np.array(scl)
    print(scl.shape)
    #print(scl)
    #print(scl)
    #exit()

    start = time.time()
    r, bid_mix_3, rev_mix, cost_mix = safe_opt(scl)
    end = time.time()
    print('elapsed time optimizing', end - start)
    
    # roi mask and roi lb_rev mask 
    # find revenue given roi target

    start = time.time()
    rev_array = np.linspace(0., MAX_REV, N_REV)
    cost_array = np.linspace(0., MAX_EXP, N_COST)
    roi_mask = np.empty((N_REV,N_COST))

    for i, rev in enumerate(rev_array):
        for j, cost in enumerate(cost_array):
            if ~np.isnan(rev/cost) & (rev/cost >= MIN_ROI):
                roi_mask[i,j] = 1.
            else:
                roi_mask[i,j] = 0.
    end = time.time()

    print('elapsed time building roi mask', end - start)

    start = time.time()
    idx = np.unravel_index(np.nanargmax(r*roi_mask, axis=None), shape=r.shape)
    end = time.time()
    print('elapsed time building roi mask', end - start)
    print('idx, ', idx)
    print('r target', r[idx])
    print('rev_array[idx]', rev_array[idx[0]])
    print('roi rev_a/c_a[idx]', rev_array[idx[0]]/cost_array[idx[1]])
    print('roi in r target', r[idx]/cost_array[idx[1]]) 
    start = time.time()
    rev_array = np.linspace(0., MAX_REV, N_REV)
    cost_array = np.linspace(0., MAX_EXP, N_COST)
    roi_mask = np.empty((N_REV,N_COST))

    for i, rev in enumerate(rev_array):
        for j, cost in enumerate(cost_array):
            if ~np.isnan(rev/cost) & (rev/cost >= MIN_ROI):
                roi_mask[i,j] = rev
            else:
                roi_mask[i,j] = np.nan 
    end = time.time()
    print('elapsed time building roi mask', end - start)
    start = time.time()
    idx = np.unravel_index(np.nanargmax(r-roi_mask, axis=None), shape=r.shape)
    end = time.time()
    print('idx, ', idx)
    print('elapsed time building roi mask', end - start)
    print('r target', r[idx])
    print('rev_array[idx]', rev_array[idx[0]])
    print('roi rev_a/c_a[idx]', rev_array[idx[0]]/cost_array[idx[1]])
    print('r target - roi_mask idx', (r-roi_mask)[idx])
    #print('r target - roi_mask', (r-roi_mask)[218,:])
    #print('r target - roi_mask', (r-roi_mask)[385,:])
    print('roi in r target', r[idx]/cost_array[idx[1]]) 
    #sys.exit()
    


    #sys.exit()
    start = time.time()
    scl = np.array([bid_max_lb_revenue_given_cost(bid, s.revenue(bid, noise=False), s.cost(bid, noise=False), s.revenue(bid, noise=False), s.cost(bid, noise=False)) for s in campaign_f])
    end = time.time()
    print('elapsed time build scl-classic', end - start)
    print('is close scl and scl2?', np.allclose(scl2[0,0,:, 1], scl[0,:, 1]))
    #print('scl', scl)
    #sys.exit()
    #print('r', r[0, :])
    # end = time.time()
    # print('elapsed time safe_opt_roi:', end - start)
    # # print('r', r[:, 1])
    # roi_array = np.linspace(0., MAX_ROI, N_ROI)
    # cost_array = np.linspace(0., MAX_EXP, N_COST)
    # 
    # for j, c in enumerate(cost_array):
    # #    for i, v in enumerate(r[:, 1]):
    #     if r[j,2]/c < roi_array[2]:
    #             print('oh shit')
    #             print('roi', roi_array[2])
    #             print('v/c', r[j,2]/c)

#   #   print('r0', r[:, 0])
#   #   print('r1', r[:, 1])
#   #   print('r2', r[:, 2])
#   #   print('r/c0', r[:, 0]/cost_array)
#   #   print('r/c1', r[:, 1]/cost_array)
#   #   print('r/c2', r[:, 2]/cost_array)
#   #   print('rev_mix', rev_mix)
    # roi_array = np.linspace(0., MAX_ROI, N_ROI)
#    print('cost*roi', cost_array*roi_array.reshape((-1,1)))

    start = time.time()
    r2, bid_mix_2, rev_mix, cost_mix_2 = optimize(scl)
    end = time.time()
    print('elapsed time optimize:', end - start)
    # print('r2', r2)
    #print('is equal r_safe_opt_roi and r_optimize?', np.array_equal(r[:,0], r2))
    #print('is equal r and r_2?', np.array_equal(r, r_given_revsum))

    start = time.time()
    r3, bid_mix, rev_mix, cost_mix = safe_optimize_roi(scl)
    end = time.time()
    print('elapsed time safe_optimize_roi2:', end - start)

    start = time.time()
    r4, bid_mix_4, rev_mix, cost_mix_3 = optimize_3(scl)
    end = time.time()
    #print('r4', r4)
    print('elapsed time optimize:', end - start)
    print('is equal r_optimize_3 and r_optimize?', np.array_equal(r4, r2))
    print('is close r2 and r4?', np.allclose(r4, r2))
    print('is close r and r_safe?', np.allclose(r[0, :], r2))
    print('is close r and r_safe?', np.allclose(r[0, :], r4))
    print('r', r[0,:])
    print('r2', r2)
    print('is equal bidmixr and bidmixr_2?', np.allclose(bid_mix_4, bid_mix_3[0,:,:], rtol=0, atol=0, equal_nan=True))  #np.isclose(r3[:,1], r[:,1]))
    print('is equal costmixr and costmixr_2?', np.allclose(cost_mix_3, cost_mix[0,:,:], rtol=0, atol=0, equal_nan=True))  #np.isclose(r3[:,1], r[:,1]))
    #sys.exit()
    print('rev_array[2]', np.linspace(0., MAX_REV, N_REV)[2])
    print('r\n', r[2, :])
    sys.exit()
    #print('cost_mix', cost_mix[0,:,:])
    #print('cost_mix3', cost_mix_2)
    # print('scl', scl)
    # print('scl2', scl2)
    print('scl shape', scl.shape)
    print('is close scl and scl2?', np.allclose(scl2[0,0,:, 1], scl[0,:, 1]))
    #print('is close scl and scl2?', np.allclose(scl2[:,0, 1], scl[:, 1]))
    #print('r', r2)
    #print('r3', r4)
    print('is equal bid_mix?', np.array_equal(bid_mix_2, bid_mix_4))
    print('is equal bid_mix_rev?', np.array_equal(bid_mix_2, bid_mix_3[0, :, :]))
    print(bid_mix_2[-1])
    print(bid_mix[-1, 0])
    sys.exit()
    # print('r2', r2)
#     print('r/c0', r3[:, 0]/cost_array)
#     print('r/c1', r3[:, 1]/cost_array)
#     print('r/c2', r3[:, 2]/cost_array)
    #print('is equal r and r_2?', np.array_equal(r3[:,0], r2))
    #print('is equal r and r_2?', np.array_equal(r3[:,0], r[:,0]))
    #print('is equal r and r_2?', np.array_equal(r3[:,1], r[:,1]))
    print('is equal r_safe_opt_roi and r_safe_opt_roi2?', np.allclose(r, r3, rtol=0, atol=0, equal_nan=True))  #np.isclose(r3[:,1], r[:,1]))
#     print('r[1:', r[1:])
#     print('r:,1', r[:, 1:])
#     print('r', r)
#     print('r/c_a', (r.T/cost_array).T)
#     print('bid_mix3', bid_mix_3)
#     print('bid_mix2', bid_mix_2)
    #print('bid_mix', bid_mix)
#    print('bid_mix30', bid_mix_3[:, 0])
    print('is equal r and r_2?', np.allclose(r[:,0], r2, rtol=0, atol=0, equal_nan=True))  #np.isclose(r3[:,1], r[:,1]))
    print('is equal bidmix safe_opt_roi and bidmix safe_opt_roi2?', np.allclose(bid_mix, bid_mix_3, rtol=0, atol=0, equal_nan=True))  #np.isclose(r3[:,1], r[:,1]))
    print('is equal bidmixr and bidmixr_2?', np.allclose(bid_mix_2, bid_mix_3[:,0], rtol=0, atol=0, equal_nan=True))  #np.isclose(r3[:,1], r[:,1]))
    #print('is equal r and r_2?', np.isclose(r3[:,2], r[:,2]))

    sys.exit()
