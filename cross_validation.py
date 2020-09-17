import gp_model as gp
import logging
import numpy as np
import plot
from config import *
from optimize import bid_maxrevenue_given_cost, optimize

logger = logging.getLogger(__name__)

def init_crossval_gp(data):
    logger.info('enter init_crossval_gp')
    rng = np.random.default_rng()
    # assert(data.shape[1] == 3)

    # divide data in 2 subsets
    indeces = rng.permutation(data.shape[0])

    logger.debug(f'data length: {len(data)}')
    logger.debug(f'data shape: {data.shape}')
    logger.debug(f'indeces:\n{(indeces)}')
    logger.debug(f'input data:\n{data}')

    a = data[indeces[:len(indeces)//2]]
    b = data[indeces[len(indeces)//2:]]

    logger.debug(f'indeces dataset a:\n{a}')
    logger.debug(f'indeces dataset b:\n{b}')

    gp_a = gp.GPModel()
    gp_b = gp.GPModel()

    gp_a.update(a[:, 0], a[:, 1], a[:, 2])
    gp_b.update(b[:, 0], b[:, 1], b[:, 2])

    return gp_a, gp_b

def gp_sets(data):
    subcampaigns_number = data.shape[0]
    logger.debug(f'subcampaigns number: {subcampaigns_number}')
    logger.debug(f'data:\n{data}')

    gps = [init_crossval_gp(data[i]) for i in range(subcampaigns_number)]
    logger.debug(f'gps:\n{gps}')
   
    gps_a, gps_b = zip(*gps)  # gps[0], gps[1]  # [init_crossval_gp(data[i]) for i in range(subcampaigns_number)]
    logger.debug(f'gpsa:\n{gps_a}')
    logger.debug(f'gpsb:\n{gps_b}')
    return gps_a, gps_b

def cross_validate(data):

    bid = np.linspace(0, MAX_BID, SAMPLES_INVERSE).reshape(-1, 1)

    gps_a, gps_b = gp_sets(data)

    scl_a = [bid_maxrevenue_given_cost(bid, gp.revenue(bid), gp.cost(bid))
           for gp in gps_a]
    scl_b = [bid_maxrevenue_given_cost(bid, gp.revenue(bid), gp.cost(bid))
           for gp in gps_b]

    scl_a = np.array(scl_a)
    scl_b = np.array(scl_b)
    
    logger.debug(f'scl_a shape {scl_a.shape}')

    r_a, bid_mix_a, _, _ = optimize(scl_a)
    r_b, bid_mix_b, _, _ = optimize(scl_b)
    logger.info(f'the resulting bid_mix_a:\n{bid_mix_a}')
    logger.info(f'the resulting bid_mix_b:\n{bid_mix_b}')

    # assert all([r_a[i] <= r_a[i+1] for i in range(N_COST-1)])
    logger.debug(f'assert r_a monotonicity: {np.all([r_a[i] <= r_a[i+1] if ~np.isnan(r_a[i]) else True for i in range(N_COST-1)])}')
    logger.debug(f'r_a:\n{r_a}')
    logger.debug(f'r_b:\n{r_b}')
    # assert all([r_b[i] <= r_b[i+1] for i in range(N_COST-1)])
    logger.debug(f'assert r_b monotonicity: {np.all([r_b[i] <= r_b[i+1] if ~np.isnan(r_b[i]) else True for i in range(N_COST-1)])}')

    bid_mix_a = np.array(bid_mix_a)
    bid_mix_b = np.array(bid_mix_b)
    logger.debug(f'shape bid_mix: {bid_mix_a.shape}')
    
    cost_array = np.linspace(0, MAX_EXP, N_COST)
    r_a_on_b = np.empty((N_COST))
    
    logger.info('check bid_mix values')
    # scl_a = [bid_maxrevenue_given_cost(bid_mix_a[:,s], gp.revenue(bid_mix_a[:, s]), gp.cost(bid_mix_a[:,s])) for s, gp in enumerate(gps_b)]
    # scl_a = np.array(scl_a)
    # r_a, _, _, _ = optimize(scl_a)
    # scl_b = [bid_maxrevenue_given_cost(bid_mix_b[:,s], gp.revenue(bid_mix_b[:, s]), gp.cost(bid_mix_b[:,s])) for s, gp in enumerate(gps_a)]
    # scl_b = np.array(scl_b)
    # r_b, _, _, _ = optimize(scl_b)

    r_b_given_bid_a = np.array([sum([s.revenue(bid_mix_a[c, j]) if not np.isnan(bid_mix_a[c,j]) else 0.0 for j, s in enumerate(gps_b)]) for c in range(N_COST)]).squeeze()
    # r_b_given_bid_a_bis = np.empty((N_COST))
    # for c in range(N_COST):
    #     r_b_given_bid_a_bis[c] = sum([s.revenue(bid_mix_a[c,j]) if not np.isnan(bid_mix_a[c,j]) else 0.0 for j, s in enumerate(gps_b)])

    r_a_given_bid_b = np.array([sum([s.revenue(bid_mix_b[c, j]) if not np.isnan(bid_mix_b[c,j]) else 0.0 for j, s in enumerate(gps_a)]) for c in range(N_COST)]).squeeze()
    # assert all([r_a[i] <= r_a[i+1] for i in range(N_COST-1)])
    logger.debug(f'assert r_a monotonicity: {all([r_a[i] <= r_a[i+1] for i in range(N_COST-1)])}')
    # assert all([r_b[i] <= r_b[i+1] for i in range(N_COST-1)])
    logger.debug(f'assert r_b monotonicity: {all([r_b[i] <= r_b[i+1] for i in range(N_COST-1)])}')
    
    logger.debug(f'r_b_given_bid_a\n{r_b_given_bid_a}')
    logger.debug(f'r_b_given_bid_a shape\n {r_b_given_bid_a.shape}')
    # logger.debug(f'r_b_given_bid_a_bis\n {r_b_given_bid_a_bis}')
    logger.debug(f'assert monotonicity r_b_given_bid_a: {all([r_b_given_bid_a[i] <= r_b_given_bid_a[i+1] for i in range(N_COST-1)])}')
    # logger.debug(f'is equal r_b_given_bid_a and r__b_given_bid_a_bis? {np.array_equal(r_b_given_bid_a, r_b_given_bid_a_bis)}')

    # plot.plot(x=cost_array, y=np.mean([r_a, r_b], axis=0))
    # assert all([np.mean([r_a, r_b], axis=0)[i] == (r_a[i]+r_b[i])/2 for i in range(N_COST)])
    logger.debug(f'assert mean: {all([np.mean([r_a_given_bid_b, r_b_given_bid_a], axis=0)[i] == (r_a_given_bid_b[i]+r_b_given_bid_a[i])/2 for i in range(N_COST)])}')

    # plot.plot(x=cost_array, y=r_b_given_bid_a)
    # plot.plot(x=cost_array, y=np.mean([r_b_given_bid_a, r_a_given_bid_b], axis=0))

    r_a_given_bid_b = np.maximum.accumulate(r_a_given_bid_b)
    r_b_given_bid_a = np.maximum.accumulate(r_b_given_bid_a)
    r = np.mean([r_b_given_bid_a, r_a_given_bid_b], axis=0)
    roi = r[1:] / cost_array[1:]

    return r, roi

if __name__ == '__main__':
   import subcampaign as sc
   logging.basicConfig(level=logging.DEBUG,
                        # format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        # datefmt='%m-%d %H:%M',
                        filename='/tmp/mycvgpapp.log',
                        filemode='w')
   rng = np.random.default_rng()
   # subcampaign_a = np.array([[0, 1, 2, 3, 7, 4, 5], [2, 3, 4, 9, 5, 6, 7], [4, 9, 3, 0, 3, 4, 2]]).T
   # subcampaign_b = np.array([[0, 1, 2, 3, 7, 4, 5], [2, 3, 4, 9, 5, 6, 7], [4, 9, 3, 0, 3, 4, 2]]).T
   # data = np.stack((subcampaign_a,subcampaign_b))
   # data = np.full((5,4), rng.normal())

   campaign_a = [sc.Subcampaign(64, 0.3, 69, 0.2),
                 sc.Subcampaign(70, 0.4, 78, 0.2),
                 sc.Subcampaign(70, 0.3, 73, 0.1)]

   bid = np.linspace(0, MAX_BID, SAMPLES_INVERSE).reshape(-1, 1)
   obs = 30
   #data = np.empty()

   data = np.full((len(campaign_a), obs, 3), 0.0)
   print(data)
   print('data shape', data.shape)
   for i, s in enumerate(campaign_a):
       bids = rng.choice(bid, obs)
       print('bids shape', bids.shape)
       bids = bids
       print('bids', bids)
       print('bids shape', bids.shape)
       costs = s.cost(bids)
       revenues = s.revenue(bids)
       print('c', costs)
       print('r', revenues)
       print('shape c', costs.shape)
       print('shape r', revenues.shape)
       val = np.hstack((bids, costs, revenues))
       print('val shape', val.shape)
       data[i, :, :] = val  # np.stack((bids, costs, revenues), axis=1)
       print(data)
       #data = np.stack(data, sc_data)
       # data = np.stack(bids, subc.cost(bids), subc.revenue(bids)) for bids in rng.choice(bid, (10, 1)), 

   print(data)
   # gp_sets(data)
   crvmean, crv_givenbids = cross_validate(data)

   cost_array = np.linspace(0, MAX_EXP, N_COST)
   plot.plot(x=cost_array, y=crvmean)
   plot.plot(x=cost_array[1:], y=crv_givenbids)
    # plot.plot(x=cost_array, y=np.mean([r_b_given_bid_a, r_a_given_bid_b], axis=0))
   # init_crossval_gp(data)

