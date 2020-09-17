from config import *
import logging
import numpy as np


logger = logging.getLogger(__name__)


class Subcampaign:
    """
    Class to model a subcampaign
    """

    def __init__(self, asymptotic_cost=20.0, slope_cost=1.0,
                 asymptotic_rev=30.0, slope_rev=1.0, noise_sd_cost=None, noise_sd_rev=None, snr=25):
        """
        Initializer

        Args:
            asymptotic_cost: upper bound for the cost (comulative daily cost)
            slope_cost: cost slope - the smaller, the steeper
            asymptotic_rev: upper bound for the revenue
            slope_rev: revenue slope
            noise_sd: noise level - noise standard deviation
        """
        self.asymptotic_cost = asymptotic_cost  # upper bound
        self.slope_cost = slope_cost  # slope: the smaller, the steeper
        self.asymptotic_rev = asymptotic_rev
        self.slope_rev = slope_rev
        self.noise_sd_cost = noise_sd_cost
        self.noise_sd_rev = noise_sd_rev
        self.snr = snr

        if noise_sd_cost == None:
            bid = np.linspace(0., MAX_BID, N_BID)
            self.noise_sd_cost = 0. #noise_sd_cost
            cost = self.cost(bid, noise=False)
            self.noise_sd_cost = self._noise(cost, snr)

        if noise_sd_rev == None:
            bid = np.linspace(0., MAX_BID, N_BID)
            self.noise_sd_rev = 0. #noise_sd_rev
            revenue = self.revenue(bid, noise=False)
            self.noise_sd_rev = self._noise(revenue, snr)

    def cost(self, bid, noise=True):
        return (self.asymptotic_cost*(1 - np.exp(-bid/self.slope_cost))
                + noise*np.random.normal(0, self.noise_sd_cost, bid.shape))

    def revenue(self, bid, noise=True):
        return (self.asymptotic_rev * (1 - np.exp(-bid/self.slope_rev))
                + noise*np.random.normal(0, self.noise_sd_rev, bid.shape))

    def bid(self, cost):
        return - self.slope_cost*np.log(1 - cost/self.asymptotic_cost)

    def _noise(self, signal, snr=20):
        # Calculate signal power and convert to dB 
        sig_pwr = signal ** 2
        sig_mean_pwr = np.mean(sig_pwr)
        sig_mean_db = 10 * np.log10(sig_mean_pwr)
        noise_mean_db = sig_mean_db - snr 
        noise_mean_pwr = 10 ** (noise_mean_db / 10)
        noise_sd = np.sqrt(noise_mean_pwr)
        return noise_sd

    def __str__(self):
        #return str(self.__class__) + ": " + str(self.__dict__)
        return str(self.__dict__)


def random_campaign(N=5):
        campaign = []
        from numpy.random import default_rng
        rng_snr = np.random.default_rng()
        rng_cost = np.random.default_rng()
        rng_rev = np.random.default_rng()

        for i in range(N):
            asymptotic_cost = float(rng_cost.integers(50, 101))
            asymptotic_rev = float(rng_rev.integers(400, 701))

            # To sample Unif[a, b), b > a multiply the output of random by (b-a) and add a:
            # (b - a) * random() + a
            slope_cost = (1.1 - 0.2) * rng_cost.random() + 0.2
            slope_rev = (slope_cost - 0.2) * rng_rev.random() + 0.2 

            snr = rng_snr.integers(30, 36)
            campaign.append(Subcampaign(asymptotic_cost, slope_cost, asymptotic_rev, slope_rev, snr=snr))

        return campaign



if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)

    # campaign_a = [Subcampaign(59, 1.3, 25, 1.0), Subcampaign(20, 2.4, 30, 2.2),
    #               Subcampaign(20, 1.3, 35, 1.2)]
    # campaign_a = [Subcampaign(59, 1.3, 63, 1.0),
    #               Subcampaign(20, 2.4, 30, 2.2),
    #               Subcampaign(20, 1.3, 35, 1.2)]

    campaign_a = [Subcampaign(64, 0.3, 265, 0.2),
                  Subcampaign(70, 0.4, 355, 0.2),
                  Subcampaign(70, 0.3, 326, 0.1),
                  Subcampaign(50, 0.3, 185, 0.2),
                  Subcampaign(55, 0.4, 208, 0.1),
                  Subcampaign(120, 0.4, 724, 0.2),
                  Subcampaign(100, 0.3, 669, 0.25),
                  Subcampaign(90, 0.34, 595, 0.15),
                  Subcampaign(95, 0.38, 616, 0.19),
                  Subcampaign(110, 0.4, 675, 0.12)]
    # A2
    campaign_a2 = [Subcampaign(64, 0.5, 265, 0.3),
                  Subcampaign(70, 0.7, 355, 0.5),
                  Subcampaign(70, 0.6, 326, 0.4),
                  Subcampaign(50, 0.6, 185, 0.4),
                  Subcampaign(55, 0.7, 208, 0.6),
                  Subcampaign(120, 0.7, 724, 0.5),
                  Subcampaign(100, 0.5, 669, 0.35),
                  Subcampaign(90, 0.53, 595, 0.37),
                  Subcampaign(95, 0.55, 616, 0.40),
                  Subcampaign(110, 0.6, 675, 0.43)]
    # B
    campaign_b = [Subcampaign(64, 2.3, 265, 1.2),
                  Subcampaign(70, 2.4, 355, 1.2),
                  Subcampaign(70, 2.3, 326, 1.1),
                  Subcampaign(50, 2.3, 185, 1.2),
                  Subcampaign(55, 2.4, 208, 1.1),
                  Subcampaign(120, 2.4, 724, 1.2),
                  Subcampaign(100, 2.3, 669, 1.25),
                  Subcampaign(90, 2.34, 595, 1.15),
                  Subcampaign(95, 2.38, 616, 1.19),
                  Subcampaign(110, 2.4, 675, 1.12)]
    # C
    campaign_c = [Subcampaign(64, 4.3, 265, 1.2),
                  Subcampaign(70, 4.4, 355, 1.2),
                  Subcampaign(70, 4.3, 326, 1.1),
                  Subcampaign(50, 4.3, 185, 1.2),
                  Subcampaign(55, 4.4, 208, 1.1),
                  Subcampaign(120, 4.4, 724, 1.2),
                  Subcampaign(100, 4.3, 669, 1.25),
                  Subcampaign(90, 4.34, 595, 1.15),
                  Subcampaign(95, 4.38, 616, 1.19),
                  Subcampaign(110, 4.4, 675, 1.12)]
    # # D
    campaign_d = [Subcampaign(64, 0.3, 265, 4.2),
                  Subcampaign(70, 0.4, 355, 4.2),
                  Subcampaign(70, 0.3, 326, 4.1),
                  Subcampaign(50, 0.3, 185, 4.2),
                  Subcampaign(55, 0.4, 208, 4.1),
                  Subcampaign(120, 0.4, 724, 4.2),
                  Subcampaign(100, 0.3, 669, 4.25),
                  Subcampaign(90, 0.34, 595, 4.15),
                  Subcampaign(95, 0.38, 616, 4.19),
                  Subcampaign(110, 0.4, 675, 4.12)]

    campaign_e = [Subcampaign(164, 0.81, 747, .65),
                  Subcampaign(170, .94, 795, 0.72),
                  Subcampaign(170, 0.75, 786, 0.59)]  #,

    campaign_f = [Subcampaign(160, 0.65, 847, .45, snr=50),
                  Subcampaign(170, .62, 895, 0.42, snr=50),
                  Subcampaign(170, 0.69, 886, 0.49, snr=50)] #,

    # campaign_g = [Subcampaign(60, 0.65, 447, .45, snr=50),
    #               Subcampaign(70, .62, 495, 0.42, snr=50),
    #               Subcampaign(75, .67, 503, 0.43, snr=50),
    #               Subcampaign(65, .68, 473, 0.47, snr=50),
    #               Subcampaign(70, 0.69, 486, 0.49, snr=50)] #,
    # campaign_g = [Subcampaign(60, 0.65, 497, .41, snr=30),
    #               Subcampaign(77, .62, 565, 0.48, snr=30),
    #               Subcampaign(75, .67, 573, 0.43, snr=30),
    #               Subcampaign(65, .68, 503, 0.47, snr=30),
    #               Subcampaign(70, 0.69, 536, 0.40, snr=30)] #,
    campaign = random_campaign()

    for i, s in enumerate(campaign):
        print(f'\t\tsc.Subcampaign({s.asymptotic_cost}, {s.slope_cost}, {s.asymptotic_rev}, {s.slope_rev}, snr={s.snr}){"," if i < 4 else ""}')
    import matplotlib.pyplot as plt

    for i, s in enumerate(campaign):
        print(f'\tnoise cost {i}: {s.noise_sd_cost}')
        print(f'\tnoise rev {i}: {s.noise_sd_rev}')
        plt.plot(np.linspace(0.0, 2.0, 201), s.cost(np.linspace(0.0, 2.0, 201), noise=False))
        #         name=f'campaign_g_{i}noiseless_cost'))
        plt.plot(np.linspace(0.0, 2.0, 201), s.cost(np.linspace(0.0, 2.0, 201), noise=True))
        #         name=f'campaign_g_{i}noise_cost'))
        plt.show()
        plt.plot(np.linspace(0.0, 2.0, 201), s.revenue(np.linspace(0.0, 2.0, 201), noise=False))
        #         name=f'campaign_g_{i}noiseless_rev'))
        plt.plot(np.linspace(0.0, 2.0, 201), s.revenue(np.linspace(0.0, 2.0, 201), noise=True))
        #         name=f'campaign_g_{i}noise_rev'))
        plt.show()

    print('rev 0.01:', campaign_e[0].revenue(np.array([0.01]), noise=False))
    print('cost 0.01:', campaign_e[0].cost(np.array([0.01]), noise=False))
    print('rev/cost 0.01:', campaign_e[0].revenue(np.array([0.01]), noise=False)/campaign_e[0].cost(np.array([0.01]), noise=False))

    print('rev 1.51:', campaign_e[0].revenue(np.array([1.51]), noise=False))
    print('cost 1.51:', campaign_e[0].cost(np.array([1.51]), noise=False))
    print('rev/cost 1.51:', campaign_e[0].revenue(np.array([1.51]), noise=False)/campaign_e[0].cost(np.array([1.51]), noise=False))
