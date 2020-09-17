import gp_model
import logging
import numpy as np
import plot
import subcampaign as sc
from config import *
from gpflow.utilities import tabulate_module_summary, print_summary, parameter_dict


def learn_hp(s, n):
    bids = np.random.uniform(0., MAX_BID, 1101)
    gp = gp_model.GPModel(OPT=True)

    costs = s.cost(bids)
    revs = s.revenue(bids)

    gp.update(bids, costs, revs)

    print(f'summary model_cost {n}')
    print_summary(gp.model_cost)
    print(f'summary model_rev {n}')
    print_summary(gp.model_rev)

    print('\'cost_variance\': ', gp.model_cost.kernel.variance)
    print('\'rev_variance\': ', gp.model_rev.kernel.variance)
    print('\'cost_lengthscales\': ', gp.model_cost.kernel.lengthscales)
    print('\'rev_lengthscales\': ', gp.model_rev.kernel.lengthscales)
    print('\'cost_likelihood\': ', gp.model_cost.likelihood.variance)
    print('\'rev_likelihood\': ', gp.model_rev.likelihood.variance)


if __name__ == '__main__':
    import time
    start = time.time()

    logging.basicConfig(level=logging.ERROR,
    #                     # format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    #                     # datefmt='%m-%d %H:%M',
                         filename='/tmp/learngphp.log',
                         filemode='w')
    campaign_f = [sc.Subcampaign(160, 0.65, 847, .45, snr=50),
                  sc.Subcampaign(170, .62, 895, 0.42, snr=50),
                  sc.Subcampaign(170, 0.69, 886, 0.49, snr=50)] #,

    # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=29),
    #               sc.Subcampaign(77, .62, 565, 0.48, snr=29),
    #               sc.Subcampaign(75, .67, 573, 0.43, snr=29),
    #               sc.Subcampaign(65, .68, 503, 0.47, snr=29),
    #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=29)] #,


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

    campaign = rnd_campaign_10

    for i, s in enumerate(campaign):
        learn_hp(s, i)
    end = time.time()
    print('elapsed time:', end - start)
