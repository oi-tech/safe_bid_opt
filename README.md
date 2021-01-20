# Safe Bid Optimization with Return-On-Investment Constraints

The code for optimizing the bids of online advertising campaigns while respecting Return-On-Investment (ROI) and budget constraints.

## Overview

An online advertising campaign is composed of subcampaigns, where the advertiser's goal is to set bids that maximize the overall cumulative expected revenue, while keeping the overall ROI above a fixed value and the overall budget below a daily value.
`agents.py` models the learning policies of the advertisers, that determine the bids for every subcampaign (with the method `bid_choice`) given the estimates of the unknown subcampaigns' functions (modeled in `subcampaign.py`) using Gaussian Processes (modeled in `gp_model.py`), optimizing the bids leveraging on the functions defined in `optimize.py`.

`config.py` sets the advertisers constraints, discretizations' ranges and values:
* `MIN_ROI`: the ROI constraint.
* `MAX_BID`,`MAX_EXP`, `MAX_REV`: the maximum bid, the budget constraint, and the maximum discretized revenue.
* `N_BID`. `N_COST`, `N_REV`: the number of bid, cost and revenue values.

`environment.py` provides functions to simulate an online advertising campaign.  
The function `environment` simulates a campaign given a list of subcampaigns, a list of agents, and a time horizon `T`, saving the results in pickle format.  

## Requirements

  * matplotlib==3.1.3
  * gpflow==2.0.5
  * tikzplotlib==0.9.4
  * tf nightly==2.2.0.dev20200308
  * numpy==1.18.1
  * tensorflow probability==0.10.0
  * scikit learn==0.23.2
  * tensorflow==2.3.0
