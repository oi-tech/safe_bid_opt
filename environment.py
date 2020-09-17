import agent
import cross_validation as crv
import itertools
import logging
import numpy as np
import pathlib
import plot
import pickle as pl
import subcampaign as sc
import time
import tensorflow as tf
from config import *
from multiprocessing.dummy import Pool
import uuid
from gpflow.utilities import tabulate_module_summary
import copy
import matplotlib.pyplot as plt
#import seaborn as sns
import tikzplotlib
#sns.set()

logger = logging.getLogger(__name__)

def analize_runs(settings, results, path=None, Show=False, Save=True):
    subcampaigns, agents, runs_num, T = settings

    agents_to_file = [agents[i].replace(" ", "_").replace("%", "_") for i in range(len(agents))]
    
    X = np.full((len(agents), len(subcampaigns), T+1), 0.0)
    total_rev = np.empty((len(agents), T, runs_num))
    total_cost = np.empty((len(agents), T, runs_num))
    total_exp_rev = np.empty((len(agents), T, runs_num))
    total_exp_cost = np.empty((len(agents), T, runs_num))
    total_exp_roi = np.empty((len(agents), T, runs_num))
    total_real_roi = np.empty((len(agents), T, runs_num))
    cumulative_rev = np.empty((len(agents), T, runs_num))
    cumulative_cost = np.empty((len(agents), T, runs_num))

    mean_rev = np.empty((len(agents), T))
    mean_cumulative_rev = np.empty((len(agents), T))
    mean_cost = np.empty((len(agents), T))
    mean_cumulative_cost = np.empty((len(agents), T))
    sigma_rev = np.empty((len(agents), T))
    sigma_cumulative_rev = np.empty((len(agents), T))
    sigma_cost = np.empty((len(agents), T))
    sigma_cumulative_cost = np.empty((len(agents), T))
    mean_exp_roi = np.empty((len(agents), T))
    sigma_exp_roi = np.empty((len(agents), T))
    mean_real_roi = np.empty((len(agents), T))
    sigma_real_roi = np.empty((len(agents), T))

    percentile90_rev = np.empty((len(agents), T))
    percentile90_cumrev = np.empty((len(agents), T))
    percentile90_cost = np.empty((len(agents), T))
    percentile90_cumcost = np.empty((len(agents), T))
    percentile90_real_roi = np.empty((len(agents), T))
    percentile50_rev = np.empty((len(agents), T))
    percentile50_cumrev = np.empty((len(agents), T))
    percentile50_cost = np.empty((len(agents), T))
    percentile50_cumcost = np.empty((len(agents), T))
    percentile50_real_roi = np.empty((len(agents), T))
    percentile10_rev = np.empty((len(agents), T))
    percentile10_cumrev = np.empty((len(agents), T))
    percentile10_cost = np.empty((len(agents), T))
    percentile10_cumcost = np.empty((len(agents), T))
    percentile10_real_roi = np.empty((len(agents), T))
    roi_violation = np.empty((len(agents), T))
    roi_violation_days = np.empty((len(agents)))
    roi_per_run_violations = np.empty((len(agents), runs_num))
    cost_per_run_violations = np.empty((len(agents), runs_num))
    cost_violation = np.empty((len(agents), T))
    cost_violation_days = np.empty((len(agents)))

    timeline = T - INITIAL_BIDS + 1
    cumulative_rev_shifted = np.empty((len(agents), timeline, runs_num))
    percentile10_cumrev_shifted = np.empty((len(agents), timeline))
    percentile50_cumrev_shifted = np.empty((len(agents), timeline))
    percentile90_cumrev_shifted = np.empty((len(agents), timeline))
    mean_cumulative_rev_shifted = np.empty((len(agents), timeline))
    sigma_cumulative_rev_shifted = np.empty((len(agents), timeline))

    print('T', T)
    print('subcampaigns', subcampaigns)
    print('agents', agents)
    print('runs', runs_num)
    # return
    c = agent.Clairvoyant_Agent(subcampaigns)
    X_opt = c.bid_choice()
    if logger.isEnabledFor(logging.WARNING):
        logger.warning(f'optimal bid_mix:\n{X_opt}')
    optimum_rev = sum([s.revenue(X_opt[i], noise=False) for i, s in enumerate(subcampaigns)])
    optimum_rev = np.full((T), optimum_rev)
    optimum_cost = sum([s.cost(X_opt[i], noise=False) for i, s in enumerate(subcampaigns)])
    optimum_cost = np.full((T), optimum_cost)
    target_cost = np.full((T), MAX_EXP)
    clairvoyant_roi = optimum_rev/optimum_cost 
    target_roi = np.full((T), MIN_ROI)

    for i, run in enumerate(results):
        X, Y_cost, Y_rev, f_cost, f_rev, exp_roi, exp_cost, exp_rev = run
        # show_results(subcampaigns, agents, X, Y_cost, Y_rev, f_cost, f_rev, exp_roi, exp_cost, exp_rev)
        # print('X', X)
        for a in range(len(agents)):
            # print('X[a]', X[a])
            # print('f_rev', f_rev[a])
            for t in range(T):
            # total_rev[i][t] = np.sum(f_rev[i, :, t])
                total_rev[a][t][i] = np.sum(f_rev[a, : ,t])
                total_exp_rev[a][t][i] = np.sum(Y_rev[a, : ,t])
                total_cost[a][t][i] = np.sum(f_cost[a, : ,t])
                total_exp_cost[a][t][i] = np.sum(Y_cost[a, : ,t])
                print('exp_roi', exp_roi[a,:,t])
                print('exp_roi', exp_roi[a,0,t])
                total_exp_roi[a][t][i] = exp_roi[a,0,t]
                total_real_roi[a][t][i] = total_rev[a][t][i] / total_cost[a][t][i]
            print(f'total_rev, run{i}:\n{total_rev[a]}')

    # print(f'total_rev\n:{total_rev}')
    max_rev = np.nanmax(total_rev) if np.nanmax(total_rev) > np.nanmax(optimum_rev) else np.nanmax(target_rev)
    min_rev = np.nanmin(total_rev) #if np.nanmax(total_rev) > np.nanmax(target_rev) else np.nanmax(target_rev)
    min_rev -= min_rev/100
    min_cost = np.nanmin(total_cost) 
    min_cost -= min_cost/100
    max_cost = np.nanmax(total_cost) if np.nanmax(total_cost) > np.nanmax(target_cost) else np.nanmax(target_cost)
    max_real_roi = np.nanmax(total_real_roi) if np.nanmax(total_real_roi) > np.nanmax(clairvoyant_roi) else np.nanmax(clairvoyant_roi)
    min_real_roi = np.nanmin(total_real_roi)
    min_roi = LOW_ROI
    # timeline = T - INITIAL_BIDS + 1
    timeline = T - INITIAL_BIDS
    # half_timeline = (T - INITIAL_BIDS + 1) // 2
    half_timeline = (T - INITIAL_BIDS) // 2
    # half_timeline_shifted = half_timeline + INITIAL_BIDS -1
    half_timeline_shifted = half_timeline + INITIAL_BIDS

    for a in range(len(agents)):
        print(f'total_rev{a}\n:{total_rev[a]}')
        mean_rev[a] = np.mean(total_rev[a], axis=1, dtype=np.float64)
        print('mean_rev', mean_rev)
        percentile90_rev[a] = np.percentile(total_rev[a], 90, axis=1)
        print('percentile90_rev', percentile90_rev)
        percentile50_rev[a] = np.percentile(total_rev[a], 50, axis=1)
        print('percentile50_rev', percentile50_rev)
        percentile10_rev[a] = np.percentile(total_rev[a], 10, axis=1)
        print('percentile10_rev', percentile10_rev)

        cumulative_rev[a] = np.cumsum(total_rev[a], axis=0)
        print('cumulative_rev', cumulative_rev[a])
        percentile90_cumrev[a] = np.percentile(cumulative_rev[a], 90, axis=1)
        percentile50_cumrev[a] = np.percentile(cumulative_rev[a], 50, axis=1)
        print('percentile50_cumrev', percentile50_cumrev[a])
        percentile10_cumrev[a] = np.percentile(cumulative_rev[a], 10, axis=1)
        mean_cumulative_rev[a] = np.mean(cumulative_rev[a], axis=1, dtype=np.float64)
        sigma_cumulative_rev[a] = np.std(cumulative_rev[a], axis=1, dtype=np.float64)

        cumulative_rev_shifted[a] = np.cumsum(total_rev[a, INITIAL_BIDS - 1:, :], axis=0)
        print('cumulative_rev_shifted', cumulative_rev_shifted)
        percentile90_cumrev_shifted[a] = np.percentile(cumulative_rev_shifted[a, :, :], 90, axis=1)
        percentile50_cumrev_shifted[a] = np.percentile(cumulative_rev_shifted[a, :, :], 50, axis=1)
        percentile10_cumrev_shifted[a] = np.percentile(cumulative_rev_shifted[a, :, :], 10, axis=1)
        mean_cumulative_rev_shifted[a] = np.mean(cumulative_rev_shifted[a, :, :], axis=1, dtype=np.float64)
        sigma_cumulative_rev_shifted[a] = np.std(cumulative_rev_shifted[a, :, :], axis=1, dtype=np.float64)

        mean_cost[a] = np.mean(total_cost[a], axis=1, dtype=np.float64)
        print('mean_cost', mean_cost)
        percentile90_cost[a] = np.percentile(total_cost[a], 90, axis=1)
        print('percentile90_cost', percentile90_cost)
        percentile50_cost[a] = np.percentile(total_cost[a], 50, axis=1)
        print('percentile50_cost', percentile50_cost)
        percentile10_cost[a] = np.percentile(total_cost[a], 10, axis=1)
        print('percentile10_cost', percentile10_cost)
        
        cumulative_cost[a] = np.cumsum(total_cost[a], axis=0)
        percentile90_cumcost[a] = np.percentile(cumulative_cost[a], 90, axis=1)
        percentile50_cumcost[a] = np.percentile(cumulative_cost[a], 50, axis=1)
        percentile10_cumcost[a] = np.percentile(cumulative_cost[a], 10, axis=1)
        mean_cumulative_cost[a] = np.mean(cumulative_cost[a], axis=1, dtype=np.float64)
        sigma_cumulative_cost[a] = np.std(cumulative_cost[a], axis=1, dtype=np.float64)

        sigma_rev[a] = np.std(total_rev[a], axis=1, dtype=np.float64)
        print('sigma_rev', sigma_rev)
        sigma_cost[a] = np.std(total_cost[a], axis=1, dtype=np.float64)
        print('sigma_cost', sigma_cost)
        mean_exp_roi[a] = np.mean(total_exp_roi[a], axis=1, dtype=np.float64)
        print('mean_exp_roi', mean_exp_roi[a], )
        sigma_exp_roi[a] = np.std(total_exp_roi[a], axis=1, dtype=np.float64)
        print('sigma_exp_roi', sigma_exp_roi[a])

        mean_real_roi[a] = np.mean(total_real_roi[a], axis=1, dtype=np.float64)
        print('total_real_roi', total_real_roi[a])
        print('mean_real_roi', mean_real_roi[a])
        percentile90_real_roi[a] = np.percentile(total_real_roi[a], 90, axis=1)
        print('percentile90_real_roi', percentile90_real_roi)
        percentile50_real_roi[a] = np.percentile(total_real_roi[a], 50, axis=1)
        print('percentile50_real_roi', percentile50_real_roi)
        percentile10_real_roi[a] = np.percentile(total_real_roi[a], 10, axis=1)
        print('percentile10_real_roi', percentile10_real_roi)

        sigma_real_roi[a] = np.std(total_real_roi[a], axis=1, dtype=np.float64)
        print('sigma_real_roi', sigma_real_roi[a])
        roi_violation[a] = np.count_nonzero((total_real_roi[a]<MIN_ROI)&(np.isfinite(total_real_roi[a])), axis=1)/runs_num
        print('roi_violation', roi_violation[a])
        cost_violation[a] = np.count_nonzero((total_cost[a]>MAX_EXP)&(np.isfinite(total_cost[a])), axis=1)/runs_num
        # roi_violation_days[a] = np.count_nonzero(roi_violation[a, INITIAL_BIDS -1:])  #, axis=1)
        roi_violation_days[a] = np.count_nonzero(roi_violation[a, INITIAL_BIDS:])  #, axis=1)
        # cost_violation_days[a] = np.count_nonzero(cost_violation[a, INITIAL_BIDS -1:])  # , axis=1)
        cost_violation_days[a] = np.count_nonzero(cost_violation[a, INITIAL_BIDS:])  # , axis=1)

        for i, run in enumerate(results):
            # roi_per_run_violations[a, i] = np.count_nonzero((total_real_roi[a, INITIAL_BIDS -1:, i] < MIN_ROI)) # &(np.isfinite(total_real_roi[a, INITIAL_BIDS - 1:, i])))
            roi_per_run_violations[a, i] = np.count_nonzero((total_real_roi[a, INITIAL_BIDS:, i] < MIN_ROI)) # &(np.isfinite(total_real_roi[a, INITIAL_BIDS - 1:, i])))
            # cost_per_run_violations[a, i] = np.count_nonzero((total_cost[a, INITIAL_BIDS -1:, i] > MAX_EXP)) # &(np.isfinite(total_cost[a,INITIAL_BIDS - 1:, i])))
            cost_per_run_violations[a, i] = np.count_nonzero((total_cost[a, INITIAL_BIDS:, i] > MAX_EXP)) # &(np.isfinite(total_cost[a,INITIAL_BIDS - 1:, i])))

    print('VALUES')
    print(f'T:{T}\n')
    print(f'ROI target: {MIN_ROI}\n\n')
    print(f'timeline: {timeline}')
    print(f'half_timeline: {half_timeline}\n')
    print(f'half_timeline_shifted: {half_timeline_shifted}')
    print(f'INITIAL_BIDS:{INITIAL_BIDS}')
    for a in range(len(agents)):
        print(f'{agents[a]}')
        print(f'cumulated reward mean at T/2: {mean_cumulative_rev[a, half_timeline_shifted] - mean_cumulative_rev[a, INITIAL_BIDS-1] + mean_rev[a, INITIAL_BIDS-1]}\n' 
              f'cumulated reward mean at T: {mean_cumulative_rev[a, -1] - mean_cumulative_rev[a, INITIAL_BIDS-1] + mean_rev[a, INITIAL_BIDS-1]}\n' 
              f'cumulated reward sigma at T: {sigma_cumulative_rev[a, -1]}\n' 
              f'cumulated reward sigma at T/2: {sigma_cumulative_rev[a, half_timeline_shifted]}\n' 
              f'50p cumulated reward at T: {percentile50_cumrev[a, -1] - percentile50_cumrev[a, INITIAL_BIDS -1] + mean_rev[a, INITIAL_BIDS-1]}\n'
              f'50p cumulated reward at T/2: {percentile50_cumrev[a, half_timeline_shifted] - percentile50_cumrev[a, INITIAL_BIDS-1] + mean_rev[a, INITIAL_BIDS-1]}\n'
              f'90p cumulated reward at T: {percentile90_cumrev[a, -1] - percentile90_cumrev[a, INITIAL_BIDS -1] + mean_rev[a, INITIAL_BIDS-1]}\n'
              f'90p cumulated reward at T/2: {percentile90_cumrev[a, half_timeline_shifted] - percentile90_cumrev[a, INITIAL_BIDS -1] + mean_rev[a, INITIAL_BIDS-1]}\n'
              f'10p cumulated reward at T: {percentile10_cumrev[a, -1] - percentile10_cumrev[a, INITIAL_BIDS -1] + mean_rev[a, INITIAL_BIDS-1]}\n'
              f'10p cumulated reward at T/2: {percentile10_cumrev[a, half_timeline_shifted] - percentile10_cumrev[a, INITIAL_BIDS -1] + mean_rev[a, INITIAL_BIDS-1]}\n'
              f'days roi violations per run mean : {np.mean(roi_per_run_violations[a])}\n'
              f'rate of days roi violations per run mean : {np.mean(roi_per_run_violations[a])/timeline}\n'
              f'days cost violations per run mean : {np.mean(cost_per_run_violations[a])}\n'
              f'rate of days cost violations per run mean : {np.mean(cost_per_run_violations[a])/timeline}\n\n'
              f'roi violation days: {roi_violation_days[a]}\n'
              f'roi violation days rate: {roi_violation_days[a]/timeline}\n'
              f'cost violation days: {cost_violation_days[a]}\n'
              f'cost violation days rate: {cost_violation_days[a]/timeline}\n'
              f'cumulated cost mean at T/2: {mean_cumulative_cost[a, half_timeline]}\n' 
              f'cumulated cost mean at T: {mean_cumulative_cost[a, -1]}\n' 
              f'cumulated cost sigma at T: {sigma_cumulative_cost[a, -1]}\n' 
              f'cumulated cost sigma at T/2: {sigma_cumulative_cost[a, half_timeline]}\n' 
              f'50p cumulated cost at T: {percentile50_cumcost[a, -1]}\n'
              f'50p cumulated cost at T/2: {percentile50_cumcost[a, half_timeline]}\n'
              f'90p cumulated cost at T: {percentile90_cumcost[a, -1]}\n'
              f'90p cumulated cost at T/2: {percentile90_cumcost[a, half_timeline]}\n'
              f'10p cumulated cost at T: {percentile10_cumcost[a, -1]}\n'
              f'10p cumulated cost at T/2: {percentile10_cumcost[a, half_timeline]}\n'
              f'shifted cumulated reward mean at T/2: {mean_cumulative_rev_shifted[a, half_timeline]}\n' 
              f'shifted cumulated reward mean at T: {mean_cumulative_rev_shifted[a, -1]}\n' 
              f'shifted cumulated reward sigma at T: {sigma_cumulative_rev_shifted[a, -1]}\n' 
              f'shifted cumulated reward sigma at T/2: {sigma_cumulative_rev_shifted[a, half_timeline]}\n' 
              f'shifted 50p cumulated reward at T: {percentile50_cumrev_shifted[a, -1]}\n'
              f'shifted 50p cumulated reward at T/2: {percentile50_cumrev_shifted[a, half_timeline]}\n'
              f'shifted 90p cumulated reward at T: {percentile90_cumrev_shifted[a, -1]}\n'
              f'shifted 90p cumulated reward at T/2: {percentile90_cumrev_shifted[a, half_timeline]}\n'
              f'shifted 10p cumulated reward at T: {percentile10_cumrev_shifted[a, -1]}\n'
              f'shifted 10p cumulated reward at T/2: {percentile10_cumrev_shifted[a, half_timeline]}\n'
              )


    max_rev = np.nanmax(mean_rev+2*sigma_rev) if np.nanmax(mean_rev+2*sigma_rev) > max_rev else max_rev
    max_rev += max_rev/100
    max_cost = np.nanmax(mean_cost+2*sigma_cost) if np.nanmax(mean_cost+2*sigma_cost) > max_cost else max_cost
    max_cost += max_cost/100
    max_real_roi = np.nanmax(mean_real_roi+2*sigma_real_roi) if np.nanmax(mean_real_roi+2*sigma_real_roi) > max_real_roi else max_real_roi
    max_real_roi += max_real_roi/100
    max_exp_roi = np.nanmax(mean_exp_roi+2*sigma_exp_roi)
    max_roi = max_real_roi if max_real_roi > max_exp_roi else max_exp_roi
    max_roi += max_roi/100
     
    # plots
    for a in range(len(agents)):
        fig_handle = plt.figure()
        plt.plot(np.linspace(0, T, T), mean_rev[a], 'b-', label=f'{agents[a]} revenue') 
        plt.plot(np.linspace(0, T, T), optimum_rev, 'r:', label=r'Optimum revenue')
        plt.fill_between(np.linspace(0, T, T), mean_rev[a] + sigma_rev[a] * 1.96, mean_rev[a] - sigma_rev[a] * 1.96, alpha=0.2)
        for r in range(runs_num):
            plt.plot(np.linspace(0, T, T), total_rev[a, :, r], "C0", linewidth=0.5)
        x_label = 'time'
        y_label = 'revenue'
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.gca().legend()
        bottom, top = plt.ylim()
        plt.ylim(min_rev, max_rev)
        if Show:
            plt.show()
        if  Save:
            name = f'cumulative_revenue_{agents[a]}'
            plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
            pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
            tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
            plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
            pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
            tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

        plt.close()

        fig_handle = plt.figure()
        plt.plot(np.linspace(0, T, T), mean_cost[a], 'b-', label=f'{agents[a]} cost') 
        plt.plot(np.linspace(0, T, T), optimum_cost, 'r:', label=r'Optimum cost' if MIN_ROI > 0. else 'Target cost')
        plt.fill_between(np.linspace(0, T, T), mean_cost[a] + sigma_cost[a] * 1.96, mean_cost[a] - sigma_cost[a] * 1.96, alpha=0.2)
        for r in range(runs_num):
            plt.plot(np.linspace(0, T, T), total_cost[a, :, r], "C0", linewidth=0.5)
        x_label = 'time'
        y_label = 'cost'
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.gca().legend()
        bottom, top = plt.ylim()
        plt.ylim(min_cost, max_cost)
        if Show:
            plt.show()
        if  Save:
            name = f'cumulative_cost{agents[a]}'
            plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
            pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
            tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
            plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
            pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
            tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

        plt.close()

    fig_handle = plt.figure()
    plt.plot(np.linspace(0, T, T), optimum_rev, 'r:', label=r'Optimum revenue')
    for a in range(len(agents)):
        plt.plot(np.linspace(0, T, T), mean_rev[a], label=f'{agents[a]} revenue') 
    x_label = 'time'
    y_label = 'revenue'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().legend()
    bottom, top = plt.ylim()
    plt.ylim(min_rev, max_rev)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_cumulative_revenue'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()

    fig_handle = plt.figure()
    # try axhline
    plt.plot(np.linspace(0, T, T), optimum_cost, 'r:', label=r'Optimum cost' if MIN_ROI > 0. else 'Target cost')
    for a in range(len(agents)):
        plt.plot(np.linspace(0, T, T), mean_cost[a], label=f'{agents[a]} cost') 
    x_label = 'time'
    y_label = 'cost'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().legend()
    bottom, top = plt.ylim()
    plt.ylim(min_cost, max_cost)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_cumulative_cost'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()


    fig_handle = plt.figure()
    for a in range(len(agents)):
        plt.plot(np.linspace(0, T, T), mean_rev[a])  #, label=f'{agents[a]} cost') 
        plt.fill_between(np.linspace(0, T, T), mean_rev[a] + sigma_rev[a] * 1.96, mean_rev[a] - sigma_rev[a] * 1.96, alpha=0.6, label=f'{agents[a]} revenue')
    plt.plot(np.linspace(0, T, T), optimum_rev, 'r:', label=r'Optimum revenue', linewidth=3)
    x_label = 'time'
    y_label = 'revenue'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().legend()
    bottom, top = plt.ylim()
    plt.ylim(min_rev, max_rev)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_confidence_int_rev'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()

    # # roi: expected vs real ROI

    # for a in range(len(agents)):
    #     fig_handle = plt.figure()
    #     plt.plot(np.linspace(0, T, T), mean_exp_roi[a], 'b-', label=f'{agents[a]} expected roi') 
    #     plt.plot(np.linspace(0, T, T), mean_real_roi[a],'m-',  label=f'{agents[a]} real roi') 
    #     plt.plot(np.linspace(0, T, T), clairvoyant_roi, 'r:', label=r'clairvoyant roi')
    #     if MIN_ROI > 0.0:
    #         plt.plot(np.linspace(0, T, T), np.full((T), MIN_ROI), 'g-.', label='roi target')  # clairvoyant_roi, 'r:', label=r'claivoyant roi')
    #     plt.fill_between(np.linspace(0, T, T), mean_exp_roi[a] + sigma_exp_roi[a] * 1.96, mean_exp_roi[a] - sigma_exp_roi[a] * 1.96, alpha=0.2)
    #     plt.fill_between(np.linspace(0, T, T), mean_real_roi[a] + sigma_real_roi[a] * 1.96, mean_real_roi[a] - sigma_real_roi[a] * 1.96, alpha=0.2)
    #     for r in range(runs_num):
    #         plt.plot(np.linspace(0, T, T), total_exp_roi[a, :, r], "C0", linewidth=0.5)
    #         plt.plot(np.linspace(0, T, T), total_real_roi[a, :, r],"hotpink", linewidth=0.5)
    #     plt.gca().legend()
    #     x_label = 'time'
    #     y_label = 'ROI'
    #     plt.xlabel(x_label)
    #     plt.ylabel(y_label)
    #     bottom, top = plt.ylim()
    #     plt.ylim(min_roi, max_roi)
    #     if Show:
    #         plt.show()
    #     if  Save:
    #         name = f'cumulative_roi_{agents[a]}'
    #         plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
    #         pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
    #         tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
    #         plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
    #         pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
    #         tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    #     plt.close()

    for a in range(len(agents)):
        fig_handle = plt.figure()
        #plt.plot(np.linspace(0, T, T), mean_exp_roi[a], 'b-', label=f'{agents[a]} expected roi') 
        plt.plot(np.linspace(0, T, T), mean_real_roi[a],'m-',  label=f'{agents[a]} real roi') 
        #plt.plot(np.linspace(0, T, T), clairvoyant_roi, 'r:', label=r'clairvoyant roi')
        if MIN_ROI > 0.0:
            plt.plot(np.linspace(0, T, T), np.full((T), MIN_ROI), 'g-.', label='roi target')  # clairvoyant_roi, 'r:', label=r'claivoyant roi')
        #plt.fill_between(np.linspace(0, T, T), mean_exp_roi[a] + sigma_exp_roi[a] * 1.96, mean_exp_roi[a] - sigma_exp_roi[a] * 1.96, alpha=0.2)
        plt.fill_between(np.linspace(0, T, T), mean_real_roi[a] + sigma_real_roi[a] * 1.96, mean_real_roi[a] - sigma_real_roi[a] * 1.96, alpha=0.2)
        for r in range(runs_num):
            #plt.plot(np.linspace(0, T, T), total_exp_roi[a, :, r], "C0", linewidth=0.5)
            plt.plot(np.linspace(0, T, T), total_real_roi[a, :, r],"hotpink", linewidth=0.5)
        plt.gca().legend()
        x_label = 'time'
        y_label = 'ROI'
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        bottom, top = plt.ylim()
        plt.ylim(min_roi, max_real_roi)
        #plt.ylim(LOW_ROI, MAX_ROI)
        if Show:
            plt.show()
        if  Save:
            name = f'cumulative_realroi_{agents[a]}'
            plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
            pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
            tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
            plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
            pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
            tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

        plt.close()

    fig_handle = plt.figure()
    plt.plot(np.linspace(0, T, T), np.full((T), MIN_ROI), 'r:', label=r'Target ROI')
    for a in range(len(agents)):
        plt.plot(np.linspace(0, T, T), mean_real_roi[a], label=f'{agents[a]} ROI') 
    plt.gca().legend()
    x_label = 'time'
    y_label = 'ROI'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    bottom, top = plt.ylim()
    plt.ylim(min_roi, max_real_roi)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_cumulative_roi'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()

    fig_handle = plt.figure()
    plt.plot(np.linspace(0, T, T), np.full((T), MIN_ROI), 'r:', label=r'Target ROI', linewidth=2)
    for a in range(len(agents)):
        plt.plot(np.linspace(0, T, T), mean_real_roi[a]) #, label=f'{agents[a]} ROI') 
        plt.fill_between(np.linspace(0, T, T), mean_real_roi[a] + sigma_real_roi[a] * 1.96, mean_real_roi[a] - sigma_real_roi[a] * 1.96,
                                     label=f'{agents[a]} ROI', alpha=0.6)
    plt.gca().legend()
    x_label = 'time'
    y_label = 'ROI'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    bottom, top = plt.ylim()
    plt.ylim(min_roi, max_real_roi)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_cumulative_confidence_interval_roi'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()




    for a in range(len(agents)):
        fig_handle = plt.figure()
        #plt.plot(np.linspace(0, T, T), mean_exp_roi[a], 'b-', label=f'{agents[a]} expected roi') 
        plt.plot(np.linspace(0, T, T), roi_violation[a], label=f'{agents[a]} roi violation rate') 
        #plt.bar(np.linspace(0, T, T), roi_violation[a], label=f'{agents[a]} roi violation rate') 
        #plt.plot(np.linspace(0, T, T), clairvoyant_roi, 'r:', label=r'clairvoyant roi')
#         if MIN_ROI > 0.0:
#             plt.plot(np.linspace(0, T, T), np.full((T), MIN_ROI), 'g-.', label='roi target')  # clairvoyant_roi, 'r:', label=r'claivoyant roi')
        #plt.fill_between(np.linspace(0, T, T), mean_exp_roi[a] + sigma_exp_roi[a] * 1.96, mean_exp_roi[a] - sigma_exp_roi[a] * 1.96, alpha=0.2)
#         plt.fill_between(np.linspace(0, T, T), mean_real_roi[a] + sigma_real_roi[a] * 1.96, mean_real_roi[a] - sigma_real_roi[a] * 1.96, alpha=0.2)
#         for r in range(runs_num):
#             #plt.plot(np.linspace(0, T, T), total_exp_roi[a, :, r], "C0", linewidth=0.5)
#             plt.plot(np.linspace(0, T, T), total_real_roi[a, :, r],"hotpink", linewidth=0.5)
        plt.gca().legend()
        x_label = 'time'
        y_label = 'ROI violation rate'
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        bottom, top = plt.ylim()
        plt.ylim(0., 1.)
        #plt.ylim(LOW_ROI, MAX_ROI)
        if Show:
            plt.show()
        if  Save:
            name = f'cumulative_roi_violation_rate_{agents[a]}'
            plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
            pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
            tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
            plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
            pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
            tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

        plt.close()


    fig, axs = plt.subplots(3, sharex=True)
    import matplotlib.lines as mlines
    for a in range(len(agents)):
        fig, axs = plt.subplots(3, sharex=True)
        #fig_handle = plt.figure()
        fig.suptitle(f'{agents[a]} 90th, 50th and 10th percentiles')
        plus_line = mlines.Line2D([], [], color='k', marker="+", markersize=5, linestyle='None', label='90th percentile') 
        minus_line = mlines.Line2D([], [], color='k', marker="_", markersize=5, linestyle='None', label='10th percentile') 
        simple_line = mlines.Line2D([], [], color='k', label='50th percentile')
        target_line = mlines.Line2D([], [], label='Target', linestyle='dashdot', color='k', linewidth=2.5)
        optimum_line = mlines.Line2D([], [], label='Optimum', linestyle=':', color='k', linewidth=2.5)
        optimum_cost_line = mlines.Line2D([], [], color='r', linestyle=':', label=r'Optimum cost', linewidth=2.5)
        target_cost_line = mlines.Line2D([], [], label='Target cost', linestyle='dashdot', color='r', linewidth=2.5)
        optimum_rev_line = mlines.Line2D([], [], color='r', linestyle=':', label=r'Optimum revenue', linewidth=2.5)
        target_real_roi_line = mlines.Line2D([], [], label='Target ROI', linestyle='dashdot', color='g', linewidth=2.5)
        #plt.legend(handles=[plus_line, minus_line, simple_line, target_cost_line, target_real_roi_line])
        axs[0].legend(handles=[optimum_line], handlelength=3, loc='best', fontsize='small') #, target_real_roi_line])
        axs[1].legend(handles=[plus_line, minus_line, simple_line], loc='best', fontsize='x-small')
        axs[2].legend(handles=[target_line], handlelength=3, loc='best', fontsize='small')
        axs[0].plot(np.linspace(0,T, T), percentile90_cost[a], 'b+', label=f'{agents[a]} 90th percentile cost', linewidth='1')
        axs[0].plot(np.linspace(0,T, T), percentile50_cost[a], 'b', label=f'{agents[a]} 50th percentile cost', linewidth='1')
        axs[0].plot(np.linspace(0,T, T), percentile10_cost[a], 'b--', label=f'{agents[a]} 10th percentile cost', linewidth='1')
        axs[0].plot(np.linspace(0, T, T), optimum_cost, 'r:', label=r'Optimum cost', linewidth=2.5)
        axs[0].plot(np.linspace(0, T, T), target_cost, color='r', linestyle='dashdot', label=r'Target cost', linewidth=2.5)
        axs[0].set_ylim(min_cost, max_cost)
        axs[0].fill_between(np.linspace(0, T, T), percentile90_cost[a], percentile10_cost[a], color='b', alpha=0.2)
        axs[0].set(ylabel='cost')
        axs[1].plot(np.linspace(0,T, T), percentile90_rev[a], 'g+', label=f'{agents[a]} 90th percentile revenue', linewidth='1')
        axs[1].plot(np.linspace(0,T, T), percentile50_rev[a], 'g', label=f'{agents[a]} 50th percentile revenue', linewidth='1')
        axs[1].plot(np.linspace(0,T, T), percentile10_rev[a], 'g--', label=f'{agents[a]} 10th percentile revenue', linewidth='1')
        axs[1].fill_between(np.linspace(0, T, T), percentile90_rev[a], percentile10_rev[a], color='g', alpha=0.2)
        axs[1].plot(np.linspace(0, T, T), optimum_rev, 'r:', label=r'Optimum revenue', linewidth=2.5)
        #plt.plot(timeline, target_rev[INITIAL_BIDS-1:], 'r:', label=r'Optimum', linewidth=3)
        axs[1].set_ylim(min_rev, max_rev)
        axs[1].set(ylabel='revenue')
        axs[2].plot(np.linspace(0,T, T), percentile90_real_roi[a], 'r+', label=f'{agents[a]} 90th percentile roi', linewidth='1')
        axs[2].plot(np.linspace(0,T, T), percentile50_real_roi[a], 'r', label=f'{agents[a]} 50th percentile roi', linewidth='1')
        axs[2].plot(np.linspace(0,T, T), percentile10_real_roi[a], 'r--', label=f'{agents[a]} 10th percentile roi', linewidth='1')
        axs[2].fill_between(np.linspace(0, T, T), percentile90_real_roi[a], percentile10_real_roi[a], color='r', alpha=0.2)
        axs[2].plot(np.linspace(0, T, T), target_roi, 'g-.', label='roi target', linewidth=2.5)
        axs[2].plot(np.linspace(0, T, T), clairvoyant_roi, 'g:', label='roi optimum', linewidth=2.5)
        axs[2].set_ylim(min_real_roi, max_real_roi)
        axs[2].set(ylabel='ROI')
        if Show:
            plt.show()
        if  Save:
            name = f'cumulative_percentile_plots_{agents_to_file[a]}'
            plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
            pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
            tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
            plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
            pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
            tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

        plt.close()


    timeline = np.linspace(0, T - INITIAL_BIDS + 1, T - INITIAL_BIDS + 1)
    # no initial bids
    for a in range(len(agents)):
        fig, axs = plt.subplots(3, sharex=True)
        #fig_handle = plt.figure()
        fig.suptitle(f'{agents[a]} 90th, 50th and 10th percentiles')
        plus_line = mlines.Line2D([], [], color='k', marker="+", markersize=5, linestyle='None', label='90th percentile') 
        minus_line = mlines.Line2D([], [], color='k', marker="_", markersize=5, linestyle='None', label='10th percentile') 
        simple_line = mlines.Line2D([], [], color='k', label='50th percentile')
        target_line = mlines.Line2D([], [], label='Target', linestyle='dashdot', color='k', linewidth=2.5)
        optimum_line = mlines.Line2D([], [], label='Optimum', linestyle=':', color='k', linewidth=2.5)
        optimum_cost_line = mlines.Line2D([], [], color='r', linestyle=':', label=r'Optimum cost', linewidth=2.5)
        target_cost_line = mlines.Line2D([], [], label='Target cost', linestyle='dashdot', color='r', linewidth=2.5)
        optimum_rev_line = mlines.Line2D([], [], color='r', linestyle=':', label=r'Optimum revenue', linewidth=2.5)
        target_real_roi_line = mlines.Line2D([], [], label='Target ROI', linestyle='dashdot', color='g', linewidth=2.5)
        #plt.legend(handles=[plus_line, minus_line, simple_line, target_cost_line, target_real_roi_line])

        axs[0].legend(handles=[optimum_line], handlelength=3, loc='best', fontsize='small') #, target_real_roi_line])
        axs[1].legend(handles=[plus_line, minus_line, simple_line], loc='best', fontsize='x-small')
        axs[2].legend(handles=[target_line], handlelength=3, loc='best', fontsize='small')

        axs[0].plot(timeline, percentile90_cost[a, INITIAL_BIDS - 1:], 'b+', label=f'{agents[a]} 90th percentile cost', linewidth='1')
        axs[0].plot(timeline, percentile50_cost[a, INITIAL_BIDS - 1:], 'b', label=f'{agents[a]} 50th percentile cost', linewidth='1')
        axs[0].plot(timeline, percentile10_cost[a, INITIAL_BIDS - 1:], 'b--', label=f'{agents[a]} 10th percentile cost', linewidth='1')
        axs[0].plot(timeline, optimum_cost[INITIAL_BIDS - 1:], 'r:', label=r'Optimum cost', linewidth=2.5)
        axs[0].plot(timeline, target_cost[INITIAL_BIDS - 1:], color='r', linestyle='dashdot', label=r'Target cost', linewidth=2.5)
        axs[0].set_ylim(min_cost, max_cost)
        axs[0].fill_between(timeline, percentile90_cost[a, INITIAL_BIDS - 1:], percentile10_cost[a, INITIAL_BIDS - 1:], color='b', alpha=0.2)
        axs[0].set(ylabel='cost')
        axs[1].plot(timeline, percentile90_rev[a, INITIAL_BIDS - 1:], 'g+', label=f'{agents[a]} 90th percentile revenue', linewidth='1')
        axs[1].plot(timeline, percentile50_rev[a, INITIAL_BIDS - 1:], 'g', label=f'{agents[a]} 50th percentile revenue', linewidth='1')
        axs[1].plot(timeline, percentile10_rev[a, INITIAL_BIDS - 1:], 'g--', label=f'{agents[a]} 10th percentile revenue', linewidth='1')
        axs[1].fill_between(timeline, percentile90_rev[a, INITIAL_BIDS - 1:], percentile10_rev[a, INITIAL_BIDS - 1:], color='g', alpha=0.2)
        axs[1].plot(timeline, optimum_rev[INITIAL_BIDS - 1:], 'r:', label=r'Optimum revenue', linewidth=2.5)
        #plt.plot(timeline, target_rev[INITIAL_BIDS-1:], 'r:', label=r'Optimum', linewidth=3)
        axs[1].set_ylim(min_rev, max_rev)
        axs[1].set(ylabel='revenue')
        axs[2].plot(timeline, percentile90_real_roi[a, INITIAL_BIDS - 1:], 'r+', label=f'{agents[a]} 90th percentile roi', linewidth='1')
        axs[2].plot(timeline, percentile50_real_roi[a, INITIAL_BIDS - 1:], 'r', label=f'{agents[a]} 50th percentile roi', linewidth='1')
        axs[2].plot(timeline, percentile10_real_roi[a, INITIAL_BIDS - 1:], 'r--', label=f'{agents[a]} 10th percentile roi', linewidth='1')
        axs[2].fill_between(timeline, percentile90_real_roi[a, INITIAL_BIDS - 1:], percentile10_real_roi[a, INITIAL_BIDS - 1:], color='r', alpha=0.2)
        axs[2].plot(timeline, target_roi[INITIAL_BIDS - 1:], 'g-.', label='roi target', linewidth=2.5)
        axs[2].plot(timeline, clairvoyant_roi[INITIAL_BIDS - 1:], 'g:', label='roi optimum', linewidth=2.5)
        axs[2].set_ylim(min_real_roi, max_real_roi)
        axs[2].set(ylabel='ROI')
        if Show:
            plt.show()
        if  Save:
            name = f'cumulative_percentile_plots_no_first_bids{agents_to_file[a]}'
            plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
            pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
            tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
            plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
            pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
            tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

        plt.close()

    cmap = plt.get_cmap('jet_r')
    colors = ['g', 'b', 'c', 'm', 'y', 'k']

    plus_line = mlines.Line2D([], [], color='k', marker="+", markersize=5, linestyle='None', label='90th percentile') 
    minus_line = mlines.Line2D([], [], color='k', marker="_", markersize=5, linestyle='None', label='10th percentile') 
    simple_line = mlines.Line2D([], [], color='k', label='50th percentile')
    optimum_cost_line = mlines.Line2D([], [], color='r', linestyle=':', label=r'Optimum cost', linewidth=2.5)
    target_cost_line = mlines.Line2D([], [], label='Target cost', linestyle='dashdot', color='r', linewidth=2.5)
    optimum_rev_line = mlines.Line2D([], [], color='r', linestyle=':', label=r'Optimum revenue', linewidth=2.5)
    target_real_roi_line = mlines.Line2D([], [], label='Target ROI', linestyle='dashdot', color='g', linewidth=2.5)

    timeline = np.linspace(0, T - INITIAL_BIDS + 1, T - INITIAL_BIDS + 1)

    fig_handle = plt.figure()
    plt.title(f'Revenue 90th, 50th and 10th percentiles')
    legend_plots = []
    for a in range(len(agents)):
        #color = cmap(float(a)/len(agents))
        plt.plot(timeline, percentile90_rev[a, INITIAL_BIDS-1:], color=colors[a], linestyle='None', linewidth='1', marker='+')
        plt.plot(timeline, percentile50_rev[a, INITIAL_BIDS-1:], color=colors[a])
        plt.plot(timeline, percentile10_rev[a, INITIAL_BIDS-1:], color=colors[a], linestyle='--')
        legend_plots.append(plt.fill_between(timeline, percentile90_rev[a, INITIAL_BIDS-1:], percentile10_rev[a, INITIAL_BIDS-1:], color=colors[a], alpha=0.4, label=f'{agents[a]}'))
    plt.plot(timeline, optimum_rev[INITIAL_BIDS-1:], 'r:', label=r'Optimum', linewidth=3)
    x_label = 'time'
    y_label = 'revenue'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().legend()
    legend_plots.extend([plus_line, minus_line, simple_line])
    plt.gca().legend(handles=legend_plots, loc='best')
    bottom, top = plt.ylim()
    plt.ylim(min_rev, max_rev)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_revs'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()




    legend_plots = []
    fig_handle = plt.figure()
    plt.title(f'Cost 90th, 50th and 10th percentiles')
    for a in range(len(agents)):
        #color = cmap(float(a)/len(agents))
        plt.plot(timeline, percentile90_cost[a, INITIAL_BIDS-1:], color=colors[a], linestyle='None', linewidth='1', marker='+')
        plt.plot(timeline, percentile50_cost[a, INITIAL_BIDS-1:], color=colors[a])
        plt.plot(timeline, percentile10_cost[a, INITIAL_BIDS-1:], color=colors[a], linestyle='--')
        legend_plots.append(plt.fill_between(timeline, percentile90_cost[a, INITIAL_BIDS-1:], percentile10_cost[a, INITIAL_BIDS-1:], color=colors[a], alpha=0.4, label=f'{agents[a]} cost'))
    #plt.plot(timeline, target_cost[INITIAL_BIDS-1:], 'r:-', label=r'Target', linewidth=3)
    plt.plot(timeline, optimum_cost[INITIAL_BIDS-1:], 'r:', label=r'Optimum', linewidth=3)
    plt.plot(timeline, target_cost[INITIAL_BIDS-1:], color='r', linestyle='dashdot', label=r'Target cost', linewidth=2.5)
    x_label = 'time'
    y_label = 'cost'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    legend_plots.extend([plus_line, minus_line, simple_line])
    plt.gca().legend()
    bottom, top = plt.ylim()
    plt.ylim(min_cost, max_cost)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_costs'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()



    legend_plots = []
    fig_handle = plt.figure()
    plt.title(f'ROI 90th, 50th and 10th percentiles')
    for a in range(len(agents)):
        #color = cmap(float(a)/len(agents))
        plt.plot(timeline, percentile90_real_roi[a, INITIAL_BIDS-1:], color=colors[a], linestyle='None', linewidth='1', marker='+')
        plt.plot(timeline, percentile50_real_roi[a, INITIAL_BIDS-1:], color=colors[a])
        plt.plot(timeline, percentile10_real_roi[a, INITIAL_BIDS-1:], color=colors[a], linestyle='--')
        legend_plots.append(plt.fill_between(timeline, percentile90_real_roi[a, INITIAL_BIDS-1:], percentile10_real_roi[a, INITIAL_BIDS-1:], color=colors[a], alpha=0.4, label=f'{agents[a]} ROI'))
    plt.plot(timeline, clairvoyant_roi[INITIAL_BIDS-1:], 'r:', label=r'Optimum', linewidth=3)
    plt.plot(timeline, np.full((T - INITIAL_BIDS + 1), MIN_ROI), 'r-.', label='Target', linewidth=2.5)
    #    plt.plot(np.linspace(0, T, T), clairvoyant_roi, 'r:', label=r'clairvoyant roi')
    x_label = 'time'
    y_label = 'ROI'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    legend_plots.extend([plus_line, minus_line, simple_line])
    plt.gca().legend()
    bottom, top = plt.ylim()
    plt.ylim(min_real_roi, max_real_roi)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_roi'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()



    fig_handle = plt.figure()
    plt.title(f'Revenue 90th, 50th and 10th percentiles')
    for a in range(len(agents)):
        #color = cmap(float(a)/len(agents))
        plt.plot(timeline, percentile90_rev[a, INITIAL_BIDS-1:], color=colors[a], linestyle='None', linewidth='1', marker='+')
        plt.plot(timeline, percentile50_rev[a, INITIAL_BIDS-1:], color=colors[a])
        plt.plot(timeline, percentile10_rev[a, INITIAL_BIDS-1:], color=colors[a], linestyle='--')
        #plt.fill_between(timeline, percentile90_rev[a, INITIAL_BIDS-1:], percentile10_rev[a, INITIAL_BIDS-1:], color=colors[a], alpha=0.4, label=f'{agents[a]} revenue')
    plt.plot(timeline, optimum_rev[INITIAL_BIDS-1:], 'r:', label=r'Optimum', linewidth=3)
    x_label = 'time'
    y_label = 'revenue'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().legend()
    bottom, top = plt.ylim()
    plt.ylim(min_rev, max_rev)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_revs_no_area'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()




    fig_handle = plt.figure()
    plt.title(f'Cost 90th, 50th and 10th percentiles')
    for a in range(len(agents)):
        #color = cmap(float(a)/len(agents))
        plt.plot(timeline, percentile90_cost[a, INITIAL_BIDS-1:], color=colors[a], linestyle='None', linewidth='1', marker='+')
        plt.plot(timeline, percentile50_cost[a, INITIAL_BIDS-1:], color=colors[a])
        plt.plot(timeline, percentile10_cost[a, INITIAL_BIDS-1:], color=colors[a], linestyle='--')
        #plt.fill_between(timeline, percentile90_cost[a, INITIAL_BIDS-1:], percentile10_cost[a, INITIAL_BIDS-1:], color=colors[a], alpha=0.4, label=f'{agents[a]} cost')
    #plt.plot(timeline, target_cost[INITIAL_BIDS-1:], 'r:', label=r'Target', linewidth=3)
    plt.plot(timeline, target_cost[INITIAL_BIDS-1:], color='r', linestyle='dashdot', label=r'Target cost', linewidth=2.5)
    plt.plot(timeline, optimum_cost[INITIAL_BIDS-1:], 'r:', label=r'Optimum', linewidth=3)
    x_label = 'time'
    y_label = 'cost'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().legend()
    bottom, top = plt.ylim()
    plt.ylim(min_cost, max_cost)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_costs_no_area'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()



    fig_handle = plt.figure()
    plt.title(f'ROI 90th, 50th and 10th percentiles')
    for a in range(len(agents)):
        #color = cmap(float(a)/len(agents))
        plt.plot(timeline, percentile90_real_roi[a, INITIAL_BIDS-1:], color=colors[a], linestyle='None', linewidth='1', marker='+')
        plt.plot(timeline, percentile50_real_roi[a, INITIAL_BIDS-1:], color=colors[a])
        plt.plot(timeline, percentile10_real_roi[a, INITIAL_BIDS-1:], color=colors[a], linestyle='--')
        #plt.fill_between(timeline, percentile90_real_roi[a, INITIAL_BIDS-1:], percentile10_real_roi[a, INITIAL_BIDS-1:], color=colors[a], alpha=0.4, label=f'{agents[a]} ROI')
    plt.plot(timeline, clairvoyant_roi[INITIAL_BIDS-1:], 'r:', label=r'Optimum', linewidth=3)
    plt.plot(timeline, np.full((T - INITIAL_BIDS + 1), MIN_ROI), 'r-.', label='Target', linewidth=2.5)
    #    plt.plot(np.linspace(0, T, T), clairvoyant_roi, 'r:', label=r'clairvoyant roi')
    x_label = 'time'
    y_label = 'ROI'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().legend()
    bottom, top = plt.ylim()
    plt.ylim(min_real_roi, max_real_roi)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_roi_no_area'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()



    fig_handle = plt.figure()
    plt.title(f'Revenue 50th percentile')
    for a in range(len(agents)):
        #color = cmap(float(a)/len(agents))
        #plt.plot(timeline, percentile90_rev[a, INITIAL_BIDS-1:], color=colors[a], linestyle='None', linewidth='1', marker='+')
        plt.plot(timeline, percentile50_rev[a, INITIAL_BIDS-1:], color=colors[a], label=f'{agents[a]} revenue')
        #plt.plot(timeline, percentile10_rev[a, INITIAL_BIDS-1:], color=colors[a], linestyle='--')
        #plt.fill_between(timeline, percentile90_rev[a, INITIAL_BIDS-1:], percentile10_rev[a, INITIAL_BIDS-1:], color=colors[a], alpha=0.4, label=f'{agents[a]} revenue')
    plt.plot(timeline, optimum_rev[INITIAL_BIDS-1:], 'r:', label=r'Optimum', linewidth=3)
    x_label = 'time'
    y_label = 'revenue'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().legend()
    bottom, top = plt.ylim()
    plt.ylim(min_rev, max_rev)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_median_revs'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()




    fig_handle = plt.figure()
    plt.title(f'Cost 50th percentile')
    for a in range(len(agents)):
        #color = cmap(float(a)/len(agents))
        #plt.plot(timeline, percentile90_cost[a, INITIAL_BIDS-1:], color=colors[a], linestyle='None', linewidth='1', marker='+')
        plt.plot(timeline, percentile50_cost[a, INITIAL_BIDS-1:], color=colors[a], label=f'{agents[a]} cost')
        #plt.plot(timeline, percentile10_cost[a, INITIAL_BIDS-1:], color=colors[a], linestyle='--')
        #plt.fill_between(timeline, percentile90_cost[a, INITIAL_BIDS-1:], percentile10_cost[a, INITIAL_BIDS-1:], color=colors[a], alpha=0.4, label=f'{agents[a]} cost')
    #plt.plot(timeline, target_cost[INITIAL_BIDS-1:], 'r:', label=r'Target', linewidth=3)
    plt.plot(timeline, optimum_cost[INITIAL_BIDS-1:], 'r:', label=r'Optimum', linewidth=3)
    plt.plot(timeline, target_cost[INITIAL_BIDS-1:], color='r', linestyle='dashdot', label=r'Target cost', linewidth=2.5)
    x_label = 'time'
    y_label = 'cost'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().legend()
    bottom, top = plt.ylim()
    plt.ylim(min_cost, max_cost)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_median_costs'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()



    fig_handle = plt.figure()
    plt.title(f'ROI 50th percentile')
    for a in range(len(agents)):
        #color = cmap(float(a)/len(agents))
        #plt.plot(timeline, percentile90_real_roi[a, INITIAL_BIDS-1:], color=colors[a], linestyle='None', linewidth='1', marker='+')
        plt.plot(timeline, percentile50_real_roi[a, INITIAL_BIDS-1:], color=colors[a], label=f'{agents[a]}')
        #plt.plot(timeline, percentile10_real_roi[a, INITIAL_BIDS-1:], color=colors[a], linestyle='--')
        #plt.fill_between(timeline, percentile90_real_roi[a, INITIAL_BIDS-1:], percentile10_real_roi[a, INITIAL_BIDS-1:], color=colors[a], alpha=0.4, label=f'{agents[a]} ROI')
    plt.plot(timeline, clairvoyant_roi[INITIAL_BIDS-1:], 'r:', label=r'Optimum', linewidth=3)
    plt.plot(timeline, np.full((T - INITIAL_BIDS + 1), MIN_ROI), 'r-.', label='Target', linewidth=2.5)
    #    plt.plot(np.linspace(0, T, T), clairvoyant_roi, 'r:', label=r'clairvoyant roi')
    x_label = 'time'
    y_label = 'ROI'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().legend()
    bottom, top = plt.ylim()
    plt.ylim(min_real_roi, max_real_roi)
    if Show:
        plt.show()
    if  Save:
        name = f'confront_median_roi'
        plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
        pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
        tikz_path = f'{path+"/cumulative_tikz" if path != None else "cumulative_tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    plt.close()
        # plt.plot(np.linspace(0, T, T), mean_cost[a], 'b-', label=f'{agents[a]} cost') 
        # plt.plot(np.linspace(0, T, T), target_cost, 'r:', label=r'Target cost')
        # plt.gca().legend()
        # plt.fill_between(np.linspace(0, T, T), mean_cost[a] + sigma_cost[a] * 1.96, mean_cost[a] - sigma_cost[a] * 1.96, alpha=0.2)
        # for r in range(runs_num):
        #     plt.plot(np.linspace(0, T, T), total_cost[a, :, r], "C0", linewidth=0.5)
        # plt.show()

    # bid plots
    # #for t in range(T):
    # bid_array = np.linspace(0., MAX_BID, N_BID)
    # for i in range(len(agents)):
    #     for j in range(len(subcampaigns)):
    #         obs = [np.count_nonzero(X[i][j][:] == v) for v in bid_array]
    #         #labels = [str(bid) if bid % 0.5 == 0 else '' for bid in bid_array]
    #         fig_handle = plt.figure()
    #         plt.bar(bid_array, obs, width=0.005, linewidth=0, label=f'{agents[i]} subcampaign {j} chosen bids')
    #         plt.gca().legend()
    #         #plt.xticks(bid_array, bid_array)
    #         if Show:
    #             plt.show()
    #         if  Save:
    #             name = f'{agents_to_file[i]}_sc{j}_bid_choices'
    #             plots_path = f'{path+"/cumulative_plots" if path != None else "cumulative_plots"}'
    #             pickle_path = f'{path+"/cumulative_pickle" if path != None else "cumulative_pickle"}'
    #             plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
    #             pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))

    #         plt.close()

                #obs_cost = Y_cost[i, j, :t+1]



def load_run_settings(path):
    with open(f'{path}/settings.pickle', 'rb') as f:
        runs_num = pl.load(f)
        # print(runs_num)
        T = pl.load(f)
        subcampaigns = pl.load(f)
        agents = pl.load(f)
    return runs_num, T, subcampaigns, agents

def load_run_data(path):
    with open(f'{path}/run_data.pickle', 'rb') as f:
        T = pl.load(f)
        subc = pl.load(f)
        agents = pl.load(f)
        target_rev = pl.load(f)
        target_roi = pl.load(f)
        cost_array = pl.load(f)
        bid_array = pl.load(f)
        X = pl.load(f)
        Y_cost = pl.load(f)
        Y_rev = pl.load(f)
        f_rev = pl.load(f)
        f_cost = pl.load(f)
        r_data = pl.load(f)
        roi_data = pl.load(f)
        exp_roi = pl.load(f)
        exp_cost = pl.load(f)
        exp_rev = pl.load(f)
        real_sc_costs = pl.load(f)
        real_sc_revs = pl.load(f)
        mean_gps_cost = pl.load(f)
        mean_gps_rev = pl.load(f)
        var_gps_cost = pl.load(f)
        var_gps_rev = pl.load(f)
        MIN_ROI = pl.load(f)

    # generate_gp_plots(f'{path}', T, subc, agents, bid_array, X, Y_cost, Y_rev,
    #                   mean_gps_cost, mean_gps_rev, var_gps_cost, var_gps_rev,
    #                   real_sc_costs, real_sc_revs)

    # generate_rev_cost_plots(f'{path}', T, subc, agents, cost_array,
    #                         target_rev,
    #                         r_data, exp_cost, exp_rev)

    # generate_roi_plots(f'{path}', T, MIN_ROI, subc, agents, cost_array,
    #                         target_rev, target_roi,
    #                         r_data, roi_data, exp_roi, exp_cost, exp_rev)

    return X, Y_cost, Y_rev, f_cost, f_rev, exp_roi, exp_cost, exp_rev, path


def generate_roi_plots(path, T, MIN_ROI, subc, agents, cost_array,
                            target_rev, target_roi,
                            r_data, roi_data, exp_roi, exp_cost, exp_rev):
    
    agents_to_file = [agents[i].replace(" ", "_").replace("%", "_") for i in range(len(agents))]

    max_rev = np.nanmax(r_data) if np.nanmax(r_data) > np.nanmax(target_rev) else np.nanmax(target_rev)
    min_rev = np.nanmin(r_data) if np.nanmin(r_data) < np.nanmin(target_rev) else np.nanmin(target_rev)
    max_rev += max_rev/100

    max_roi = np.nanmax(roi_data) if np.nanmax(roi_data) > np.nanmax(target_roi) else np.nanmax(target_roi)
    min_roi = np.nanmin(roi_data) if np.nanmin(roi_data) < np.nanmin(target_roi) else np.nanmin(target_roi)
    max_roi += max_roi/100

    for t in range(T):
        for i in range(len(agents)):
            plot.confront_plot(x=r_data[i, t, ~np.isnan(roi_data[i, t])],
                               x_target=target_rev[~np.isnan(target_rev)], 
                               y=roi_data[i, t, ~np.isnan(roi_data[i, t])],
                               target=target_roi[~np.isnan(target_rev)],
                               y_lim=(min_roi, max_roi), x_lim=(min_rev, max_rev),
                               x_label='revenue', y_label='roi', label=f'estimated ROI', # {agents[i]} day {t+1}',

                               title=f'{agents[i]} day {t+1}',
                               name=f'{agents_to_file[i]}_expected_roi_day_{t+1:02}',
                               target_label='true ROI',
                               Show=False, path=f'{path}/roi')

            plot.confront_plot(x=r_data[i, t, ~np.isnan(roi_data[i, t])],
                               x_target=np.linspace(min_rev, max_rev, 10),
                               y=roi_data[i, t, ~np.isnan(roi_data[i, t])],
                               #target=np.full((r_data[i, t, ~np.isnan(roi_data[i, t])].shape[0]), MIN_ROI),
                               target=np.full((10), MIN_ROI),
                               x_lim=(min_rev, max_rev), y_lim=(min_roi, max_roi),
                               x_label='revenue', y_label='roi', label=f'estimated ROI',  # {agents[i]} day {t+1}',

                               title=f'{agents[i]} day {t+1}',
                               name=f'{agents_to_file[i]}_expected_vs_target_roi_day_{t+1:02}',
                               target_label='ROI target',
                               Show=False, path=f'{path}/roi')

            #     plot.confront_plot(x=r[~np.isnan(roi)], x_target=target_rev[idx:], y=roi[~np.isnan(roi)], target=target_roi[idx:],
            #                        x_label='revenue', y_label='roi', label=f'estimated ROI {a.__str__()} day {t+1}',
            #                        name=f'{type(a).__name__}_expected_roi_day_{t+1:02}',
            #                        target_label='true ROI',
            #                        Show=False, path=f'{path}/roi/roinotnan')

            #     plot.confront_plot(x=r[~np.isnan(roi)], y=roi[idx_roi:], target=np.full((r[~np.isnan(roi)].shape[0]), MIN_ROI),
            #                        x_label='revenue', y_label='roi', label=f'estimated ROI {a.__str__()} day {t+1}',
            #                        name=f'{type(a).__name__}_expected_vs_min_roi_day_{t+1:02}',
            #                        target_label='ROI target',
            #                        Show=False, path=f'{path}/roi/roinotnan')

def generate_rev_cost_plots(path, T, subc, agents, cost_array,
                            target_rev, r_data, exp_cost, exp_rev):

    agents_to_file = [agents[i].replace(" ", "_").replace("%", "_") for i in range(len(agents))]

    max_rev = np.nanmax(r_data) if np.nanmax(r_data) > np.nanmax(target_rev) else np.nanmax(target_rev)
    max_rev += max_rev/100
    y_lim = 0, max_rev

    for t in range(T):
        for i in range(len(agents)):
            plot.confront_plot(x=cost_array, y=r_data[i, t], target=target_rev,
                               x_label='cost', y_label='revenue', y_lim=y_lim,
                               target_label='true revenue', label=f'estimated revenue',
                               title=f'{agents[i]} day {t+1}',
                               name=f'{agents_to_file[i]}_expected_revenue_day_{t+1:02}',
                               Show=False, path=f'{path}/rev')
            plot.confront_plot(x=cost_array, y=np.where(np.isnan(r_data[i, t]), 0., r_data[i, t]), y_lim=y_lim,
                               target=target_rev, x_label='cost', target_label='true revenue', y_label='revenue',
                               label=f'estimated revenue',  # {agents[i]} day {t+1}',
                               title=f'{agents[i]} day {t+1}',
                               name=f'{agents_to_file[i]}_expected_revenue_day_{t+1:02}',
                               Show=False, path=f'{path}/rev/revnantozero')


def generate_gp_plots(path, T, subc, agents, bid_array, X, Y_cost, Y_rev, mean_gps_cost, mean_gps_rev, var_gps_cost, var_gps_rev, real_sc_costs, real_sc_revs):

    agents_to_file = [agents[i].replace(" ", "_").replace("%", "_") for i in range(len(agents))]
    max_y_rev = [np.nanmax(real_sc_revs[j])*1.1 for j in range(subc)]
    min_y_rev = [np.nanmin(real_sc_revs[j])*1.1 for j in range(subc)]
    #min_y_rev = [-0.5 for j in range(subc)]
    max_y_cost = [np.nanmax(real_sc_costs[j])*1.1 for j in range(subc)]
    min_y_cost = [np.nanmin(real_sc_costs[j])*1.1 for j in range(subc)]
    #min_y_cost = [-0.5 for j in range(subc)]

    for t in range(T):
        for i in range(len(agents)):
            for j in range(subc):
                obs = X[i][j][:t+1]
                obs_cost = Y_cost[i, j, :t+1]
                obs_rev = Y_rev[i, j, :t+1]

                plot.plot_gp(x=bid_array, y=mean_gps_cost[i, j, t], var_y=var_gps_cost[i, j, t],
                             true_y=real_sc_costs[j],
                             obs=obs, obs_y=obs_cost,
                             y_label='cost', y_lim=(min_y_cost[j], max_y_cost[j]),
                             title=f'{agents[i]} subcampaign {j} cost model day {t}',
                             name=f'{agents_to_file[i]}_sc{j}_cost_day_{t:03}', path=f'{path}/gp')
                plot.plot_gp(x=bid_array, y=mean_gps_rev[i, j, t], var_y=var_gps_rev[i, j, t],
                             true_y=real_sc_revs[j],
                             obs=obs, obs_y=obs_rev,
                             y_label='rev', y_lim=(min_y_rev[j], max_y_rev[j]),

                             title=f'{agents[i]} subcampaign {j} revenue model day {t}',
                             name=f'{agents_to_file[i]}_sc{j}_rev_day{t:03}', path=f'{path}/gp')

#roi plots
#rev/cost plots


def load_run_config(path):
    with open(f'{path}/config.pickle', 'rb') as f:
        MAX_BID = pl.load(f)
        print('max_bid', MAX_BID)
        MAX_EXP = pl.load(f)
        print('max_exp', MAX_EXP)
        N_BID = pl.load(f)
        print('N_BID', N_BID)
        N_COST = pl.load(f)
        print('N_COST', N_COST)
        SAMPLES_INVERSE = pl.load(f)
        print('SAMPLES_INVERSE', SAMPLES_INVERSE)
        FIRST_BID = pl.load(f)
        print('FIRST_BID', FIRST_BID)
        MIN_ROI = pl.load(f)
        print('MIN_ROI', MIN_ROI)
        MAX_ROI = pl.load(f)
        print('MAX_ROI', MAX_ROI)
        LOW_ROI = pl.load(f)
        print('LOW_ROI', LOW_ROI)
        N_ROI = pl.load(f)
        print('N_ROI', N_ROI)
        return

def load_run_results(path):
    import glob, os
    os.chdir(f'{path}')
    results = []
    runs_num = 0
    for file in glob.glob("results*.pickle"):
        #with open(f'{path}/results.pickle', 'rb') as f:
        with open(file, 'rb') as f:
            runs_num += pl.load(f)
            # print(runs_num)
            T = pl.load(f)
            subcampaigns = pl.load(f)
            agents = pl.load(f)
            print(agents)
            results.extend(pl.load(f))
            #print(results)


    print(results)
    # with open(f'{path}/results.pickle', 'rb') as f:
    #     runs_num = pl.load(f)
    #     # print(runs_num)
    #     T = pl.load(f)
    #     subcampaigns = pl.load(f)
    #     agents = pl.load(f)
    #     print(agents)
    #     results = pl.load(f)
    #    # print(results)
    pathlib.Path(f'{path}/cumulative_plots').mkdir(parents=True, exist_ok=False)
    pathlib.Path(f'{path}/cumulative_pickle').mkdir(parents=True, exist_ok=False)
    pathlib.Path(f'{path}/cumulative_tikz').mkdir(parents=True, exist_ok=False)
    settings = (subcampaigns, agents, runs_num, T)
    analize_runs(settings, results, path)


def fix_hps():
    # # campaign_f = [sc.Subcampaign(160, 0.65, 847, .45, snr=25),
    # #               sc.Subcampaign(170, .62, 895, 0.42, snr=25),
    # #               sc.Subcampaign(170, 0.69, 886, 0.49, snr=25)] #,

    # hps = []

    # hp0 = { 'cost_variance': 0.5563902088650581,
    #         'rev_variance': 0.8514694098978604,
    #         'cost_lengthscales': 0.551249229940729,
    #         'rev_lengthscales': 0.4070104205028163,
    #         'cost_likelihood': 0.0011351938864693956,
    #         'rev_likelihood': 0.0018841000928501192
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.7322489921262527,
    #         'rev_variance': 1.0295037258245967,
    #         'cost_lengthscales': 0.5448653949910967,
    #         'rev_lengthscales': 0.4461761766108243,
    #         'cost_likelihood': 0.001361229118821822,
    #         'rev_likelihood': 0.0022267850618664644
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.6664919853811859,
    #         'rev_variance': 0.8646995271354193,
    #         'cost_lengthscales': 0.6560802960947089,
    #         'rev_lengthscales': 0.44331041687157874,
    #         'cost_likelihood': 0.0012161782494236786,
    #         'rev_likelihood': 0.0019786095711969718
    #       }
    # hps.append(hp2)

    # # campaign_f = [sc.Subcampaign(160, 0.65, 847, .45, snr=50),
    # #               sc.Subcampaign(170, .62, 895, 0.42, snr=50),
    # #               sc.Subcampaign(170, 0.69, 886, 0.49, snr=50)] #,

    # hps = []

    # hp0 = { 'cost_variance':0.7925765016053457, 
    #         'rev_variance': 1.4654051788009033,
    #         'cost_lengthscales': 0.5103564983693624,
    #         'rev_lengthscales': 0.421827959625221,
    #         'cost_likelihood': 4.260073712892668e-05,
    #         'rev_likelihood': 4.442015088098358e-05
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 1.007474929262363,
    #         'rev_variance': 1.9100687566823442,
    #         'cost_lengthscales': 0.5182096925410925,
    #         'rev_lengthscales': 0.41489719175838635,
    #         'cost_likelihood': 4.298110079247549e-05,
    #         'rev_likelihood': 4.4870859346413537e-05
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.9781469951794194,
    #         'rev_variance': 1.7559644893478692,
    #         'cost_lengthscales': 0.5645334294872469,
    #         'rev_lengthscales': 0.4675688864426567,
    #         'cost_likelihood': 4.282071585626794e-05,
    #         'rev_likelihood': 4.4970854700918863e-05
    #       }
    # hps.append(hp2)

    # # scale 100 cost 600 rev
    # # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=35),
    # #               sc.Subcampaign(77, .62, 565, 0.48, snr=35),
    # #               sc.Subcampaign(75, .67, 573, 0.43, snr=35),
    # #               sc.Subcampaign(65, .68, 503, 0.47, snr=35),
    # #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=35)] #,

    # hps = []

    # hp0 = { 'cost_variance': 0.7010173319736385, 
    #         'rev_variance': 0.7174035250696708,
    #         'cost_lengthscales': 0.6437911518276911,
    #         'rev_lengthscales': 0.3442673083018199,
    #         'cost_likelihood': 8.932541234431545e-05,
    #         'rev_likelihood': 0.00016906274136484347
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.9520458436769585,
    #         'rev_variance': 1.1337440878482705,
    #         'cost_lengthscales': 0.5715282156422046,
    #         'rev_lengthscales': 0.4606781392390854,
    #         'cost_likelihood': 0.00012311291566450078,
    #         'rev_likelihood': 0.00020646067635341446
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.5204703619590815,
    #         'rev_variance': 1.2855496262434014,
    #         'cost_lengthscales': 0.47286465102190617,
    #         'rev_lengthscales': 0.3705667489306539,
    #         'cost_likelihood': 0.00012326819059616857,
    #         'rev_likelihood': 0.00023032844191228516
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.7142816850390007,
    #         'rev_variance': 0.9307958396327624,
    #         'cost_lengthscales': 0.6342441567296747,
    #         'rev_lengthscales': 0.36351695492278674,
    #         'cost_likelihood': 9.769006477556555e-05,
    #         'rev_likelihood': 0.0001723178102458892
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.632919712651113,
    #         'rev_variance': 1.264528189325425,
    #         'cost_lengthscales': 0.6273570219713037,
    #         'rev_lengthscales': 0.39489475060783674,
    #         'cost_likelihood': 0.00011124573851550638,
    #         'rev_likelihood': 0.00020126612675532124
    #       }
    # hps.append(hp4)

    # # scale 100 cost 600 rev
    # # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=40),
    # #               sc.Subcampaign(77, .62, 565, 0.48, snr=40),
    # #               sc.Subcampaign(75, .67, 573, 0.43, snr=40),
    # #               sc.Subcampaign(65, .68, 503, 0.47, snr=40),
    # #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=40)] #,

    # hps = []

    # hp0 = { 'cost_variance': 0.6379943245040469, 
    #         'rev_variance': 0.9856639368837411,
    #         'cost_lengthscales': 0.5995287161932094,
    #         'rev_lengthscales': 0.36590040069781987,
    #         'cost_likelihood': 5.463032480135929e-05,
    #         'rev_likelihood': 7.888449014937199e-05
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.7668155209238494,
    #         'rev_variance': 0.7741405540550268,
    #         'cost_lengthscales': 0.49721440306287956,
    #         'rev_lengthscales': 0.36306026509333245,
    #         'cost_likelihood': 6.561483627345668e-05,
    #         'rev_likelihood': 8.910198971016156e-05
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.8773085510264393,
    #         'rev_variance': 1.8196964248474783,
    #         'cost_lengthscales': 0.588331287186139,
    #         'rev_lengthscales': 0.45724158804583226,
    #         'cost_likelihood': 6.575870009292755e-05,
    #         'rev_likelihood': 9.525930268936762e-05
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.6780253693457337,
    #         'rev_variance': 1.218267896319748,
    #         'cost_lengthscales': 0.6171449112604721,
    #         'rev_lengthscales': 0.4506324536402907,
    #         'cost_likelihood': 5.7033434046274794e-05,
    #         'rev_likelihood': 7.995451298064032e-05
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.6871380262780308,
    #         'rev_variance': 1.6358167643946602,
    #         'cost_lengthscales': 0.5892668368781879,
    #         'rev_lengthscales': 0.44477407255977325,
    #         'cost_likelihood': 6.019374277691669e-05,
    #         'rev_likelihood': 8.754827443191217e-05
    #       }
    # hps.append(hp4)

    # # scale 200 cost 800 rev
    # # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=40),
    # #               sc.Subcampaign(77, .62, 565, 0.48, snr=40),
    # #               sc.Subcampaign(75, .67, 573, 0.43, snr=40),
    # #               sc.Subcampaign(65, .68, 503, 0.47, snr=40),
    # #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=40)] #,

    # hps = []

    # hp0 = { 'cost_variance': 0.37722838753220106, 
    #         'rev_variance': 0.7527928750782581,
    #         'cost_lengthscales': 0.7167457963786387,
    #         'rev_lengthscales': 0.43203874159862954,
    #         'cost_likelihood': 4.37567092517135e-05,
    #         'rev_likelihood': 5.843584253945462e-05
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.396109112787972,
    #         'rev_variance': 0.8290043362964498,
    #         'cost_lengthscales': 0.608209009132937,
    #         'rev_lengthscales': 0.4816434033218942,
    #         'cost_likelihood': 4.6362066025700906e-05,
    #         'rev_likelihood': 6.121497412895392e-05
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.4213082828376098,
    #         'rev_variance': 1.0142945239966552,
    #         'cost_lengthscales': 0.7029557178358959,
    #         'rev_lengthscales': 0.4786502746423139,
    #         'cost_likelihood': 4.548771494015358e-05,
    #         'rev_likelihood': 6.303574034511346e-05
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.33495947234387896,
    #         'rev_variance': 0.7290359908979,
    #         'cost_lengthscales': 0.6617566562181175,
    #         'rev_lengthscales': 0.469567006679756,
    #         'cost_likelihood': 4.416758511532566e-05,
    #         'rev_likelihood': 5.659076437391122e-05
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.3468217272831681,
    #         'rev_variance': 0.8533165595885502,
    #         'cost_lengthscales': 0.648648074743574,
    #         'rev_lengthscales': 0.43026147218165856,
    #         'cost_likelihood': 4.4972325256546986e-05,
    #         'rev_likelihood': 6.055560915070317e-05
    #       }
    # hps.append(hp4)

    # # scale 100 cost 600 rev
    # # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=45),
    # #               sc.Subcampaign(77, .62, 565, 0.48, snr=45),
    # #               sc.Subcampaign(75, .67, 573, 0.43, snr=45),
    # #               sc.Subcampaign(65, .68, 503, 0.47, snr=45),
    # #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=45)] #,

    # hps = []

    # hp0 = { 'cost_variance': 0.709426028238513, 
    #         'rev_variance': 1.0981282388830376,
    #         'cost_lengthscales': 0.6068367654142249,
    #         'rev_lengthscales': 0.37633925897662446,
    #         'cost_likelihood': 4.476547079336845e-05,
    #         'rev_likelihood': 5.121766115027427e-05
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.8446545021316847,
    #         'rev_variance': 1.5197453594934065,
    #         'cost_lengthscales': 0.5281859349604773,
    #         'rev_lengthscales': 0.4400793283704084,
    #         'cost_likelihood': 4.8675389301862554e-05,
    #         'rev_likelihood': 5.472047674121923e-05
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.7018163373667905,
    #         'rev_variance': 1.3740367524749737,
    #         'cost_lengthscales': 0.517482315399603,
    #         'rev_lengthscales': 0.3852897284119158,
    #         'cost_likelihood': 4.7333786163598314e-05,
    #         'rev_likelihood': 5.5438741698896914e-05
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.7296017732651253,
    #         'rev_variance': 1.2083041743767498,
    #         'cost_lengthscales': 0.6067947396008786,
    #         'rev_lengthscales': 0.4287950866603174,
    #         'cost_likelihood': 4.558909634501272e-05,
    #         'rev_likelihood': 5.175376300691936e-05
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.7249195286152986,
    #         'rev_variance': 1.5409845448915394,
    #         'cost_lengthscales': 0.5815815554324073,
    #         'rev_lengthscales': 0.3979555275142917,
    #         'cost_likelihood': 4.526964566354467e-05,
    #         'rev_likelihood': 5.441567172775077e-05
    #       }
    # hps.append(hp4)

    # # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=25),
    # #               sc.Subcampaign(77, .62, 565, 0.48, snr=25),
    # #               sc.Subcampaign(75, .67, 573, 0.43, snr=25),
    # #               sc.Subcampaign(65, .68, 503, 0.47, snr=25),
    # #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=25)] #,

    # hps = []

    # hp0 = { 'cost_variance': 0.49695568072636437, 
    #         'rev_variance': 0.7219556762818629,
    #         'cost_lengthscales': 0.6350981200107161,
    #         'rev_lengthscales': 0.3858728477512838,
    #         'cost_likelihood': 0.0005970721114897112,
    #         'rev_likelihood': 0.0014857313040850895
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.5056804266315851,
    #         'rev_variance': 0.6956450755960444,
    #         'cost_lengthscales': 0.4465655059213654,
    #         'rev_lengthscales': 0.4403604016900734,
    #         'cost_likelihood': 0.0010392050255264044,
    #         'rev_likelihood': 0.0016717492006001144
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.7293849179032256,
    #         'rev_variance': 0.8472584068174056,
    #         'cost_lengthscales': 0.68239325370927,
    #         'rev_lengthscales': 0.43291871506336815,
    #         'cost_likelihood': 0.0009298616624518518,
    #         'rev_likelihood': 0.0019353624516116675
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.363311319307842,
    #         'rev_variance': 0.5915162506492595,
    #         'cost_lengthscales': 0.4731675019023338,
    #         'rev_lengthscales': 0.38610021159815927,
    #         'cost_likelihood': 0.0007263785930016948,
    #         'rev_likelihood': 0.0014914811122530712
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.4536708217739793,
    #         'rev_variance': 0.6807253708245534,
    #         'cost_lengthscales': 0.5927783502229618,
    #         'rev_lengthscales': 0.36308745189957703,
    #         'cost_likelihood': 0.0007789222542632095,
    #         'rev_likelihood': 0.0017730439766957353
    #       }
    # hps.append(hp4)

    # # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=28),
    # #               sc.Subcampaign(77, .62, 565, 0.48, snr=28),
    # #               sc.Subcampaign(75, .67, 573, 0.43, snr=28),
    # #               sc.Subcampaign(65, .68, 503, 0.47, snr=28),
    # #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=28)] #,

    # hps = []

    # hp0 = { 'cost_variance': 0.5090271256147253, 
    #         'rev_variance': 0.7860349194939322,
    #         'cost_lengthscales': 0.6285476847577661,
    #         'rev_lengthscales': 0.38398045852322094,
    #         'cost_likelihood': 0.00034002765010545934,
    #         'rev_likelihood': 0.000735668629382191
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.6844360313469388,
    #         'rev_variance': 0.8700019510184126,
    #         'cost_lengthscales': 0.5749362718187291,
    #         'rev_lengthscales': 0.4628551203116298,
    #         'cost_likelihood': 0.0005888114027581906,
    #         'rev_likelihood': 0.0009043725622319846
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.7934373512365999,
    #         'rev_variance': 1.0291211325882772,
    #         'cost_lengthscales': 0.6930822650651861,
    #         'rev_lengthscales': 0.43480808032358753,
    #         'cost_likelihood': 0.0004923489119140154,
    #         'rev_likelihood': 0.0010387763826549413
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.6964695441614799,
    #         'rev_variance': 0.4997106000326671,
    #         'cost_lengthscales': 0.627029483280445,
    #         'rev_lengthscales': 0.37065477144563364,
    #         'cost_likelihood': 0.0003794167003178204,
    #         'rev_likelihood': 0.0007690795311010852
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.6402504497438354,
    #         'rev_variance': 1.1798261650249842,
    #         'cost_lengthscales': 0.6076532592718005,
    #         'rev_lengthscales': 0.4315780696969566,
    #         'cost_likelihood': 0.00042347864206433445,
    #         'rev_likelihood': 0.0008918202117746999
    #       }
    # hps.append(hp4)

    # # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=29),
    # #               sc.Subcampaign(77, .62, 565, 0.48, snr=29),
    # #               sc.Subcampaign(75, .67, 573, 0.43, snr=29),
    # #               sc.Subcampaign(65, .68, 503, 0.47, snr=29),
    # #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=29)] #,

    # hps = []

    # hp0 = { 'cost_variance': 0.5151229962471036, 
    #         'rev_variance': 0.4476596682116408,
    #         'cost_lengthscales': 0.5600454734649167,
    #         'rev_lengthscales': 0.3303723518030036,
    #         'cost_likelihood': 0.00026451629869795663,
    #         'rev_likelihood': 0.0006069486939866592
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.513679698006691,
    #         'rev_variance': 1.097790082696714,
    #         'cost_lengthscales': 0.4606194077246793,
    #         'rev_lengthscales': 0.45056041348235293,
    #         'cost_likelihood': 0.00043763427141502956,
    #         'rev_likelihood': 0.0007197483479158731
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.5794973761993473,
    #         'rev_variance': 0.46926752234262153,
    #         'cost_lengthscales': 0.5380521571714192,
    #         'rev_lengthscales': 0.2796064957300898,
    #         'cost_likelihood': 0.00039764740735991896,
    #         'rev_likelihood': 0.0007192244876474071
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.38532475545791983,
    #         'rev_variance': 0.8334710314807499,
    #         'cost_lengthscales': 0.5292157659939537,
    #         'rev_lengthscales': 0.4413155409038679,
    #         'cost_likelihood': 0.0002979105789743087,
    #         'rev_likelihood': 0.0005603241023627523
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.6496859657094327,
    #         'rev_variance': 0.8015173009641171,
    #         'cost_lengthscales': 0.5780994841997271,
    #         'rev_lengthscales': 0.4070662776445191,
    #         'cost_likelihood': 0.00033829732189724783,
    #         'rev_likelihood': 0.0006971559268296111
    #       }
    # hps.append(hp4)

    # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=30),
    #               sc.Subcampaign(77, .62, 565, 0.48, snr=30),
    #               sc.Subcampaign(75, .67, 573, 0.43, snr=30),
    #               sc.Subcampaign(65, .68, 503, 0.47, snr=30),
    # #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=30)] #,

    # hps = []

    # hp0 = { 'cost_variance': 0.3115836862655195, 
    #         'rev_variance': 0.6690399001366217,
    #         'cost_lengthscales': 0.7606483956609061,
    #         'rev_lengthscales': 0.4253855501898366,
    #         'cost_likelihood': 7.662809576414123e-05,
    #         'rev_likelihood': 0.00022203163643009122
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.4437876153284067,
    #         'rev_variance': 0.49937513901777747,
    #         'cost_lengthscales': 0.7198938185983373,
    #         'rev_lengthscales': 0.46980050498427245,
    #         'cost_likelihood': 0.00010585567065420387,
    #         'rev_likelihood': 0.0002563713471115533
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.3163064304529755,
    #         'rev_variance': 0.7615342236129639,
    #         'cost_lengthscales': 0.5620140474254327,
    #         'rev_lengthscales': 0.4714089339452792,
    #         'cost_likelihood': 0.00010514284221900336,
    #         'rev_likelihood': 0.00027958826993770516
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.34906918122925495,
    #         'rev_variance': 0.6198262931215144,
    #         'cost_lengthscales': 0.7224852698271806,
    #         'rev_lengthscales': 0.48335139026847507,
    #         'cost_likelihood': 8.655123142871402e-05,
    #         'rev_likelihood': 0.00021409675768565393
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.41834375115875894,
    #         'rev_variance': 0.5829240045310492,
    #         'cost_lengthscales': 0.7272350262522077,
    #         'rev_lengthscales': 0.38643483773730897,
    #         'cost_likelihood': 9.416345668851653e-05,
    #         'rev_likelihood': 0.0002605487275852179
    #       }
    # hps.append(hp4)

    # # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=50),
    # #               sc.Subcampaign(77, .62, 565, 0.48, snr=50),
    # #               sc.Subcampaign(75, .67, 573, 0.43, snr=50),
    # #               sc.Subcampaign(65, .68, 503, 0.47, snr=50),
    # #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=50)] #,

    # hps = []

    # hp0 = { 'cost_variance': 0.32419976395136274, 
    #         'rev_variance': 0.7743237769013634,
    #         'cost_lengthscales': 0.676087919699994,
    #         'rev_lengthscales': 0.41857192551841677,
    #         'cost_likelihood': 4.031930997181654e-05,
    #         'rev_likelihood': 4.169799100831181e-05
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.4204467273523313,
    #         'rev_variance': 0.7993575729307415,
    #         'cost_lengthscales': 0.6224546095212766,
    #         'rev_lengthscales': 0.4688418740537017,
    #         'cost_likelihood': 4.0626498315690266e-05,
    #         'rev_likelihood': 4.180661808387804e-05
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.3980146924811013,
    #         'rev_variance': 1.0115434971375619,
    #         'cost_lengthscales': 0.6595875568229689,
    #         'rev_lengthscales': 0.45155004637052476,
    #         'cost_likelihood': 4.0516759053911336e-05,
    #         'rev_likelihood': 4.211058976545578e-05
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.3550551213053936,
    #         'rev_variance': 0.7297186535075504,
    #         'cost_lengthscales': 0.6948292105691559,
    #         'rev_lengthscales': 0.4814166725525413,
    #         'cost_likelihood': 4.038041329983571e-05,
    #         'rev_likelihood': 4.141664928044492e-05
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.38933953668979987,
    #         'rev_variance': 0.8549943313731717,
    #         'cost_lengthscales': 0.7181966921918171,
    #         'rev_lengthscales': 0.40842279053972613,
    #         'cost_likelihood': 4.038852043790623e-05,
    #         'rev_likelihood': 4.204577069728798e-05
    #       }
    # hps.append(hp4)

    # #campaign_g2 = [sc.Subcampaign(60, 0.45, 497, .31, snr=29),
    # #              sc.Subcampaign(77, .52, 565, 0.38, snr=29),
    # #              sc.Subcampaign(75, .47, 573, 0.35, snr=29),
    # #              sc.Subcampaign(65, .58, 503, 0.43, snr=29),
    # #              sc.Subcampaign(70, 0.59, 536, 0.38, snr=29)] #,

    # hps = []

    # hp0 = { 'cost_variance': 0.9016192642763469, 
    #         'rev_variance': 1.1395017859584244,
    #         'cost_lengthscales': 0.5127401463055302,
    #         'rev_lengthscales': 0.3492118723057916,
    #         'cost_likelihood': 0.0003182557998018349,
    #         'rev_likelihood': 0.000719511020309407
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.730508135434845,
    #         'rev_variance': 1.3996785580059876,
    #         'cost_lengthscales': 0.4972420968684829,
    #         'rev_lengthscales': 0.43250071498454534,
    #         'cost_likelihood': 0.0005098180886308955,
    #         'rev_likelihood': 0.0008074694208573991
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.6836667812355607,
    #         'rev_variance': 0.9092036664517563,
    #         'cost_lengthscales': 0.41588770465192315,
    #         'rev_lengthscales': 0.31751701521179876,
    #         'cost_likelihood': 0.000491372770813165,
    #         'rev_likelihood': 0.0008586456831206621
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.8303995233938715,
    #         'rev_variance': 0.6845877274587653,
    #         'cost_lengthscales': 0.6320606830194324,
    #         'rev_lengthscales': 0.4052564800190061,
    #         'cost_likelihood': 0.00033286345963469736,
    #         'rev_likelihood': 0.0006397714587408444
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.482875245747199,
    #         'rev_variance': 1.0117095421889417,
    #         'cost_lengthscales': 0.46743413458216404,
    #         'rev_lengthscales': 0.3536627922966732,
    #         'cost_likelihood': 0.0003828103911428222,
    #         'rev_likelihood': 0.0007645050041974488
    #       }
    # hps.append(hp4)


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
    # # rnd_campaign_01 = [
    # #             sc.Subcampaign(83.0, 0.9390275741377971, 530.0, 0.35653961965235176, snr=34),
    # #             sc.Subcampaign(97.0, 0.8565310891376972, 417.0, 0.6893952328071604, snr=30),
    # #             sc.Subcampaign(72.0, 0.4845001118567374, 548.0, 0.29997157236918714, snr=35),
    # #             sc.Subcampaign(100.0, 0.6618767352251798, 571.0, 0.5709120360333635, snr=34),
    # #             sc.Subcampaign(96.0, 0.24623017989617788, 550.0, 0.24553470046274206, snr=31)
    # # #             ]

    # hps = []

    # hp0 = { 'cost_variance': 0.5714255029435338, 
    #         'rev_variance': 0.9094424856134248,
    #         'cost_lengthscales': 0.8206937822368247,
    #         'rev_lengthscales': 0.4148244702799004,
    #         'cost_likelihood': 8.263199806301899e-05,
    #         'rev_likelihood': 0.000150610336764791
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.4898625472620412,
    #         'rev_variance': 0.3208321196294313,
    #         'cost_lengthscales': 0.6686839890464469,
    #         'rev_lengthscales': 0.5273151914128179,
    #         'cost_likelihood': 0.00019766781560771185,
    #         'rev_likelihood': 0.00016529546900978984
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.4983699407182139,
    #         'rev_variance': 1.5853015408033715,
    #         'cost_lengthscales': 0.46153438570302774,
    #         'rev_lengthscales': 0.3848859138222554,
    #         'cost_likelihood': 7.601605719871999e-05,
    #         'rev_likelihood': 0.00014045376488632557
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.6913466249078662,
    #         'rev_variance': 0.38224980950886617,
    #         'cost_lengthscales': 0.5892063740914937,
    #         'rev_lengthscales': 0.3967635807589267,
    #         'cost_likelihood': 0.00012351589989881506,
    #         'rev_likelihood': 0.00013951698069128123
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.4600709198698711,
    #         'rev_variance': 1.4702037503054994,
    #         'cost_lengthscales': 0.24931826454985578,
    #         'rev_lengthscales': 0.3409595651329481,
    #         'cost_likelihood': 0.0002917121125016849,
    #         'rev_likelihood': 0.00030646542151017243
    #       }
    # hps.append(hp4)

    # # rnd_campaign_02 = [
    # #             sc.Subcampaign(83.0, 0.22405720652229322, 597.0, 0.2022227936494456, snr=30),
    # #             sc.Subcampaign(98.0, 0.8499901193552486, 682.0, 0.5208800234776618, snr=31),
    # #             sc.Subcampaign(56.0, 0.7267762997014247, 698.0, 0.3670485687338533, snr=31),
    # #             sc.Subcampaign(60.0, 0.559426504706663, 456.0, 0.39340047292135694, snr=32),
    # #             sc.Subcampaign(51.0, 0.7831724596643996, 444.0, 0.6895966545162417, snr=32)
    # # #             ]

    # hps = []

    # hp0 = { 'cost_variance': 0.8469049810225547, 
    #         'rev_variance': 1.1944882327598165,
    #         'cost_lengthscales': 0.28639416982442795,
    #         'rev_lengthscales': 0.263505960861699,
    #         'cost_likelihood': 0.00029715353851408416,
    #         'rev_likelihood': 0.0005112949281804639
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.5825792886452279,
    #         'rev_variance': 0.8156515600333464,
    #         'cost_lengthscales': 0.7376689460123516,
    #         'rev_lengthscales': 0.4943563593741877,
    #         'cost_likelihood': 0.00017509999206957307,
    #         'rev_likelihood': 0.0003630180632680592
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.3077278936277148,
    #         'rev_variance': 0.5937240417804808,
    #         'cost_lengthscales': 0.6360359875454311,
    #         'rev_lengthscales': 0.2979073101379007,
    #         'cost_likelihood': 8.634969389537303e-05,
    #         'rev_likelihood': 0.0004782870767060289
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.37703587874060207,
    #         'rev_variance': 0.8142517280632858,
    #         'cost_lengthscales': 0.5226692733463462,
    #         'rev_lengthscales': 0.4567714433806241,
    #         'cost_likelihood': 8.810442197983003e-05,
    #         'rev_likelihood': 0.00016758490616617096
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.39357548373450824,
    #         'rev_variance': 0.5058993151625275,
    #         'cost_lengthscales': 0.795443743902963,
    #         'rev_lengthscales': 0.6275811873237777,
    #         'cost_likelihood': 6.735895440982739e-05,
    #         'rev_likelihood': 0.00012588080648821013
    #       }
    # hps.append(hp4)



    # # rnd_campaign_03 = [
    # #             sc.Subcampaign(97.0, 0.2254717323719329, 570.0, 0.21712317683804871, snr=30),
    # #             sc.Subcampaign(78.0, 0.6809006218688995, 514.0, 0.638902808265124, snr=31),
    # #             sc.Subcampaign(53.0, 1.0518056176361688, 426.0, 0.6940683038810456, snr=34),
    # #             sc.Subcampaign(80.0, 0.4129454844360104, 469.0, 0.39195310889540247, snr=32),
    # #             sc.Subcampaign(82.0, 0.9188239551808393, 548.0, 0.34500566310456804, snr=30)
    # # #             ]
    # hps = []

    # hp0 = { 'cost_variance': 1.1324731104567878, 
    #         'rev_variance': 0.7422523630380573,
    #         'cost_lengthscales': 0.3111346169341272,
    #         'rev_lengthscales': 0.24299374315660313,
    #         'cost_likelihood': 0.00038056464010249695,
    #         'rev_likelihood': 0.000451555256084912
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.43623036621881467,
    #         'rev_variance': 0.7408732490005936,
    #         'cost_lengthscales': 0.5633223962581508,
    #         'rev_lengthscales': 0.6858747276064396,
    #         'cost_likelihood': 0.0001436811914266462,
    #         'rev_likelihood': 0.00018773356836017204
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.3206145915857011,
    #         'rev_variance': 0.5202722503984358,
    #         'cost_lengthscales': 0.9730366211293333,
    #         'rev_lengthscales': 0.6285073429063599,
    #         'cost_likelihood': 5.345497622468908e-05,
    #         'rev_likelihood': 8.767179030781554e-05
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.596360657794305,
    #         'rev_variance': 0.7022046696731797,
    #         'cost_lengthscales': 0.44308489114775546,
    #         'rev_lengthscales': 0.43517043174855696,
    #         'cost_likelihood': 0.00014837067141494504,
    #         'rev_likelihood': 0.00018427498154320414
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.28610534254943476,
    #         'rev_variance': 0.4920105080569547,
    #         'cost_lengthscales': 0.5579053908933993,
    #         'rev_lengthscales': 0.32022906251136285,
    #         'cost_likelihood': 0.0001532709418353767,
    #         'rev_likelihood': 0.000354287065796406
    #       }
    # hps.append(hp4)

    # # rnd_campaign_04 = [
    # #             sc.Subcampaign(62.0, 0.460443097667253, 487.0, 0.34862416791066253, snr=32),
    # #             sc.Subcampaign(79.0, 1.0217507506678771, 494.0, 0.4244629979053927, snr=30),
    # #             sc.Subcampaign(76.0, 0.5159200609033661, 467.0, 0.32645838278958483, snr=33),
    # #             sc.Subcampaign(69.0, 0.8948304325148888, 684.0, 0.7226499482261013, snr=34),
    # #             sc.Subcampaign(99.0, 1.0568307954024765, 494.0, 0.26571147363891007, snr=34)
    # # #             ]


    # hps = []

    # hp0 = { 'cost_variance': 0.651049401021401, 
    #         'rev_variance': 0.703040676982267,
    #         'cost_lengthscales': 0.4379374675758587,
    #         'rev_lengthscales': 0.3608961551565295,
    #         'cost_likelihood': 0.00018506672313839016,
    #         'rev_likelihood': 0.00023076592754809605
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.6417766729533382,
    #         'rev_variance': 0.729981629883889,
    #         'cost_lengthscales': 0.8518690546134389,
    #         'rev_lengthscales': 0.41176456956434043,
    #         'cost_likelihood': 0.0002472075999104216,
    #         'rev_likelihood': 0.0003493293867325081
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.5449615302561648,
    #         'rev_variance': 0.5891966065037081,
    #         'cost_lengthscales': 0.40047769098022096,
    #         'rev_lengthscales': 0.30395092467855256,
    #         'cost_likelihood': 0.00020328647054779797,
    #         'rev_likelihood': 0.0002031862843895277
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.675821315439306,
    #         'rev_variance': 0.8258611903014537,
    #         'cost_lengthscales': 0.7606471402412487,
    #         'rev_lengthscales': 0.5830681629934436,
    #         'cost_likelihood': 0.00011389584127699436,
    #         'rev_likelihood': 0.00021062299066999183
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.7832867887458823,
    #         'rev_variance': 1.2136916276448753,
    #         'cost_lengthscales': 0.7796348246136865,
    #         'rev_lengthscales': 0.32417410409766845,
    #         'cost_likelihood': 0.00016926542978340205,
    #         'rev_likelihood': 0.00018011193702387648
    #       }
    # hps.append(hp4)

    # rnd_campaign_05 = [
    #             sc.Subcampaign(52.0, 0.7236376036521779, 525.0, 0.2581539194710056, snr=31),
    #             sc.Subcampaign(87.0, 0.8347532352531297, 643.0, 0.607729693610427, snr=35),
    #             sc.Subcampaign(68.0, 1.0545161819092699, 455.0, 0.39019889273979436, snr=33),
    #             sc.Subcampaign(99.0, 1.0715069147863183, 440.0, 0.7409946097511, snr=33),
    #             sc.Subcampaign(94.0, 0.9434443599935285, 600.0, 0.3880212001600485, snr=33)
    # #             ]

    hps = []

    hp0 = { 'cost_variance': 0.39242030541047845, 
            'rev_variance': 1.053454224301622,
            'cost_lengthscales': 0.606199747264256,
            'rev_lengthscales': 0.29211002074334635,
            'cost_likelihood': 0.0001391318346696047,
            'rev_likelihood': 0.0003940383323336704
          }

    hps.append(hp0)

    hp1 = { 'cost_variance': 0.44681869875340874,
            'rev_variance': 1.1170695665885946,
            'cost_lengthscales': 0.48043708361522236,
            'rev_lengthscales': 0.5402291744470531,
            'cost_likelihood': 0.000125699111527948,
            'rev_likelihood': 0.00016680714696665268
          }

    hps.append(hp1)

    hp2 = { 'cost_variance': 0.5467865884677027,
            'rev_variance': 0.5054201904013241,
            'cost_lengthscales': 0.8299812758283986,
            'rev_lengthscales': 0.31943834494217527,
            'cost_likelihood': 0.00011422120801111565,
            'rev_likelihood': 0.00018277305621309576
          }
    hps.append(hp2)

    hp3 = { 'cost_variance': 0.804992979605706,
            'rev_variance': 0.7093139970480768,
            'cost_lengthscales': 0.7648295048161938,
            'rev_lengthscales': 0.6894815823910733,
            'cost_likelihood': 0.00020469835372356052,
            'rev_likelihood': 0.00012021960491663314
          }
    hps.append(hp3)

    hp4 = { 'cost_variance': 0.8122399091442097,
            'rev_variance': 0.8037009695311311,
            'cost_lengthscales': 0.7531882803621165,
            'rev_lengthscales': 0.3299669278624351,
            'cost_likelihood': 0.0002019067910999419,
            'rev_likelihood': 0.0002899610209649814
          }
    hps.append(hp4)

    # # rnd_campaign_06 = [
    # #             sc.Subcampaign(71.0, 0.8750247925943013, 617.0, 0.8440732817094825, snr=31),
    # #             sc.Subcampaign(53.0, 0.8411068818594563, 518.0, 0.6772664432914361, snr=33),
    # #             sc.Subcampaign(87.0, 1.0703614987489847, 547.0, 0.8667712665875702, snr=32),
    # #             sc.Subcampaign(98.0, 0.6310736950229558, 567.0, 0.2521616349302094, snr=31),
    # #             sc.Subcampaign(59.0, 0.28867007783272997, 576.0, 0.24763783882935328, snr=32)
    # #             ]

    # hps = []

    # hp0 = { 'cost_variance': 0.5630561107620213, 
    #         'rev_variance': 0.7550029840781375,
    #         'cost_lengthscales': 0.6378810563130448,
    #         'rev_lengthscales': 0.6624151039069884,
    #         'cost_likelihood': 0.00018778808818201,
    #         'rev_likelihood': 0.0002744385557676864
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.39110191008145667,
    #         'rev_variance': 0.5568439380906406,
    #         'cost_lengthscales': 0.587654479129593,
    #         'rev_lengthscales': 0.513623986435323,
    #         'cost_likelihood': 9.154644786831486e-05,
    #         'rev_likelihood': 0.0001754320258527301
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.5115058152527682,
    #         'rev_variance': 0.5647279066527712,
    #         'cost_lengthscales': 0.6050282910361209,
    #         'rev_lengthscales': 0.6009411194485712,
    #         'cost_likelihood': 0.00019209401684412114,
    #         'rev_likelihood': 0.0001807142511157444
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.671392763323708,
    #         'rev_variance': 0.9148855572060799,
    #         'cost_lengthscales': 0.44328454747674895,
    #         'rev_lengthscales': 0.2662405758484695,
    #         'cost_likelihood': 0.00041759121853629977,
    #         'rev_likelihood': 0.0004123839117982562
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.8249296579521309,
    #         'rev_variance': 1.0883587755152835,
    #         'cost_lengthscales': 0.33811898478080915,
    #         'rev_lengthscales': 0.26836560048827407,
    #         'cost_likelihood': 0.00019386556154333003,
    #         'rev_likelihood': 0.0003687495638556048
    #       }
    # hps.append(hp4)

    # # rnd_campaign_07 = [
    # #             sc.Subcampaign(77.0, 0.8109246960855101, 409.0, 0.5079367092300731, snr=31),
    # #             sc.Subcampaign(78.0, 0.2469886328248737, 592.0, 0.23086355578873, snr=31),
    # #             sc.Subcampaign(91.0, 0.7744431310645019, 628.0, 0.5712510728701953, snr=35),
    # #             sc.Subcampaign(50.0, 0.5162810922290018, 613.0, 0.359321181934275, snr=34),
    # #             sc.Subcampaign(71.0, 0.3794531987004781, 513.0, 0.3077863622385426, snr=35)
    # #             ]
    # hps = []

    # hp0 = { 'cost_variance': 0.6915600956472779, 
    #         'rev_variance': 0.8189502589694475,
    #         'cost_lengthscales': 0.6422683936889187,
    #         'rev_lengthscales': 0.5676443832471718,
    #         'cost_likelihood': 0.00023816608867979378,
    #         'rev_likelihood': 0.0001934883173308735
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 1.0972125595237503,
    #         'rev_variance': 1.5383720290073732,
    #         'cost_lengthscales': 0.2863681851617017,
    #         'rev_lengthscales': 0.3026575518620878,
    #         'cost_likelihood': 0.0004088217981299267,
    #         'rev_likelihood': 0.000517565086688922
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.9184321461151556,
    #         'rev_variance': 1.0132537696465742,
    #         'cost_lengthscales': 0.6420033408065684,
    #         'rev_lengthscales': 0.49686775688510854,
    #         'cost_likelihood': 0.00015035370351175573,
    #         'rev_likelihood': 0.00018269823135369
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.48095895864266436,
    #         'rev_variance': 1.9259912161808987,
    #         'cost_lengthscales': 0.4565330898211615,
    #         'rev_lengthscales': 0.42798941964706494,
    #         'cost_likelihood': 9.279867200568286e-05,
    #         'rev_likelihood': 0.00023109140589531546
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.7321400383235943,
    #         'rev_variance': 1.171679089607045,
    #         'cost_lengthscales': 0.3505189657326618,
    #         'rev_lengthscales': 0.33604845683075507,
    #         'cost_likelihood': 0.00014586340756177817,
    #         'rev_likelihood': 0.0001522010725904958
    #       }
    # hps.append(hp4)

    # # rnd_campaign_08 = [
    # #             sc.Subcampaign(67.0, 0.6715707414871959, 602.0, 0.3266023723522325, snr=30),
    # #             sc.Subcampaign(80.0, 0.7751757702430537, 605.0, 0.2658241604853398, snr=32),
    # #             sc.Subcampaign(99.0, 0.44063783263446604, 618.0, 0.2013497671860729, snr=31),
    # #             sc.Subcampaign(77.0, 0.31016111269832836, 505.0, 0.21944833868391506, snr=31),
    # #             sc.Subcampaign(99.0, 0.4059724156582363, 588.0, 0.2913060845388664, snr=32)
    # #             ]

    # hps = []

    # hp0 = { 'cost_variance': 0.7531757683719315, 
    #         'rev_variance': 0.9200151444407842,
    #         'cost_lengthscales': 0.6874953525011533,
    #         'rev_lengthscales': 0.3302093377508162,
    #         'cost_likelihood': 0.00026580073021310045,
    #         'rev_likelihood': 0.000567129177472056
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.632860840123084,
    #         'rev_variance': 0.8626601140412962,
    #         'cost_lengthscales': 0.583314824726294,
    #         'rev_lengthscales': 0.2667267953010661,
    #         'cost_likelihood': 0.00020689842035973575,
    #         'rev_likelihood': 0.00041062824484630264
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.807185113974855,
    #         'rev_variance': 1.7149486132734062,
    #         'cost_lengthscales': 0.348889247295266,
    #         'rev_lengthscales': 0.28630594302949247,
    #         'cost_likelihood': 0.0005932357360446335,
    #         'rev_likelihood': 0.0005566759808692388
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.28873152440311217,
    #         'rev_variance': 1.4074356353634363,
    #         'cost_lengthscales': 0.21951831432450977,
    #         'rev_lengthscales': 0.3286890067546473,
    #         'cost_likelihood': 0.00037667530912578403,
    #         'rev_likelihood': 0.0003699399269217344
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 1.2488165645924847,
    #         'rev_variance': 0.9629966654530842,
    #         'cost_lengthscales': 0.42027527356682814,
    #         'rev_lengthscales': 0.29814161243061216,
    #         'cost_likelihood': 0.0004511992333645435,
    #         'rev_likelihood': 0.00035146009514089283
    #       }
    # hps.append(hp4)

    # # rnd_campaign_09 = [
    # #             sc.Subcampaign(53.0, 0.6181790644241923, 486.0, 0.4181144656709901, snr=35),
    # #             sc.Subcampaign(82.0, 0.8639181969833853, 684.0, 0.3300544004489604, snr=34),
    # #             sc.Subcampaign(58.0, 0.6695803386210537, 547.0, 0.5291694460820753, snr=33),
    # #             sc.Subcampaign(96.0, 0.8661364638643831, 419.0, 0.7291640886632436, snr=35),
    # #             sc.Subcampaign(100.0, 0.8319412598597624, 453.0, 0.67917291744797, snr=30)
    # #             ]

    # hps = []

    # hp0 = { 'cost_variance': 0.44793871977043254, 
    #         'rev_variance': 0.4890044749639321,
    #         'cost_lengthscales': 0.4804125641505192,
    #         'rev_lengthscales': 0.34558301219106347,
    #         'cost_likelihood': 7.829631471894638e-05,
    #         'rev_likelihood': 0.00012551928221110823
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 0.7823948485095233,
    #         'rev_variance': 1.2369072567782382,
    #         'cost_lengthscales': 0.7556605656770353,
    #         'rev_lengthscales': 0.33864829636774096,
    #         'cost_likelihood': 0.0001320577663817049,
    #         'rev_likelihood': 0.0003055291272297315
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 0.22812294279636613,
    #         'rev_variance': 0.9838401293189889,
    #         'cost_lengthscales': 0.40148411863543676,
    #         'rev_lengthscales': 0.524782130968852,
    #         'cost_likelihood': 0.000112715406115267,
    #         'rev_likelihood': 0.00021310376873851693
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.8664960110837066,
    #         'rev_variance': 0.6622691440329059,
    #         'cost_lengthscales': 0.6944538874237659,
    #         'rev_lengthscales': 0.6426360668567149,
    #         'cost_likelihood': 0.0001540881464857759,
    #         'rev_likelihood': 8.527646414556351e-05
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.7771188946281632,
    #         'rev_variance': 0.6251761276050931,
    #         'cost_lengthscales': 0.5609993892347229,
    #         'rev_lengthscales': 0.6404669171440422,
    #         'cost_likelihood': 0.0004922727307710513,
    #         'rev_likelihood': 0.00021994938095711754
    #       }
    # hps.append(hp4)

    # # rnd_campaign_10 = [
    # #             sc.Subcampaign(51.0, 1.049310026718121, 617.0, 0.2051759576328354, snr=30),
    # #             sc.Subcampaign(86.0, 0.7797153486111319, 520.0, 0.539937043644524, snr=35),
    # #             sc.Subcampaign(93.0, 0.23347185518099078, 422.0, 0.217994206233266, snr=30),
    # #             sc.Subcampaign(61.0, 0.5780205490681166, 559.0, 0.49006423739237187, snr=30),
    # #             sc.Subcampaign(84.0, 0.562291455390848, 457.0, 0.22447254161352354, snr=30)
    # #             ]

    # hps = []

    # hp0 = { 'cost_variance': 0.40428789770731965, 
    #         'rev_variance': 1.0217000612450298,
    #         'cost_lengthscales': 0.8459683990238257,
    #         'rev_lengthscales': 0.24155273932269924,
    #         'cost_likelihood': 0.00012206415468595707,
    #         'rev_likelihood': 0.0006832723204529774
    #       }

    # hps.append(hp0)

    # hp1 = { 'cost_variance': 1.0569400702911644,
    #         'rev_variance': 0.7600745065806216,
    #         'cost_lengthscales': 0.7295420977792082,
    #         'rev_lengthscales': 0.4941990667397541,
    #         'cost_likelihood': 0.00013881892212680436,
    #         'rev_likelihood': 0.000132774975382408
    #       }

    # hps.append(hp1)

    # hp2 = { 'cost_variance': 1.9472074874255152,
    #         'rev_variance': 0.5240722679909098,
    #         'cost_lengthscales': 0.30492218243086167,
    #         'rev_lengthscales': 0.237581414382023,
    #         'cost_likelihood': 0.0007243588821661198,
    #         'rev_likelihood': 0.000311193508229231
    #       }
    # hps.append(hp2)

    # hp3 = { 'cost_variance': 0.5659894723686771,
    #         'rev_variance': 0.9458138687755313,
    #         'cost_lengthscales': 0.537760097121161,
    #         'rev_lengthscales': 0.5092926441850676,
    #         'cost_likelihood': 0.00023040920423558864,
    #         'rev_likelihood': 0.0004629394953962
    #       }
    # hps.append(hp3)

    # hp4 = { 'cost_variance': 0.7281754985756607,
    #         'rev_variance': 0.859780170293484,
    #         'cost_lengthscales': 0.4968247588057028,
    #         'rev_lengthscales': 0.28582507403809926,
    #         'cost_likelihood': 0.0004363076715009954,
    #         'rev_likelihood': 0.00037634031895254
    #       }
    # hps.append(hp4)

    return hps



def multiple_runs(subcampaigns, agents, runs_num, T=3):
    path = str(uuid.uuid4().hex)
    pathlib.Path(f'{path}').mkdir(parents=True, exist_ok=False)

    subcampaigns = np.array(subcampaigns)
    # agents = np.array(agents)

    settings = subcampaigns, agents, runs_num, T
    # settings = zip(*settings)

    hps = fix_hps()

    with open(f'{path}/config.pickle', 'wb') as f:
        pl.dump(MAX_BID, f)
        pl.dump(MAX_EXP, f)
        pl.dump(N_BID, f)
        pl.dump(N_COST,f)
        pl.dump(SAMPLES_INVERSE, f)
        pl.dump(FIRST_BID, f)
        pl.dump(MIN_ROI, f)
        pl.dump(MAX_ROI, f)
        pl.dump(LOW_ROI, f)
        pl.dump(N_ROI,f)
    
    with open(f'{path}/settings.pickle', 'wb') as f:
        pl.dump(runs_num, f)
    #with open(f'{path}/settings.pickle', 'wb') as f:
        # pl.dump(runs_num, f)
        pl.dump(T, f)
    #with open(f'{path}/settings.pickle', 'wb') as f:
        pl.dump(subcampaigns, f)  # , -1)
        #for a in agents:
        #with open(f'{path}/settings.pickle', 'wb') as f:
        # pl.dump([f'{type(a).__name__}' for a in agents], f, -1)
        pl.dump([f'{a.__str__()}' for a in agents], f, -1)
            # pl.dump(a, f, -1)

    # p = Pool(4)
    # with Pool(4) as p:
    results = []
    #     results = p.starmap(run_environment, zip(itertools.repeat(copy.copy(subcampaigns)), itertools.repeat(copy.copy(agents)), itertools.repeat(path), np.array(range(runs_num)), itertools.repeat(T)))  
    for r in range(runs_num):
        # b = agent.TS_Agent(len(subcampaigns))
        # e = agent.TS_Conservative_Agent_1(len(subcampaigns))
        # f = agent.TS_Conservative_Agent_5(len(subcampaigns))
        # g = agent.TS_Conservative_Agent_10(len(subcampaigns))
        # h = agent.Safe_Agent(len(subcampaigns))
        # l = agent.Safe_Agent_rev(len(subcampaigns))
        # m = agent.Safe_Agent_delta(len(subcampaigns))
        # n = agent.Safe_eps_greedy_Agent(len(subcampaigns))
        # p = agent.Safe_Agent_sample_cost(len(campaign))
        # q = agent.Safe_Agent_lb_cost(len(campaign))
        # rrr = agent.Safe_Opt_Agent(len(campaign))
        # su = agent.Safe_Opt_eps_greedy_Agent(len(campaign))
        gcb = agent.GCB_Agent(len(campaign), T=T, hps=hps)
        gcbsafe = agent.GCBsafe_Agent(len(campaign), T=T, hps=hps)
        gcbsafe95 = agent.GCBsafe95_Agent(len(campaign), T=T, hps=hps, eps=0.95)
        gcbsafe90 = agent.GCBsafe90_Agent(len(campaign), T=T, hps=hps, eps=0.90)
        # gcbsafe85 = agent.GCBsafe85_Agent(len(campaign), T=T, hps=hps, eps=0.85)
        # c = agent.TS_TFP_Agent(len(subcampaigns))
        # d = agent.TS_MCMC_Agent(len(subcampaigns))
       # c = agent.Clairvoyant_Agent(subcampaigns)
        # agents = [b, e, f, g]  # , c, d]  # [a, b, c, d]
        #agents = [b, g, h]
        #agents = [h]
        #agents = [rrr, su]
        # agents = [gcb, gcbsafe95]  # , rrr]
        agents = [gcb, gcbsafe, gcbsafe95, gcbsafe90]  # , rrr]
        #agents = [gcbsafe, gcbsafe95, gcbsafe90, gcbsafe85]
        # agents = [n]
        #agents = [e]
        results.append(run_environment(subcampaigns, agents, path, r, T))
    # p.close()
    # p.join()
    # results = np.asarray(results)
    # print(results)
    with open(f'{path}/results.pickle', 'wb') as f:

        pl.dump(runs_num, f)
    #with open(f'{path}/settings.pickle', 'wb') as f:
        # pl.dump(runs_num, f)
        pl.dump(T, f)
    #with open(f'{path}/settings.pickle', 'wb') as f:
        pl.dump(subcampaigns, f)  # , -1)
        #for a in agents:
        #with open(f'{path}/settings.pickle', 'wb') as f:
        #pl.dump([f'{type(a).__name__}' for a in agents], f)  # , -1)
        pl.dump([f'{(a).__str__()}' for a in agents], f)  # , -1)
        pl.dump(results, f)
    
    return settings, results, path


def run_environment(subcampaigns, agents, path, run_n, T=31):
    # path
    print(f'run:{run_n}\nagents:\n{agents}\nsubc:\n{subcampaigns}\n')
    print('oi')
    #logger.error('oi')

    rundir = f'run_{run_n}'
    pathlib.Path(f'{path}/{rundir}').mkdir(parents=True, exist_ok=True)
    path = f'{path}/{rundir}'
    pathlib.Path(f'{path}/pickle').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/plots').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/tikz').mkdir(parents=True, exist_ok=True)
    # pathlib.Path(f'{path}/pickle/gp').mkdir(parents=True, exist_ok=True)
    # pathlib.Path(f'{path}/plots/gp').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/gp/pickle').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/gp/plots').mkdir(parents=True, exist_ok=True)
    # pathlib.Path(f'{path}/pickle/roi').mkdir(parents=True, exist_ok=True)
    # pathlib.Path(f'{path}/plots/roi').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/rev/pickle').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/rev/plots').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/rev/revnantozero/pickle').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/rev/revnantozero/plots').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/roi/pickle').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/roi/plots').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/roi/exp_roi/pickle').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/roi/exp_roi/plots').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/roi/roi_des/pickle').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/roi/roi_des/plots').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/roi/roinotnan/pickle').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/roi/roinotnan/plots').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/roi/allroi/pickle').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/roi/allroi/plots').mkdir(parents=True, exist_ok=True)
    #pathlib.Path(f'{path}/xv/pickle').mkdir(parents=True, exist_ok=True)
    #pathlib.Path(f'{path}/xv/plots').mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.ERROR,  # WARNING,
                        # format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        # datefmt='%m-%d %H:%M',
                        filename=f'{path}/mygpapprun{run_n}.log',
                        filemode='w')

    #logging.disable(logging.CRITICAL)

    # logging.getLogger('preprocess').setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    
    r = environment(subcampaigns, agents, T, path, Plot=True if run_n==-1 else False)
    # np.save(f'{path}/data_run{run_n}.npy', r)

    show_results(subcampaigns, agents, *r, path=path)


    with open(f'{path}/data_{run_n}.npy', 'wb') as f:
        pl.dump(r, f)
    return r

    # files
    # environment
    # save results
    # return

def environment(subcampaigns, agents, T=31, path=None, Show_GP=True, Plot=False):
    # observations
    X = np.full((len(agents), len(subcampaigns), T+1), 0.0)
    Y_cost = np.full((len(agents), len(subcampaigns), T), 0.0)
    f_cost = np.full((len(agents), len(subcampaigns), T), 0.0)
    Y_rev = np.full((len(agents), len(subcampaigns), T), 0.0)
    f_rev = np.full((len(agents), len(subcampaigns), T), 0.0)

    # expected values
    exp_roi = np.full((len(agents), len(subcampaigns), T+1), 0.0)
    exp_cost = np.full((len(agents), len(subcampaigns), T+1), 0.0)
    exp_rev = np.full((len(agents), len(subcampaigns), T+1), 0.0)

    cost_array = np.linspace(0, MAX_EXP, N_COST)
    bid_array = np.linspace(0, MAX_BID, N_BID)

    # agents data
    r_data = np.full((len(agents), T, cost_array.shape[0]), np.empty(cost_array.shape))
    roi_data = np.full((len(agents), T, cost_array.shape[0]), np.empty(cost_array.shape))
    
    # gps data
    real_sc_costs = [s.cost(bid_array, noise=False) for s in subcampaigns]
    real_sc_revs = [s.revenue(bid_array, noise=False) for s in subcampaigns]

    mean_gps_cost = np.full((len(agents), len(subcampaigns), T, bid_array.shape[0]), np.empty(bid_array.shape))
    mean_gps_rev = np.full((len(agents), len(subcampaigns), T, bid_array.shape[0]),  np.empty(bid_array.shape))
    var_gps_cost = np.full((len(agents), len(subcampaigns), T, bid_array.shape[0]), np.empty(bid_array.shape))
    var_gps_rev = np.full((len(agents), len(subcampaigns), T, bid_array.shape[0]), np.empty(bid_array.shape))


    c = agent.Clairvoyant_Agent(subcampaigns)
    bid_opt, target_rev, target_roi, target_rev_mix, target_cost_mix, idx_opt = c.bid_choice(ret_roi_revenue_cost=True)
    if Plot:
        # idx = np.array(np.where(target_roi>0.0)).ravel()[0] # first non-zero element
        idx = 1
        plot.plot(x=target_rev[idx:], y=target_roi[idx:], label='optimum ROI',  # clairvoyant agent ROI-revenue',
                  x_label='revenue', y_label='ROI',
                  name='optimum_roi',
                  Show=False, path=path)

        plot.plot(x=target_rev[~np.isnan(target_roi)], y=target_roi[~np.isnan(target_roi)], label='optimum ROI',  # clairvoyant agent ROI-revenue',
                  x_label='revenue', y_label='ROI',
                  name='optimum_roi_no_nan',
                  Show=False, path=path)

        idx_optimum_max = np.nanargmax(target_roi)
        # print('idx opt', idx_optimum_max)
        # print('target roi\n', target_roi[idx_optimum_max])
        # print('full target roi\n', target_roi)
        plot.plot(x=target_rev[idx_optimum_max:], y=target_roi[idx_optimum_max:], label='optimum ROI',  # clairvoyant agent ROI-revenue',
                  x_label='revenue', y_label='ROI',
                  name='optimum_nospike_roi',
                  Show=False, path=path)
        plot.plot(x=cost_array, y=target_rev, label='optimum revenue',  #clairvoyant agent revenue-cost', 
                  x_label='cost', y_label='revenue', Show=False, path=path)
        # return
    # day 0; 
    for i, a in enumerate(agents):
        X[i, :, 0] = FIRST_BID  # MAX_BID / N_BID  # first bid identical  # a.bid_choice() 
        # X[0][i] = a.bid_choice()

    if logger.isEnabledFor(logging.WARNING):
        logger.warning(f'X[0]:\n{X[0]}')
    for t in range(T):
        logging.warning(f'time:{t}')
        # get observations
        for i in range(len(agents)):
            for j, s in enumerate(subcampaigns):
                logging.warning(f'X[{i}][{j}][{t}]: {X[i][j][t]}')

                Y_cost[i][j][t] = s.cost(X[i][j][t])
                f_cost[i][j][t] = s.cost(X[i][j][t], noise=False)

                logging.warning(f'Y_cost[{i}][{j}][{t}]: {Y_cost[i][j][t]}')

                Y_rev[i][j][t] = s.revenue(X[i][j][t])
                f_rev[i][j][t] = s.revenue(X[i][j][t], noise=False)

                logging.warning(f'Y_rev[{i}][{j}][{t}]: {Y_rev[i][j][t]}')

        # update models
        for i, a in enumerate(agents):
            a.update(X[i, :, t], Y_cost[i, :, t], Y_rev[i, :, t])
            #logging.warning(f'Y_rev[{t}][{j}]: {Y_rev[t][j]}')
            #logging.warning(f'Y_cost[{t}][{j}]: {Y_cost[t][j]}')

        # get bid and expected values

        for i, a in enumerate(agents):
            r, exp_bid_mix, exp_rev_mix, exp_cost_mix = a.exp_revenue()

            bid_mix, _, roi, rev_mix, cost_mix, idx = (
                            a.bid_choice(ret_roi_revenue_cost=True)
                            )
            
            # second bid
            # if t == 0:
            #     bid_mix[idx] = np.full((1, len(subcampaigns)), SECOND_BID)
            if t < INITIAL_BIDS -1:
                bid_mix[idx] = np.full((1, len(subcampaigns)), FIRST_BID*(t+2)*1.)  #2.)  # 3.)


            X[i, :, t+1], exp_roi[i, :, t+1], exp_rev[i, :, t+1], exp_cost[i, :, t+1] = (
                            bid_mix[idx], roi[idx], rev_mix[idx], cost_mix[idx]
                            )
            # X[i, :, t+1], exp_roi[i, :, t+1], exp_rev[i, :, t+1], exp_cost[i, :, t+1] = (
            #                 bid_mix[idx], roi[idx[1]], rev_mix[idx], cost_mix[idx]
            #                 )
            # plot.plot(x=cost_array, y=r)
            roi = r / cost_array
            r_data[i, t] = r
            roi_data[i, t] = roi

            exp_cost_mix_sum = np.sum(np.array(exp_cost_mix), axis=1)
            exp_rev_mix_sum = np.sum(np.array(exp_rev_mix), axis=1)
            roi_givexp = exp_rev_mix_sum / exp_cost_mix_sum
            roi_givexp_2 = r / exp_cost_mix_sum
            # roi[0] = 0
            # cost_array as sum of cost_mix
            cost_mix_sum = np.sum(np.array(cost_mix), axis=1)
            rev_mix_sum = np.sum(np.array(rev_mix), axis=1)

            

            if Plot:
                idx = 1  # np.array(np.where(target_roi>0.)).ravel()[0] # first non-zero element
                idx_roi = 1
                try:
                    idx_roi = np.array(np.where(roi>0.)).ravel()[0]
                except IndexError:
                    idx_roi = 1
                plot.confront_plot(x=cost_array, y=r, target=target_rev, x_label='cost', y_label='revenue',
                                   target_label='true revenue', label=f'estimated revenue {a.__str__()} day {t+1}',
                                   name=f'{type(a).__name__}_expected_revenue_day_{t+1:02}',
                                   Show=False, path=f'{path}/rev')
                plot.confront_plot(x=cost_array, y=np.where(np.isnan(r), 0., r),
                                   target=target_rev, x_label='cost', target_label='true revenue', y_label='revenue',
                                   label=f'estimated revenue {a.__str__()} day {t+1}',
                                   name=f'{type(a).__name__}_expected_revenue_day_{t+1:02}',
                                   Show=False, path=f'{path}/rev/revnantozero')
        # r = np.where(np.isnan(r), 0., r)
                plot.confront_plot(x=r[idx_roi:], x_target=target_rev[idx:], y=roi[idx_roi:], target=target_roi[idx:],
                                   x_label='revenue', y_label='roi', label=f'estimated ROI {a.__str__()} day {t+1}',
                                   name=f'{type(a).__name__}_expected_roi_day_{t+1:02}',
                                   target_label='true ROI',
                                   Show=False, path=f'{path}/roi')

                plot.confront_plot(x=r[idx_roi:], y=roi[idx_roi:], target=np.full((r[idx_roi:].shape[0]), MIN_ROI),
                                   x_label='revenue', y_label='roi', label=f'estimated ROI {a.__str__()} day {t+1}',
                                   name=f'{type(a).__name__}_expected_vs_min_roi_day_{t+1:02}',
                                   target_label='ROI target',
                                   Show=False, path=f'{path}/roi')

                plot.confront_plot(x=r[~np.isnan(roi)], x_target=target_rev[idx:], y=roi[~np.isnan(roi)], target=target_roi[idx:],
                                   x_label='revenue', y_label='roi', label=f'estimated ROI {a.__str__()} day {t+1}',
                                   name=f'{type(a).__name__}_expected_roi_day_{t+1:02}',
                                   target_label='true ROI',
                                   Show=False, path=f'{path}/roi/roinotnan')

                plot.confront_plot(x=r[~np.isnan(roi)], y=roi[idx_roi:], target=np.full((r[~np.isnan(roi)].shape[0]), MIN_ROI),
                                   x_label='revenue', y_label='roi', label=f'estimated ROI {a.__str__()} day {t+1}',
                                   name=f'{type(a).__name__}_expected_vs_min_roi_day_{t+1:02}',
                                   target_label='ROI target',
                                   Show=False, path=f'{path}/roi/roinotnan')
                try:
                    idx_roi_2 = np.array(np.where(rev_mix_sum/cost_mix_sum > 0)).ravel()[0]
                except IndexError:
                    idx_roi_2 = 0
                plot.confront_plot(x=rev_mix_sum[idx_roi_2:], x_target=target_rev[idx:], y=(rev_mix_sum/cost_mix_sum)[idx_roi_2:],
                                   target=target_roi[idx:], x_label='revenue', y_label='roi given sample', target_label='target roi',
                                   label=f'{type(a).__name__}_expected_roi_given_costmix_day_{t+1:02}',
                                   Show=False, path=f'{path}/roi/exp_roi')

                plot.confront_plot(x=r, x_target=target_rev, y=roi, target=target_roi, x_label='revenue', y_label='roi',
                                   target_label='true roi', label=f'estimated ROI {a.__str__()} day {t+1}',
                                   name=f'{type(a).__name__}_expected_revenue_day_{t+1:02}',
                                   Show=False, path=f'{path}/roi/allroi')

                plot.confront_plot(x=r, y=roi, target=np.full((r.shape[0]), MIN_ROI),
                                   x_label='revenue', y_label='roi', label=f'estimated ROI {a.__str__()} day {t+1}',
                                   name=f'{type(a).__name__}_expected_vs_min_roi_day_{t+1:02}',
                                   target_label='ROI target',
                                   Show=False, path=f'{path}/roi')

                try:
                    idx_max_roi = np.nanargmax(roi)  # np.array(np.where(rev_mix_sum/cost_mix_sum > 0)).ravel()[0]
                except IndexError:
                    print('ehilauallua')
                    idx_max_roi = 1
                plot.confront_plot(x=r[idx_max_roi:], x_target=target_rev[idx_optimum_max:], y=roi[idx_max_roi:], target=target_roi[idx_optimum_max:],
                                   x_label='revenue', y_label='roi', label=f'estimated ROI {a.__str__()} day {t+1}',
                                   name=f'{type(a).__name__}_expected_roi_day_{t+1:02}',
                                   target_label='true ROI',
                                   Show=False, path=f'{path}/roi/roi_des')
                plot.confront_plot(x=r[idx_max_roi:], y=roi[idx_max_roi:], target=np.full((r[idx_max_roi:].shape[0]), MIN_ROI),
                                   x_label='revenue', y_label='roi', label=f'estimated ROI {a.__str__()} day {t+1}',
                                   name=f'{type(a).__name__}_expected_vs_min_roi_day_{t+1:02}',
                                   target_label='ROI target',
                                   Show=False, path=f'{path}/roi/roi_des')

            if t == T-1:
                for j, s in enumerate(subcampaigns):
                    if logger.isEnabledFor(logging.WARNING):
                        logger.error(f'{tabulate_module_summary(agents[0].subcampaigns[i].model_cost)}')
                        logger.error(f'{tabulate_module_summary(agents[0].subcampaigns[i].model_rev)}')


            if t > 330:  # 15:
                # data = np.hstack((np.array(X[i, :, :t+1]).T, np.array(Y_cost[i, :, :t+1]).T, np.array(Y_rev[i, :, :t+1]).T))
                # data = np.hstack((np.array(X[i, :, :t+1]).reshape((-1,1)), np.array(Y_cost[i, :, :t+1]).reshape((-1,1)), np.array(Y_rev[i, :, :t+1]).reshape((-1,1))))

                # data = np.array([np.hstack((np.array(X[i, j, :t+1]).T, np.array(Y_cost[i, j, :t+1]).T, np.array(Y_rev[i, j, :t+1]).T)) for j in range(len(subcampaigns))])
                data = np.array([np.hstack((np.array(X[i, j, :t+1]).reshape((-1,1)), np.array(Y_cost[i, j, :t+1]).reshape((-1,1)), np.array(Y_rev[i, j, :t+1]).reshape((-1,1)))) for j in range(len(subcampaigns))])
                crv_rev, crv_roi = crv.cross_validate(data)  # np.stack(X[i, :, :t+1], Y_cost[i, :, :t+1], Y_rev[i, :, :t+1]))
                if Plot:
                    plot.confront_plot(x=cost_array, y=np.where(np.isnan(crv_rev), 0., crv_rev), target=target_rev, x_label='cost', y_label='revenue', target_label='target revenue', label=f'{type(a).__name__}_crv_double_mean_est_day_nantozero{t+1:02}', Show=False, path=f'{path}/xv')
                    plot.confront_plot(x=cost_array, y=crv_rev, target=target_rev, x_label='cost', y_label='revenue', target_label='target revenue', label=f'{type(a).__name__}_crv_double_mean_est_day_{t+1:02}', Show=False, path=f'{path}/xv')
                # plot.confront_plot(x=cost_array, y=crv_bids_mean, target=target_rev, x_label='cost', y_label='revenue', target_label='target revenue', label=f'{type(a).__name__}_crv_Bids_double_mean_est_day_{t+1:02}', Show=False)

                    plot.triple_confront_plot(x=cost_array, y=crv_rev, y2=r, target=target_rev, x_label='cost', y_label='revenue', y2_label=f'{type(a).__name__}_expected_revenue_day_{t+1:02}', target_label='target revenue', label=f'{type(a).__name__} crvdouble mean estimator day {t+1}', name=f'triple_confront_plot_{type(a).__name__}_crv_double_mean_est_day_nantozero{t+1:02}', Show=False, path=f'{path}/xv')

                    plot.triple_confront_plot(x=cost_array, y=np.where(np.isnan(crv_rev), 0., crv_rev), y2=r, target=target_rev, x_label='cost', y_label='revenue', y2_label=f'{type(a).__name__}_expected_revenue_day_{t+1:02}', target_label='target revenue', label=f'{type(a).__name__} crvdouble mean estimator day {t+1}', name=f'triple_confront_plot_{type(a).__name__}_crv_double_mean_est_day_{t+1:02}', Show=False, path=f'{path}/xv')

                    idx_crv_roi = np.array(np.where(crv_roi > 0)).ravel()[0]
                    plot.triple_confront_plot(x=crv_rev[idx_crv_roi+1:], x2=r[idx_roi:], x_target=target_rev[idx:], y=crv_roi[idx_crv_roi:], y2=roi[idx_roi:], target=target_roi[idx:], x_label='revenue', y_label='roi', y2_label=f'{type(a).__name__}_expected_roi_day_{t+1:02}', target_label='target roi', label=f'{type(a).__name__} crvdouble mean roi estimator day {t+1}', name=f'triple_confront_plot_{type(a).__name__}_crv_double_mean_roi_est_day_{t+1:02}', Show=False, path=f'{path}/xv')

                    plot.triple_confront_plot(x=crv_rev, x2=r, x_target=target_rev, y=crv_roi, y2=roi, target=target_roi, x_label='revenue', y_label='roi', y2_label=f'{type(a).__name__}_expected_roi_day_{t+1:02}', target_label='target roi', label=f'{type(a).__name__} crvdouble mean roi estimator day {t+1}', name=f'triple_confront_plot_allroi{type(a).__name__}_crv_double_mean_roi_est_day_{t+1:02}', Show=False, path=f'{path}/xv')
            # logging.warning(f'X[{t}][{i}]: {X[t][i]}')

        #             plot.plot(bid, mean_cost)

        if Show_GP:
            for i, a in enumerate(agents):
                for j, s in enumerate(subcampaigns):
                    bid = np.linspace(0, MAX_BID, N_BID)
                    obs = X[i][j][:t+1]
                    # obs = np.array([X[x][i][j] for x in range(t+1)])
                    obs_cost = Y_cost[i][j][:t+1]
                    #obs_cost = np.array([Y_cost[x][i][j] for x in range(t+1)])
                    #obs_rev = np.array([Y_rev[x][i][j] for x in range(t+1)])
                    obs_rev = Y_rev[i][j][:t+1]
                    # obs_y = Y_cost[:t+1][i][j]
                    print(f'obs:\n{obs}')
                    #print(f'X[:][i][j]{X[:][i][j]}')
                    # print(f'true cost function:\n{s.cost(bid, noise=False)}')
                    (mean_cost, var_cost), (mean_rev, var_rev) = a.get_model(j,
                                                                         bid)
                    mean_gps_cost[i, j, t] = mean_cost.ravel()
                    mean_gps_rev[i, j, t] = mean_rev.ravel()
                    var_gps_cost[i, j, t] = var_cost.ravel()
                    var_gps_rev[i, j, t] = var_rev.ravel()

                    if Plot:
                        plot.plot_gp(x=bid, y=mean_cost, var_y=var_cost, true_y=s.cost(bid, noise=False),
                                     obs=obs, obs_y=obs_cost,
                                     y_label='cost',
                                     title=f'{a.__str__()} subcampaign {j} cost model day {t}',
                                     name=f'{type(a).__name__}_sc{j}_cost_day_{t:02}', path=f'{path}/gp')
                        plot.plot_gp(x=bid, y=mean_rev, var_y=var_rev, true_y=s.revenue(bid, noise=False),
                                     obs=obs, obs_y=obs_rev,
                                     y_label='rev',
                                     title=f'{a.__str__()} subcampaign {j} revenue model day {t}',
                                     name=f'{type(a).__name__}_sc{j}_rev_day{t:02}', path=f'{path}/gp')
                    # plot.plot(bid, mean_cost)

        # logger.warning(f'Y_rev:\n{Y_rev[:t+1]}')
        # logger.warning(f'Y_cost:\n{Y_cost[:t+1]}')
        # logger.warning(f'X:\n{X[:t+1]}')
        # logger.warning(f'exp_roi:\n{exp_roi[:t+1]}')
        # logger.warning(f'exp_rev:\n{exp_rev[:t+1]}')
        # logger.warning(f'exp_cost:\n{exp_cost[:t+1]}')

    # save run_data
    with open(f'{path}/run_data.pickle', 'wb') as f:
        pl.dump(T, f)
        pl.dump(len(subcampaigns), f)
        pl.dump([a.__str__() for a in agents], f)
        pl.dump(target_rev, f)
        pl.dump(target_roi, f)
        pl.dump(cost_array, f)
        pl.dump(bid_array, f)
        pl.dump(X, f)
        pl.dump(Y_cost, f)
        pl.dump(Y_rev, f)
        pl.dump(f_rev, f)
        pl.dump(f_cost, f)
        pl.dump(r_data, f)
        pl.dump(roi_data, f)
        pl.dump(exp_roi, f)
        pl.dump(exp_cost, f)
        pl.dump(exp_rev, f)
        pl.dump(real_sc_costs, f)
        pl.dump(real_sc_revs, f)
        pl.dump(mean_gps_cost, f)
        pl.dump(mean_gps_rev, f)
        pl.dump(var_gps_cost, f)
        pl.dump(var_gps_rev, f)
        pl.dump(MIN_ROI, f)

    return X, Y_cost, Y_rev, f_cost, f_rev, exp_roi, exp_cost, exp_rev


def show_results(subcampaigns, agents, X, Y_cost, Y_rev, f_cost, f_rev, exp_roi, exp_cost, exp_rev, path=None):
    TH = len(X[0, 0, :]) - 1 - INITIAL_BIDS
    if logger.isEnabledFor(logging.WARNING):
        logger.warning(f'olha aqui o {T}')
    total_rev = np.full((len(agents), TH), 0.0)
    total_cost = np.full((len(agents), TH), 0.0)
    total_roi = np.full((len(agents), TH), 0.0)
    print('agents', agents)
    c = agent.Clairvoyant_Agent(subcampaigns)
    X_opt = c.bid_choice()
    if logger.isEnabledFor(logging.WARNING):
        logger.warning(f'optimal bid_mix:\n{X_opt}')
    target_rev = sum([s.revenue(X_opt[i], noise=False) for i, s in enumerate(subcampaigns)])
    target_rev = np.full((TH), target_rev)
    target_cost = sum([s.cost(X_opt[i], noise=False) for i, s in enumerate(subcampaigns)])
    target_cost = np.full((TH), target_cost)
    clairvoyant_roi = np.full((TH), target_rev/target_cost)
    # for s in 
    for i, a in enumerate(agents):
        for t in range(TH):
            total_rev[i][t] = np.sum(f_rev[i, :, t+INITIAL_BIDS-1])
            total_cost[i][t] = np.sum(f_cost[i, :, t+INITIAL_BIDS-1])
            total_roi[i][t] = total_rev[i][t] / total_cost[i][t]

    max_rev = np.nanmax(total_rev) if np.nanmax(total_rev) > np.nanmax(target_rev) else np.nanmax(target_rev)
    max_cost = np.nanmax(total_cost) if np.nanmax(total_cost) > np.nanmax(target_cost) else np.nanmax(target_cost)
    max_roi = np.nanmax(total_roi) if np.nanmax(total_roi) > np.nanmax(clairvoyant_roi) else np.nanmax(clairvoyant_roi)

    for i, a in enumerate(agents):
        plot.confront_plot(x=np.linspace(0, TH, TH), y=total_rev[i, :], target=target_rev,
                           x_label='time', y_label='revenue',
                           target_label='revenue optimum', label=f'revenue {a.__str__()}',
                           name=f'{type(a).__name__}_revenue',  # f'{type(a).__name__}_expected_revenue_day_{t+1:02}',
                           Show=False, path=path, y_lim=(0, max_rev))
        plot.confront_plot(x=np.linspace(0, TH, TH), y=total_cost[i, :], target=target_cost,
                           x_label='time', y_label='cost', 
                           label=f'cost {a.__str__()}', name=f'{type(a).__name__}_cost', target_label='cost optimum',
                           Show=False, path=path)

        plot.triple_confront_plot(x=np.linspace(0, TH, TH), y=total_roi[i,:], y2=clairvoyant_roi, target=np.full((TH), MIN_ROI), 
                                  x_label='time', y_label='roi', y2_label=f'ROI optimum', target_label='ROI target',
                                  label=f'ROI {a.__str__()}', name=f'roi_evolution_{type(a).__name__}',
                                  Show=False, path=path, y_lim=(0, max_roi))

        plot.triple_confront_plot(x=np.linspace(0, TH, TH), y=total_cost[i,:], y2=target_cost, target=np.full((TH), MAX_EXP), 
                                  x_label='time', y_label='cost', y2_label=f'cost optimum', target_label='cost target',
                                  label=f'cost {a.__str__()}', name=f'cost_evolution_target{type(a).__name__}',
                                  Show=False, path=path, y_lim=(0, max_cost))

        #plot.plot(x=np.linspace(0, T, T), y=np.cumsum(total_rev[i, :]), x_label='time', y_label='cumulative revenue', label=f'{type(a).__name__}_cumulative_revenue', Show=False, path=path)
        #plot.plot(x=np.linspace(0, T, T), y=np.cumsum(total_cost[i, :]), x_label='time', y_label='cumulative cost', label=f'{type(a).__name__}cumulative_cost', Show=False, path=path)


if __name__ == '__main__':
    # num_threads = 1
    # tf.config.threading.set_inter_op_parallelism_threads(
    # num_threads
    # )

    start = time.time()
    print(tf.config.threading.get_inter_op_parallelism_threads())
    # tf.config.threading.set_intra_op_parallelism_threads(
    # num_threads
    # )
    print(tf.config.threading.get_intra_op_parallelism_threads())
    # logging.basicConfig(level=logging.INFO)
    # set up logging to file - see previous section for more details

    # logging.basicConfig(level=logging.ERROR,
    # #                     # format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    # #                     # datefmt='%m-%d %H:%M',
    #                      filename='/tmp/mygpapp.log',
    #                      filemode='w')

    # logging.getLogger('optimize').setLevel(logging.WARNING)
    # logging.getLogger('gp_model').setLevel(logging.INFO)
    #logging.getLogger('agent').setLevel(logging.WARNING)
    # logging.getLogger('agent').setLevel(logging.DEBUG)
    # logging.getLogger('cross_validation').setLevel(logging.DEBUG)


    # # A
    # campaign_a = [sc.Subcampaign(64, 0.3, 265, 0.2),
    #               sc.Subcampaign(70, 0.4, 355, 0.2),
    #               sc.Subcampaign(70, 0.3, 326, 0.1),
    #               sc.Subcampaign(50, 0.3, 185, 0.2),
    #               sc.Subcampaign(55, 0.4, 208, 0.1),
    #               sc.Subcampaign(120, 0.4, 724, 0.2),
    #               sc.Subcampaign(100, 0.3, 669, 0.25),
    #               sc.Subcampaign(90, 0.34, 595, 0.15),
    #               sc.Subcampaign(95, 0.38, 616, 0.19),
    #               sc.Subcampaign(110, 0.4, 675, 0.12)]

    # # # F
    # campaign_f = [sc.Subcampaign(160, 0.65, 847, .45, snr=50),
    #               sc.Subcampaign(170, .62, 895, 0.42, snr=50),
    #               sc.Subcampaign(170, 0.69, 886, 0.49, snr=50)] #,
    # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=25),
    #               sc.Subcampaign(77, .62, 565, 0.48, snr=25),
    #               sc.Subcampaign(75, .67, 573, 0.43, snr=25),
    #               sc.Subcampaign(65, .68, 503, 0.47, snr=25),
    #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=25)] #,
    # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=28),
    #               sc.Subcampaign(77, .62, 565, 0.48, snr=28),
    #               sc.Subcampaign(75, .67, 573, 0.43, snr=28),
    #               sc.Subcampaign(65, .68, 503, 0.47, snr=28),
    #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=28)] #,
    # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=29),
    #               sc.Subcampaign(77, .62, 565, 0.48, snr=29),
    #               sc.Subcampaign(75, .67, 573, 0.43, snr=29),
    #               sc.Subcampaign(65, .68, 503, 0.47, snr=29),
    #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=29)] #,

    # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=30),
    #               sc.Subcampaign(77, .62, 565, 0.48, snr=30),
    #               sc.Subcampaign(75, .67, 573, 0.43, snr=30),
    #               sc.Subcampaign(65, .68, 503, 0.47, snr=30),
    #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=30)] #,

    # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=35),
    #               sc.Subcampaign(77, .62, 565, 0.48, snr=35),
    #               sc.Subcampaign(75, .67, 573, 0.43, snr=35),
    #               sc.Subcampaign(65, .68, 503, 0.47, snr=35),
    #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=35)] #,

    # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=40),
    #               sc.Subcampaign(77, .62, 565, 0.48, snr=40),
    #               sc.Subcampaign(75, .67, 573, 0.43, snr=40),
    #               sc.Subcampaign(65, .68, 503, 0.47, snr=40),
    #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=40)] #,

    # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=45),
    #               sc.Subcampaign(77, .62, 565, 0.48, snr=45),
    #               sc.Subcampaign(75, .67, 573, 0.43, snr=45),
    #               sc.Subcampaign(65, .68, 503, 0.47, snr=45),
    #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=45)] #,

    # campaign_g = [sc.Subcampaign(60, 0.65, 497, .41, snr=50),
    #               sc.Subcampaign(77, .62, 565, 0.48, snr=50),
    #               sc.Subcampaign(75, .67, 573, 0.43, snr=50),
    #               sc.Subcampaign(65, .68, 503, 0.47, snr=50),
    #               sc.Subcampaign(70, 0.69, 536, 0.40, snr=50)] #,
    # campaign_f = [sc.Subcampaign(160, 0.65, 847, .45, snr=25),
    #               sc.Subcampaign(170, .62, 895, 0.42, snr=25),
    #               sc.Subcampaign(170, 0.69, 886, 0.49, snr=25)] #,

    # campaign_g = [sc.Subcampaign(160, 0.75, 847, .65),
    #               sc.Subcampaign(170, .72, 895, 0.62),
    #               sc.Subcampaign(170, 0.79, 886, 0.69),
    #               sc.Subcampaign(150, 0.87, 845, 0.67),
    #               sc.Subcampaign(155, 0.89, 848, 0.69)]
    #               # sc.Subcampaign(120, 0.4, 724, 0.2),
    #               # sc.Subcampaign(100, 0.3, 669, 0.25),
    #               # sc.Subcampaign(90, 0.34, 595, 0.15),
    #               # sc.Subcampaign(95, 0.38, 616, 0.19),
    #               # sc.Subcampaign(110, 0.4, 675, 0.12)]
    # campaign_g2 = [sc.Subcampaign(60, 0.45, 497, .31, snr=29),
    #               sc.Subcampaign(77, .52, 565, 0.38, snr=29),
    #               sc.Subcampaign(75, .47, 573, 0.35, snr=29),
    #               sc.Subcampaign(65, .58, 503, 0.43, snr=29),
    #               sc.Subcampaign(70, 0.59, 536, 0.38, snr=29)] #,

    # rnd_campaign_01 = [
    #             sc.Subcampaign(83.0, 0.9390275741377971, 530.0, 0.35653961965235176, snr=34),
    #             sc.Subcampaign(97.0, 0.8565310891376972, 417.0, 0.6893952328071604, snr=30),
    #             sc.Subcampaign(72.0, 0.4845001118567374, 548.0, 0.29997157236918714, snr=35),
    #             sc.Subcampaign(100.0, 0.6618767352251798, 571.0, 0.5709120360333635, snr=34),
    #             sc.Subcampaign(96.0, 0.24623017989617788, 550.0, 0.24553470046274206, snr=31)
    #             ]

    # rnd_campaign_02 = [
    #             sc.Subcampaign(83.0, 0.22405720652229322, 597.0, 0.2022227936494456, snr=30),
    #             sc.Subcampaign(98.0, 0.8499901193552486, 682.0, 0.5208800234776618, snr=31),
    #             sc.Subcampaign(56.0, 0.7267762997014247, 698.0, 0.3670485687338533, snr=31),
    #             sc.Subcampaign(60.0, 0.559426504706663, 456.0, 0.39340047292135694, snr=32),
    #             sc.Subcampaign(51.0, 0.7831724596643996, 444.0, 0.6895966545162417, snr=32)
    #             ]

    # rnd_campaign_03 = [
    #             sc.Subcampaign(97.0, 0.2254717323719329, 570.0, 0.21712317683804871, snr=30),
    #             sc.Subcampaign(78.0, 0.6809006218688995, 514.0, 0.638902808265124, snr=31),
    #             sc.Subcampaign(53.0, 1.0518056176361688, 426.0, 0.6940683038810456, snr=34),
    #             sc.Subcampaign(80.0, 0.4129454844360104, 469.0, 0.39195310889540247, snr=32),
    #             sc.Subcampaign(82.0, 0.9188239551808393, 548.0, 0.34500566310456804, snr=30)
    #             ]

    # rnd_campaign_04 = [
    #             sc.Subcampaign(62.0, 0.460443097667253, 487.0, 0.34862416791066253, snr=32),
    #             sc.Subcampaign(79.0, 1.0217507506678771, 494.0, 0.4244629979053927, snr=30),
    #             sc.Subcampaign(76.0, 0.5159200609033661, 467.0, 0.32645838278958483, snr=33),
    #             sc.Subcampaign(69.0, 0.8948304325148888, 684.0, 0.7226499482261013, snr=34),
    #             sc.Subcampaign(99.0, 1.0568307954024765, 494.0, 0.26571147363891007, snr=34)
    #             ]

    rnd_campaign_05 = [
                sc.Subcampaign(52.0, 0.7236376036521779, 525.0, 0.2581539194710056, snr=31),
                sc.Subcampaign(87.0, 0.8347532352531297, 643.0, 0.607729693610427, snr=35),
                sc.Subcampaign(68.0, 1.0545161819092699, 455.0, 0.39019889273979436, snr=33),
                sc.Subcampaign(99.0, 1.0715069147863183, 440.0, 0.7409946097511, snr=33),
                sc.Subcampaign(94.0, 0.9434443599935285, 600.0, 0.3880212001600485, snr=33)
                ]

    # rnd_campaign_06 = [
    #             sc.Subcampaign(71.0, 0.8750247925943013, 617.0, 0.8440732817094825, snr=31),
    #             sc.Subcampaign(53.0, 0.8411068818594563, 518.0, 0.6772664432914361, snr=33),
    #             sc.Subcampaign(87.0, 1.0703614987489847, 547.0, 0.8667712665875702, snr=32),
    #             sc.Subcampaign(98.0, 0.6310736950229558, 567.0, 0.2521616349302094, snr=31),
    #             sc.Subcampaign(59.0, 0.28867007783272997, 576.0, 0.24763783882935328, snr=32)
    #             ]

    # rnd_campaign_07 = [
    #             sc.Subcampaign(77.0, 0.8109246960855101, 409.0, 0.5079367092300731, snr=31),
    #             sc.Subcampaign(78.0, 0.2469886328248737, 592.0, 0.23086355578873, snr=31),
    #             sc.Subcampaign(91.0, 0.7744431310645019, 628.0, 0.5712510728701953, snr=35),
    #             sc.Subcampaign(50.0, 0.5162810922290018, 613.0, 0.359321181934275, snr=34),
    #             sc.Subcampaign(71.0, 0.3794531987004781, 513.0, 0.3077863622385426, snr=35)
    #             ]

    # rnd_campaign_08 = [
    #             sc.Subcampaign(67.0, 0.6715707414871959, 602.0, 0.3266023723522325, snr=30),
    #             sc.Subcampaign(80.0, 0.7751757702430537, 605.0, 0.2658241604853398, snr=32),
    #             sc.Subcampaign(99.0, 0.44063783263446604, 618.0, 0.2013497671860729, snr=31),
    #             sc.Subcampaign(77.0, 0.31016111269832836, 505.0, 0.21944833868391506, snr=31),
    #             sc.Subcampaign(99.0, 0.4059724156582363, 588.0, 0.2913060845388664, snr=32)
    #             ]
    # rnd_campaign_09 = [
    #             sc.Subcampaign(53.0, 0.6181790644241923, 486.0, 0.4181144656709901, snr=35),
    #             sc.Subcampaign(82.0, 0.8639181969833853, 684.0, 0.3300544004489604, snr=34),
    #             sc.Subcampaign(58.0, 0.6695803386210537, 547.0, 0.5291694460820753, snr=33),
    #             sc.Subcampaign(96.0, 0.8661364638643831, 419.0, 0.7291640886632436, snr=35),
    #             sc.Subcampaign(100.0, 0.8319412598597624, 453.0, 0.67917291744797, snr=30)
    #             ]
    # rnd_campaign_10 = [
    #             sc.Subcampaign(51.0, 1.049310026718121, 617.0, 0.2051759576328354, snr=30),
    #             sc.Subcampaign(86.0, 0.7797153486111319, 520.0, 0.539937043644524, snr=35),
    #             sc.Subcampaign(93.0, 0.23347185518099078, 422.0, 0.217994206233266, snr=30),
    #             sc.Subcampaign(61.0, 0.5780205490681166, 559.0, 0.49006423739237187, snr=30),
    #             sc.Subcampaign(84.0, 0.562291455390848, 457.0, 0.22447254161352354, snr=30)
    #             ]


    # all_camps = [rnd_campaign_01, rnd_campaign_02, rnd_campaign_03, rnd_campaign_04, rnd_campaign_05, rnd_campaign_06, rnd_campaign_07, rnd_campaign_08, rnd_campaign_09, rnd_campaign_10]
    # for i, c in enumerate(all_camps):
    #     print(f'random campaign {i+1:02d}\n')
    #     for j, s in enumerate(c):
    #         print(f'subcampaign {j}\n{s}')

    # import pprint
    # for i, c in enumerate(all_camps):
    #     pprint.pprint(f'random campaign {i+1:02d}')
    #     for j, s in enumerate(c):
    #         pprint.pprint(f'subcampaign {j}')
    #         pprint.pprint(f'{s}')


    campaign = rnd_campaign_05
    cost_array = np.linspace(0, MAX_EXP, N_COST)
    # c = agent.Clairvoyant_Agent(campaign_a)
    # bid_opt, target_rev, target_roi, target_rev_mix, target_cost_mix, idx_opt = c.bid_choice(ret_roi_revenue_cost=True)
    # # campaign_a = [sc.Subcampaign(59, 1.3, 63, 1.0),
    # #               sc.Subcampaign(20, 2.4, 30, 2.2),
    # #               sc.Subcampaign(20, 1.3, 35, 1.2)]
    # plot.plot(x=target_rev, y=target_roi, label="optimal roi")
    # plot.plot(x=cost_array, y=target_rev, label="optimal revenue")
    
    bid_array=x=np.linspace(0., MAX_BID, N_BID)
#     plot.plot(x=np.linspace(0., MAX_BID, N_BID), y=campaign_a[0].revenue(bid_array, noise=True))
#     plot.confront_plot(x=bid_array, y=campaign_a[1].revenue(bid_array, noise=True), target=campaign_a[1].revenue(bid_array, noise=False), x_label='bid', y_label='revenue', name='revenue_sc1')
# 
    #plot.plot_sc(x=bid_array, obs_y=campaign_a[1].revenue(bid_array, noise=True), y=campaign_a[1].revenue(bid_array, noise=False), x_label='bid', y_label='revenue', name='revenue_sc1')
    # print(load_run_results('aa8eda6d46bc4bb49c39e6b69eedbda3'))

    T = 61


    # a = agent.Agent(len(campaign))
    # b = agent.TS_Agent(len(campaign))
    # e = agent.TS_Conservative_Agent_1(len(campaign))
    # f = agent.TS_Conservative_Agent_5(len(campaign))
    # g = agent.TS_Conservative_Agent_10(len(campaign))
    # h = agent.Safe_Agent(len(campaign))
    # l = agent.Safe_Agent_rev(len(campaign))
    # m = agent.Safe_Agent_delta(len(campaign))
    # n = agent.Safe_eps_greedy_Agent(len(campaign))
    # p = agent.Safe_Agent_sample_cost(len(campaign))
    # q = agent.Safe_Agent_lb_cost(len(campaign))
    # rrr = agent.Safe_Opt_Agent(len(campaign))
    # su = agent.Safe_Opt_eps_greedy_Agent(len(campaign))



    # gcb = agent.GCB_Agent(len(campaign), T=T)
    # gcbsafe = agent.GCBsafe_Agent(len(campaign), T=T)
    # gcbsafe95 = agent.GCBsafe95_Agent(len(campaign), T=T, eps=0.95)
    # gcbsafe90 = agent.GCBsafe90_Agent(len(campaign), T=T, eps=0.90)

    # gcbsafe85 = agent.GCBsafe85_Agent(len(campaign), T=T, eps=0.85)
    # c = agent.TS_TFP_Agent(len(campaign_a))
    # d = agent.TS_MCMC_Agent(len(campaign_a))
   # c = agent.Clairvoyant_Agent(campaign_a)
    # agents = [b, e, f, g]  # , c, d]  # [a, b, c, d]
    #agents = [b, g, h]
    #agents = [h]
    #agents = [gcb, gcbsafe95]  # , rrr]  #rrr, su]


    # agents = [gcb, gcbsafe, gcbsafe95, gcbsafe90]  # , rrr]  #rrr, su]


    # agents = [gcbsafe, gcbsafe95, gcbsafe90, gcbsafe85]
    #agents = [n]
    # agents = [e]

    # for a in agents:
    #     with open(f'settings.pickle', 'wb') as f:
    #         pl.dump(f'{type(a).__name__}', f, -1)


    # al[1].bid_choice()
    start = time.time()
    #environment(campaign_a, agents)
    # load_run_config('28e59a0a981247ca9c395d88c16c5de8')
    #load_run_config('06c254e594c34571b0bb08066420ca6b')
    #load_run_config('8e1d3f46150f4c3b85e94325ac224623')
    #load_run_data('8e1d3f46150f4c3b85e94325ac224623/run_0')
    #load_run_data('06c254e594c34571b0bb08066420ca6b/run_0')
    # load_run_results('06c254e594c34571b0bb08066420ca6b')
    #load_run_results('ac0d538ae5f446fbb77237d418996238')
    # load_run_results('4e91d30b314d47a095768238ccf62edf')
    #load_run_results('4e0825094d104c46b97bcc71efb1dca8')
    #load_run_config('2cd7d38744b44e01a9a92f1b108dfa09')
    #load_run_results('65c4b362a023484daf86b67926695085')
    #analize_runs(multiple_runs(campaign_a, agents, 4))
    #X, Y_cost, Y_rev, f_cost, f_rev, exp_roi, exp_cost, exp_rev = environment(campaign_a, agents)

    logging.basicConfig(level=logging.ERROR,
                         # format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                         # datefmt='%m-%d %H:%M',
                         filename='/tmp/mygpapp.log',
                         filemode='w')
    #load_run_data('609851688bea4eaeabfb878d2c73ca08/run_0')
    # load_run_results('0ade4aa0dbfb4acaaf00eb35b736fea0')  # rnd01
    #load_run_results('e20e4c47b8f44fb7bceb0b7d987239e4')  # rnd02
    #load_run_results('f34a49ee3e9c4776b8c23336305caa67')  # rnd03
    # load_run_results('421cecdbb9f24a3fab4ec3a1d28d8fb2')  # rnd04
    # load_run_results('ee1dd32746da4fd88f8608784f4aad7f')  # rnd05
    load_run_results('61578a154b8d492ab14f71641b7c741b')  # rnd06
    #load_run_results('dd469e38801e4cf7b4cdf58544cfa6df')  # rnd07
    #load_run_results('101169181be54375a0e057900b487120')  # rnd08
    # load_run_results('ee2dd32746da4fd88f8608784f4aad7f')  # rnd09
    # load_run_results('7e4ce7fa51cb4d9eb28ee9cf30344d0e')  # rnd10
    #load_run_results('5cfb678860324ec299d5f51702827e47')  # rnd09
    #load_run_results('fcc353f6ead648dbb0a9afeea9f23271')  # #Experiment3
    #load_run_results('f1a8827cf06143a1baff3f1e236b85d5')
    # load_run_results('93055f79f5ab4105a15686e35554511e')
    #load_run_results('4f9bd644a5f44ff398d7a671fd755bf8')
    #load_run_results('b3c1ae98289b4ce298fad76f9820577c')
    #load_run_config('ef07ed99d78849a7b8c62c6465d5712b')
    #load_run_results('692a5191cdad414da551ac3858f13a76')  # Experiment1
    #load_run_results('07517bbd63424c28ac7dcb06a18c35db')  # Experiment2
    #load_run_data('0da07c997885493b9d297c28603f4621/run_0')
    #multiple_runs(campaign, agents, runs_num=5, T=41)
    #multiple_runs(campaign, agents, runs_num=45, T=T)

    # results = load_run_data('b440772d430a457b9ac3e4a93fb73e3b/run_0')
    # show_results(campaign, agents, *results)  # , path='b440772d430a457b9ac3e4a93fb73e3b/run_0')
    end = time.time()
    print('elapsed time:', end - start)
    if logger.isEnabledFor(logging.ERROR):
        logger.warning(f'elapsed time: {end - start}')
