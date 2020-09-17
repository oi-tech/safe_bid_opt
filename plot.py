import numpy as np
import matplotlib.pyplot as plt
import pickle as pl
import subcampaign as sc
#import seaborn as sns
import tikzplotlib
#sns.set()


def plot(x, y, x_label=None, y_label=None, label=None, name=None,
         Label=True, Save=True, Show=True, path=None, title=None,
         x_lim=None, y_lim=None):
    fig_handle = plt.figure()
    plt.plot(x, y, label=label if Label else None)  # , label= f'ROI optimum')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    if Label and label:
        plt.gca().legend()
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    if Save:
        name = f'{name if name != None else label}'
        plots_path = f'{path+"/plots" if path != None else "plots"}'
        pickle_path = f'{path+"/pickle" if path != None else "pickle"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight')  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
    if Show:
        plt.show()
    plt.close()

def confront_plot(x, y, target, x_target=None, x_label=None, y_label=None, target_label=None, label=None,
                  name=None, Label=True, Save=True, Show=False, path=None, y_lim=None, x_lim=None, title=None):
    fig_handle = plt.figure()
    x = x.ravel()
    y = y.ravel()
    target = target.ravel()
    x_target = x if x_target is None else x_target.ravel()

    plt.plot(x, y,label=label if Label else None)  # , label= f'ROI optimum')
    plt.plot(x_target, target, label=target_label if Label else None)  # , label= f'ROI optimum')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if title:
        plt.title(title)
    if Label and label:
        plt.gca().legend()
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    if Save:
        name = f'{name if name != None else label}'
        plots_path = f'{path+"/plots" if path != None else "plots"}'
        pickle_path = f'{path+"/pickle" if path != None else "pickle"}'
        tikz_path = f'{path+"/tikz" if path != None else "tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")

    if Show:
        plt.show()

    plt.close()

def triple_confront_plot(x, y, y2, target, x2=None, x_target=None, x_label=None, y_label=None, y2_label=None,
                         target_label=None, label=None, name=None, Label=True, Save=True, Show=False, path=None,
                         x_lim=None, y_lim=None, title=None):
    fig_handle = plt.figure()
    x = x.ravel()
    y = y.ravel()
    y2 = y2.ravel()
    target = target.ravel()
    x_target = x if x_target is None else x_target.ravel()
    x2 = x if x2 is None else x2.ravel()

    plt.plot(x, y,label=label if Label else None)  # , label= f'ROI optimum')
    plt.plot(x_target, target, label=target_label if Label else None)  # , label= f'ROI optimum')
    plt.plot(x2, y2, label=y2_label if Label else None)  # , label= f'ROI optimum')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.title(title)
    if title:
        plt.title(title)
    if Label and label:
        plt.gca().legend()
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    if Save:
        name = f'{name if name != None else label}'
        plots_path = f'{path+"/plots" if path != None else "plots"}'
        pickle_path = f'{path+"/pickle" if path != None else "pickle"}'
        tikz_path = f'{path+"/tikz" if path != None else "tikz"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        tikzplotlib.save(f'{tikz_path}/{name}.tex')  # "mytikz.tex")
    if Show:
        plt.show()

    plt.close()


def plot_gp(x, y, var_y, true_y, obs, obs_y,
            x_label='bid', y_label=None, label='Prediction',
            name=None, title=None,
            Label=True, Save=True, Show=False, path=None, y_lim=None):
    x = x.ravel()
    y = y.ravel()
    var_y = var_y.ravel()
    true_y = true_y.ravel()
    obs = obs.ravel()
    obs_y = obs_y.ravel()

    fig_handle = plt.figure()
    sigma = np.sqrt(var_y)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, true_y, 'r:', label=r'Target function')
    plt.plot(obs, obs_y, 'r.', markersize=10, label=f'{len(obs)} Observations')
    plt.plot(x, y, 'b-', label=label)  # subcampaign {i+1}, day {t}')
    plt.fill_between(x, y + sigma*1.96, y - sigma*1.96, alpha=0.2)
    plt.title(title)
    plt.xlim((-0.1, 2.))

    if Label and label:
        plt.gca().legend()
    if y_lim:
        plt.ylim(y_lim)
    if Save:
        name = f'{name if name != None else label}'
        plots_path = f'{path+"/plots" if path != None else "plots"}'
        pickle_path = f'{path+"/pickle" if path != None else "pickle"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
    if Show:
        plt.show()

    plt.close()

def plot_sc(x, y, obs_y,
            x_label='bid', y_label=None, label='True function',
            name=None, title=None,
            Label=True, Save=True, Show=False, path=None, y_lim=None):
    x = x.ravel()
    # print(f'x: \n{x}')
    y = y.ravel()
    # print(f'y: \n{y}')
    #var_y = var_y.ravel()
    # print(f'var_y: \n{var_y}')
    #true_y = true_y.ravel()
    # print(f'true_y: \n{true_y}')
    obs = x  #obs.ravel()
    # print(f'obs: \n{obs}')
    obs_y = obs_y.ravel()
    # print(f'obs_y: \n{obs_y}')

    fig_handle = plt.figure()
    # plt.figure()
    #sigma = np.sqrt(var_y)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.plot(x, true_y, 'r:', label=r'Target function')
    plt.plot(obs, obs_y, 'r.', markersize=10, label=f'Noise')
    plt.plot(x, y, 'b-', label=label)  # subcampaign {i+1}, day {t}')
    # plt.fill_between(x, y + sigma*1.96, y - sigma*1.96, alpha=0.2)
    plt.title(title)
    plt.xlim((-0.1, 2.))

    if Label and label:
        plt.gca().legend()
    if y_lim:
        plt.ylim(y_lim)
    if Save:
        name = f'{name if name != None else label}'
        plots_path = f'{path+"/plots" if path != None else "plots"}'
        pickle_path = f'{path+"/pickle" if path != None else "pickle"}'
        plt.savefig(f'{plots_path}/{name}.svg', bbox_inches='tight', dpi=216)  # f'plots/{name}.svg')
        pl.dump(fig_handle, open(f'{pickle_path}/{name}.pickle', 'wb'))
        # plt.savefig(f'plots/{name}.svg')
        # pl.dump(fig_handle, open(f'pickle/{name}.pickle', 'wb'))
    if Show:
        plt.show()

    plt.close()


def load(name):
    # Load figure from disk and display
    fig_handle = pl.load(open(f'{name}.pickle', 'rb'))
    plt.show()


if __name__ == '__main__':
    x = np.linspace(0.0, 2.0, 100)
   # plot(x, y, label='oi') #, x_label='bid')  # , y_label='cost')  # , Label=False, xy_Label=False)  # , Label=True)
    #c = sc.Subcampaign(70, 0.3, 78, 0.4)
    #y = c.revenue(x, noise=True)
    #plot(x, y, label='oi__oi_revenue', Save=True)
    # y2 = c.cost(x, noise=False)
    #target = c.revenue(x, noise=False)
    # plot(x, y, label='oi_cost')
    #confront_plot(x, y, target, label='ciccia', x_label='woow', y_label='ciao', Save=True, Show=True)
    # plot(x[1:], y[1:])

    # idx = np.array(np.where(y/y2>0.0)).ravel()[0]  # take the last not NaN element
    # plot(x[idx:], y/y2[idx:])

    load('cumulative_roi_free exploration agent')  #confront_cumulative_revenue')
    #load('optimal_roi_2Mbids_15kcosts')
