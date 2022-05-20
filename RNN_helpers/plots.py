import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import numpy as np
import scipy as sp
from scipy.stats import norm, gaussian_kde
from scipy.integrate import quad
from sklearn.neighbors import KernelDensity
from functools import reduce

from sympy import E
from .analyses import find_order, find_type
import itertools as it

R_COLOR = "#808080"
NNR_COLOR = '#FF767C'


def plot_violin_scatter(ax, x, data, highlights=[], highlight_marker="o", summary=True, bandwidth='scott', lim=None, end_domain=None, total=1.0, width = .8, size=10, color='k', summary_size=60, summary_color='k', alpha=.4, **kwargs):
    data = np.array(data)
    data = data[np.isfinite(data)]
    scale_factor = 1
    if lim is None:
        lim = (np.min(data), np.max(data))
    if end_domain is not None:
        assert np.all(end_domain >= data) or np.all(end_domain <= data)
        if end_domain < lim[1]:
            lim = (end_domain, lim[1])
        elif end_domain > lim[0]:
            lim = (lim[0], end_domain)
        mirror_data = -1 * (data - end_domain) + end_domain
        density_data = np.block([data, mirror_data]) 
        scale_factor = 2.0
    else:
        density_data = data.copy()
        scale_factor = 1.0

    f_density = gaussian_kde(density_data, bw_method=bandwidth)

    X_plot = scale_factor * f_density(data) * (np.random.rand(len(data)) -.5) * width + x
    
    ax.scatter(X_plot, data, s=size, c=color, **kwargs, zorder=0, alpha=alpha, lw=0)
    if len(highlights) > 0:
        ax.scatter(X_plot[highlights], data[highlights], s=size*2, color='k', marker=highlight_marker, facecolors='none')

    if summary:
        Q1, median, Q3 = (np.percentile(data, q) for q in [25, 50, 75])
        ax.scatter([x], [median], c='w', s=summary_size*1.3, zorder=1)
        ax.scatter([x], [median], c=summary_color, s=summary_size, zorder=2)
        ax.plot([x, x], [Q1, Q3], c='w', lw=np.sqrt(summary_size) * .5 * 1.3, zorder=1, solid_capstyle='round')
        ax.plot([x, x], [Q1, Q3], c=summary_color, lw=np.sqrt(summary_size)*.5, zorder=2, solid_capstyle='round')
    pass



def plot_raster(ax, spikes, trials, n_trials, trial_period=(-.2, .6), size=1, marker='o', color='k', alpha=1):
    duration_idx = (spikes/1000 >= trial_period[0]) & (spikes/1000 < trial_period[1]) 
    spikes = spikes[duration_idx]
    trials = trials[duration_idx].astype('int')
    
    ax.set_ylim(-1, n_trials + 1)
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    for i in range(n_trials):
        current_spike_times = spikes[trials == i] / 1000 
        ax.scatter(current_spike_times, len(current_spike_times)*[i], color=color, s=size, marker=marker, alpha=alpha, lw=0)
    pass 

def plot_PSTH(ax, spikes, trials, n_trials, trial_period=(-.2, 1.0), stim_period=(0, .1), plot_type='histogram', hist_bin=.02, smooth_window=np.array([]), bandwidth='scott', scotts_factor=1., show_variability=True, sem=False, avg_FR_color='k', var_FR_color=".7", var_FR_alpha=1., lw=.5, show_avg_FR=False, **kwargs):
    assert plot_type in ['histogram', 'kde', 'smooth']
    trial_duration = trial_period[1] - trial_period[0]
    duration_idx = (spikes >= trial_period[0]*1000) & (spikes < trial_period[1]*1000-1) 
    spikes = spikes[duration_idx]
    trials = trials[duration_idx].astype('int')
    average_FR = len(spikes) / (trial_duration * n_trials)
    if plot_type == 'kde':
        # spikes = spikes/1000
        # # flatten spikes
        # centered_spikes = spikes - trial_period[0]
        # kde_spikes = np.block([-centered_spikes, centered_spikes, -centered_spikes + 2*trial_duration]) + trial_period[0]
        # fr_density = gaussian_kde(kde_spikes, bw_method=bandwidth)
        # def fr(t):
        #     return(fr_density(t) * 3 * trial_duration * average_FR)
        # X_plot = np.linspace(0, trial_duration, 1000)
        # ax.plot(X_plot - stim_period[0], [fr(x) for x in X_plot], **kwargs)
        if bandwidth == 'scott':
            bandwidth = scotts_factor * 1.06 * np.std(spikes) * len(spikes)**(-1./5)
        smooth_window = [np.exp(-x**2/(2*(bandwidth**2))) for x in np.arange(-1000, 1000, 1)]
        plot_type = 'smooth'
    if plot_type == 'smooth':
        smooth_window = np.array(smooth_window) / np.sum(smooth_window)
        smooth_window = smooth_window[np.newaxis, :]
        binary_windows = np.arange(trial_period[0], trial_period[1], 0.001)
        binary_activity = np.zeros((n_trials, len(binary_windows)))
        rounded_spikes = np.round(spikes - trial_period[0]*1000).astype('int')
        for t, st in zip(trials, rounded_spikes):
            binary_activity[t, st] = 1000
        binary_activity = np.hstack([np.flip(binary_activity, axis=1), binary_activity, np.flip(binary_activity, axis=1)])
        binary_windows = np.arange(trial_period[0] - trial_duration, trial_period[1] + trial_duration, 0.001)
        rates = sp.signal.fftconvolve(binary_activity, smooth_window, mode='same', axes=1)
        avg_FR = rates.mean(axis=0)
        ax.plot(binary_windows, avg_FR, color=avg_FR_color, lw=lw)
        if show_variability:
            var_FR = rates.std(axis=0)
            if sem:
                var_FR /= np.sqrt(n_trials)
            ax.fill_between(binary_windows, avg_FR - var_FR, avg_FR + var_FR, lw=0, color=var_FR_color, alpha=var_FR_alpha)
    elif plot_type == 'histogram':
        spikes = spikes/1000
        bins = np.arange(trial_period[0], trial_period[1] + hist_bin, hist_bin)
        if show_variability:
            rates = np.zeros((n_trials, len(bins) - 1))
            for i in range(n_trials):
                for j, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
                    rate = np.sum((trials == i) & (spikes >= start) & (spikes < end)) / hist_bin
                    rates[i, j] = rate
            avg_FR = rates.mean(axis=0)
            var_FR = rates.std(axis=0)
            if sem:
                var_FR /=  np.sqrt(n_trials)
            ax.step(bins[:-1], avg_FR, where='post', color=avg_FR_color, lw=lw)
            ax.bar(bins[:-1], 2*var_FR, bottom = avg_FR - var_FR, width=hist_bin, align='edge', color=var_FR_color, alpha=var_FR_alpha, lw=0)
        else:
            # flatten spikes
            bars, bins = np.histogram(spikes, normed=True, bins=bins)
            avg_FR = bars * trial_duration * average_FR
            ax.bar(bins[:-1], avg_FR, width=hist_bin, align='edge', lw=0, color=var_FR_color)
    if show_avg_FR:
        ax.axhline(average_FR)
    # if stim_period is not None:
    #     # ax.axvspan(0, stim_period[1], color=stim_color, alpha=stim_alpha, lw=0)
    #     y_min = ax.get_ylim()[0]
    #     ax.plot([0, stim_period[1]], [y_min, y_min], lw=2)

    ax.set_xlim(trial_period[0], trial_period[1])
    pass

def plot_raster_PSTH(spikes, trials, n_trials, figsize=(2,2), gridspec_kw={}, trial_period=(-.2, .6), stim_period=(0, .1), plot_type='kde', hist_bin=.05, smooth_window=np.array([]), bandwidth='scott', scotts_factor=1.0, show_variability=True, sem=True, avg_FR_color='k', var_FR_color=".7",  var_FR_alpha=1, lw=.5, show_avg_FR=False, raster_size=1, raster_marker='o', raster_color='k', raster_alpha=1, show_stim=True, show_response=False, response_time=.6, y_lim=5, **kwargs):
    
    gridspec_kw_defaults = {'bottom': .2, 'left': .2, 'hspace':.1, 'height_ratios': [.5, .5]}
    gridspec_kw_defaults.update(gridspec_kw)

    fig, (ax_raster, ax) = plt.subplots(nrows=2, sharex=True, figsize=figsize, gridspec_kw=gridspec_kw_defaults)

    plot_PSTH(ax, spikes, trials, n_trials, trial_period=trial_period, stim_period=stim_period, plot_type=plot_type, hist_bin=hist_bin, smooth_window=smooth_window, bandwidth=bandwidth, scotts_factor=scotts_factor, show_variability=show_variability, sem=sem, avg_FR_color=avg_FR_color, var_FR_color=var_FR_color, var_FR_alpha=var_FR_alpha, lw=lw, show_avg_FR=show_avg_FR, **kwargs)
    
    plot_raster(ax_raster, spikes, trials, n_trials, trial_period=trial_period, size=raster_size, marker=raster_marker, color=raster_color, alpha=raster_alpha)

    ax.set_xlim(trial_period)
    ax.set_xticks(np.arange(trial_period[0], trial_period[1] + 0.1, .2))

    current_y_lim = ax.get_ylim()[1]
    if current_y_lim < y_lim:
        current_y_lim = y_lim
    ax.set_ylim(0, current_y_lim)
    
    # ax.set_title(f"{date}, {animal}, {cell_num}: {R_st:.2}, {R_ch:.2}, {R:.2}", size='x-small')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate\n(spikes/s)")

    if show_response:
        ax.axvline(response_time, linestyle=":", color='green')
    if show_stim:
        ax.plot([stim_period[0], stim_period[1]], [current_y_lim, current_y_lim], color='k', lw=2)
    return fig, ax, ax_raster

def add_axis_size(fig, ax_w, ax_h, left, bottom):
    fig_w, fig_h = fig.get_size_inches()
    ax = fig.add_axes([left/fig_w, bottom/fig_h, ax_w/fig_w, ax_h/fig_h])
    return ax

def make_axis_size(ax_w, ax_h, left=.3, bottom=.3, right=0, top=0):
    fig_w = (ax_w + left + right) * 1.05
    fig_h = (ax_h + bottom + top) * 1.05
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = add_axis_size(fig, ax_w, ax_h, left, bottom)
    return fig, ax

def plot_performance_curve(ax, performance_curve, **kwargs):
    stims = np.array(list(performance_curve.keys()))
    vals = np.array(list(performance_curve.values()))
    
    ind = np.argsort(stims)
    stims = stims[ind]
    vals = vals[ind]
    
    ax.plot(stims, vals, color='.3', **kwargs)
    _, max_val = ax.get_ylim()
    ax.set_ylim([0, max_val])
    pass

def plot_raster_2(responses, nosepokes=None, color='k', marker='o', num_trials=None, guides=True, draw_hist=True, stim_highlight=[100, 200], beginning_time=0, ending_time=300, bin_width=10, bw=20, num_labels=2., rate_max=None, width_per_sec=.01, trial_per_in=200, lw=1, size=1):
    if num_trials is None:
        num_trials = len(responses)
    num_trials_rate = len(responses)

    fig_h = .3 + float(num_trials) / trial_per_in
    if draw_hist:
        fig_h += .5
    fig_w = width_per_sec*(ending_time - beginning_time) + .4
    fig = plt.figure(figsize=(fig_w, fig_h))

    if draw_hist:
        raster_rect = [.5 / fig_w, .8 / fig_h, (fig_w - .7) / fig_w, (fig_h - 0.9) / fig_h]
        gap_rect = [.5 / fig_w, .7 / fig_h, (fig_w - .7) / fig_w, .1 / fig_h]
        raster_hist_rect = [.5 / fig_w, .3 / fig_h, (fig_w - .7) / fig_w, 0.4 / fig_h]
    else:
        raster_rect = [.3 / fig_w, .2 / fig_h, (fig_w - .4) / fig_w, (fig_h - .3) / fig_h]
    ax = plt.axes(raster_rect)
    if draw_hist:
        ax_gap = plt.axes(gap_rect)
        ax1 = plt.axes(raster_hist_rect)

    for i in range(num_trials):
        ax.scatter(responses[i], [i + .5] * len(responses[i]), marker="o", s=size , color=color, edgecolor='none')
        if nosepokes is not None:
            ax.scatter(nosepokes[i], [i + .5], marker=marker, s=size*5, color='k')
    ax.set_xlim([beginning_time, ending_time])
    ax.set_ylim([0, num_trials])
    # ax.set_ylabel('Trial #')

    steps = int(np.floor(num_trials / (10. * num_labels)) * 10.)
    ax.set_yticks(range(0, num_trials + 1, steps))
    if guides:
        for guide in np.arange(0, num_trials, 10):
            ax.axhline(guide, color='k', alpha=.1, linewidth=.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    if stim_highlight is not None:
        ax.axvspan(stim_highlight[0], stim_highlight[1], facecolor='k', alpha=.2)

    if draw_hist:
        ax.set_xticks([])
        ax.spines['bottom'].set_color('none')
        total_responses = np.array([item for subarray in responses for item in subarray])
        total_responses = total_responses[(total_responses > beginning_time)*(total_responses < ending_time)]
        diff = ending_time - beginning_time
        response_copy = np.hstack((total_responses, -total_responses + 2*beginning_time + 2*diff, -total_responses + 2*beginning_time))
        if bw == None:
            kde = gaussian_kde(response_copy)
        else:
            kde = gaussian_kde(response_copy, bw_method=bw/1000)            

        average_rate = 1000 * len(total_responses) / (num_trials_rate * (ending_time - beginning_time))
        pre_factor = 1000 * len(total_responses) / (num_trials_rate * quad(lambda x: kde([x])[0], beginning_time, ending_time)[0])
        firing_rate = lambda x: pre_factor * kde([x])[0]
        t = sp.linspace(beginning_time, ending_time, 200)
        ax1.set_xlim([beginning_time, ending_time])
        rates = np.array([firing_rate(i) for i in t])
        ax1.plot(t, rates, color=color, linewidth=lw)
        cur_ylim = ax1.get_ylim();
        
        if rate_max != None:
            ax1.set_ylim([0, rate_max])
        #ax1.set_ylim([0, np.max(rates)*1.2])
        #ax1.set_yticks(np.arange(0,  np.max(rates)*1.2, int(average_rate)))
        ax_gap.set_xlim([beginning_time, ending_time])
        ax_gap.spines['top'].set_color('none')
        ax_gap.spines['bottom'].set_color('none')
        ax_gap.spines['right'].set_color('none')
        ax_gap.spines['left'].set_color('none')
        ax_gap.set_xticks([])
        ax_gap.set_yticks([])
        if stim_highlight is not None:
            ax1.axvspan(stim_highlight[0], stim_highlight[1], facecolor='k', alpha=.2)
            ax_gap.axvspan(stim_highlight[0], stim_highlight[1], facecolor='k', alpha=.2)

        bins = np.arange(beginning_time, ending_time + bin_width, bin_width)
        num_bins = len(bins) - 1
        counts, bin_edges = np.histogram(total_responses, bins=bins)
        counts = 1000 * np.array(counts) / (num_trials_rate * bin_width)
        ax1.bar(bin_edges[1:] - bin_width/2., counts, bin_width, color=color, edgecolor=color, linewidth=.5, alpha=.2)
        ax1.spines['top'].set_color('none')
        ax1.spines['right'].set_color('none')
        ax1.xaxis.set_ticks_position('bottom')


def colormap_figsize(df, x_name, y_name, n_conds, scale=4):
    num_X = float(len(set(df[x_name])))
    num_Y = float(len(set(df[y_name])))
    return (n_conds*scale*(num_X/num_Y)*1.1, scale)


def plot_colormap(ax, df, x_name, y_name, z_name, cmap='jet', percent=False, rescale=None, total=0, **imshow_kwargs):
    X = list(set(df[x_name]))
    X.sort()
    Y = list(set(df[y_name]))
    Y.sort()
    ZZ = np.zeros((len(Y), len(X)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            value = df[(df[x_name] == x)&(df[y_name] == y)][z_name].mean()
            if percent: 
                ZZ[j, i] = value / total * 100.0
            else:
                ZZ[j, i] = value         
    XX, YY = np.meshgrid(X, Y)
    if rescale == None:
        img = ax.imshow(ZZ, origin="lower", cmap=cmap, **imshow_kwargs)
    else:
        img = ax.imshow(ZZ, origin="lower", cmap=cmap, norm=pltc.DivergingNorm(vmin=rescale[0], vcenter=rescale[1], vmax=rescale[2]), **imshow_kwargs)

    plt.colorbar(img, ax=ax)
    ax.set_xticks(range(0, len(X)))
    ax.set_yticks(range(0, len(Y)))
    ax.set_xticklabels(X)
    ax.set_yticklabels(Y)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    
def makeCDF(ax, values, xmin=0, xmax=1, **kwargs):
    xaxis = np.sort(values)
    xaxis = (np.insert(xaxis,0,xmin))
    xaxis = np.insert(xaxis, len(xaxis), xmax)
    yaxis = np.transpose([np.linspace(0,1,num=len(xaxis),endpoint=True)])
    ax.step(xaxis, yaxis, **kwargs)
    return xaxis, yaxis

def create_distribution_summary_figure(figsize=(4,2), show_dead=False, left=.12):
    fig = plt.figure(figsize=figsize)
    if show_dead:
        gs = plt.GridSpec(2, 2, height_ratios=[.15,.85], width_ratios=[.1, .9], hspace=0, wspace=0.3, bottom=.2, left=left, right=.94, top=.98)
        ax_dead = fig.add_subplot(gs[1, 0])
        ax_summary = fig.add_subplot(gs[0, 1])
        ax_distribution = fig.add_subplot(gs[1, 1], sharex=ax_summary)
        ax_dead.set_ylabel("% inactive")
        ax_dead.set_xticks([])
        ax_dead.set_ylim([0, 0.3])
        ax_dead.set_yticks([0, .1, .2, .3])
        ax_dead.set_yticklabels(["0%", "10%", "20%", "30%"])
    else:
        gs = plt.GridSpec(2, 1, height_ratios=[.15,.85], hspace=0, bottom=.2, left=left, right=.94, top=.98)
        ax_summary = fig.add_subplot(gs[0])
        ax_distribution = fig.add_subplot(gs[1], sharex=ax_summary)
    ax_distribution.set_ylabel("Probability density")
    ax_summary.set_yticks([])
    ax_summary.spines['bottom'].set_visible(False)
    ax_summary.xaxis.set_visible(False)
    if show_dead:
        return fig, ax_distribution, ax_summary, ax_dead
    else:
        return fig, ax_distribution, ax_summary

def add_reference_line(ax, location):
    ax.axvline(location, color='k', lw=.75)
    pass

def add_NCR_CR_scale(ax, method='one_sided', height=0.1, hoffset=0.5, voffset=.05, fontsize=10, alength=.5, awidth=.05, aoffset=0.1):
    x_min, x_max = ax.get_xlim()    
    y_min, y_max = ax.get_ylim()
    if method == 'one_sided':
        # ax.add_patch(plt.Polygon([[x_min, y_min - height], [x_min, y_min], [x_max, y_min - height]], lw=0, facecolor=NNR_COLOR))
        # ax.add_patch(plt.Polygon([[x_min, y_min], [x_max, y_min], [x_max, y_min - height]], lw=0, facecolor=R_COLOR))
        ax.add_patch(plt.Arrow(x_min + hoffset + alength + aoffset, y_min - height + voffset, -alength, 0, facecolor=NNR_COLOR, lw=0, width=awidth))
        ax.add_patch(plt.Arrow(x_max - hoffset - alength - aoffset, y_min - height + voffset, alength, 0, facecolor=R_COLOR, lw=0, width=awidth))
        ax.text(x_min + hoffset, y_min - height + voffset, "NCR", ha='right', va='center', c=NNR_COLOR, fontsize=fontsize)
        ax.text(x_max - hoffset, y_min - height + voffset, "CR", ha='left', va='center', c=R_COLOR, fontsize=fontsize)
    elif method == 'two_sided':
        ax.add_patch(plt.Polygon([[0, y_min - height], [0, y_min], [x_max, y_min - height]], lw=0, facecolor=NNR_COLOR))
        ax.add_patch(plt.Polygon([[0, y_min - height], [0, y_min], [x_min, y_min - height]], lw=0, facecolor=NNR_COLOR))
        
        ax.add_patch(plt.Polygon([[0, y_min], [x_max, y_min], [x_max, y_min - height]], lw=0, facecolor=R_COLOR))
        ax.add_patch(plt.Polygon([[0, y_min], [x_min, y_min], [x_min, y_min - height]], lw=0, facecolor=R_COLOR))

        ax.text(x_min - hoffset, y_min - height - voffset, "-CR", ha='left', va='bottom', c=R_COLOR, fontsize=fontsize)
        ax.text(x_max + hoffset, y_min - height - voffset, "+CR", ha='left', va='bottom', c=R_COLOR, fontsize=fontsize)
        ax.text(0, y_min - height - 10*voffset, "NCR", ha='center', va='top', c=NNR_COLOR, fontsize=fontsize)
    ax.set_ylim([y_min-height, y_max])
    pass

def plot_summary(ax, data, i, linecolor='k', facecolor='.5', s=200, lw=2):
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    q1 = np.percentile(data, 25)
    ax.plot([i,i], [q1, q3], linewidth=2, solid_capstyle='round', color=linecolor, zorder=10)
    ax.scatter(i, median, color=facecolor, edgecolor=linecolor, s=s, lw=lw, zorder=11)

def plot_summary_horiz(data, num, ax_s, lim=None, label=None, color='blue', linestyle='-'):
    data = np.array(data)
    median = np.percentile(data, 50)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    ax_s.plot([Q1, Q3], [num, num], c=color, lw=1.25, ls=linestyle, solid_capstyle='round')
    # ax_s.plot([median, median], [num - 0.2, num + .2], c=color, label=label)
    ax_s.scatter([median], [num], c=color, label=label, lw=0, s=10)
    pass
    

def plot_bar(frac, num, ax_z, color='blue'):
    ax_z.bar(num, frac, color=color, lw=0)
    pass

def plot_pdf(data, ax_d, lim=None, end_domain=None, label=None, color='blue', linestyle='-', bandwidth='scott', scotts_factor=1.0, lw=1.0, total=1.0, **kwargs):
    data = np.array(data)
    if bandwidth == 'scott':
        bandwidth = scotts_factor * 1.06 * np.std(data) * len(data)**(-1/5.)
    if lim is None:
        lim = (np.min(data), np.max(data))
    if end_domain is not None:
        assert np.all(end_domain >= data) or np.all(end_domain <= data)
        if end_domain < lim[1]:
            lim = (end_domain, lim[1])
        elif end_domain > lim[0]:
            lim = (lim[0], end_domain)
        mirror_data = -1 * (data - end_domain) + end_domain
        data = np.block([data, mirror_data]) 
        total *= 2.0
        
    # X_plot = np.linspace(lim[0], lim[1], 1000)[:, np.newaxis]
    # data = data[:, np.newaxis]
    # density = np.exp(KernelDensity(bandwidth=bandwidth, **kwargs).fit(data).score_samples(X_plot))

    X_plot = np.linspace(lim[0], lim[1], 1000)
    f_density = gaussian_kde(data[np.isfinite(data)], bw_method=bandwidth)
    density = f_density(X_plot)
    ax_d.plot(X_plot, density * total, c=color, label=label, ls=linestyle, lw=lw)
    ax_d.set_xlim(lim)
    pass


def plot_pdf_and_summary(data, num, ax_d, ax_s, ax_dead=None, n_dead=None, color='blue', linestyle='-', lim=None,  end_domain=None, label=None, bandwidth=.2, scotts_factor=1.0, **kwargs):
    data = np.array(data)
    plot_summary_horiz(data, num, ax_s, lim=lim, label=label, color=color, linestyle=linestyle)
    if n_dead is not None:
        n_data = len(data)
        f_dead = n_dead / (n_dead + n_data) 
        plot_pdf(data, ax_d, lim=lim, end_domain=end_domain, label=label, color=color, bandwidth=bandwidth, scotts_factor=scotts_factor, linestyle=linestyle, total=1.-f_dead, **kwargs)
        if ax_dead is not None:
            plot_bar(f_dead, num, ax_dead, color=color)
    else:
        plot_pdf(data, ax_d, lim=lim, end_domain=end_domain, label=label, color=color, bandwidth=bandwidth, linestyle=linestyle, total=1., **kwargs)
    ax_s.set_ylim([-.5, num+.5])
    pass

def add_summary_sig(pair, x, level, ax):
    ax.plot([x, x], pair, lw=.5, c='.6')
    ax.plot([x, x-.1], [pair[0], pair[0]], lw=.5, c='.6')
    ax.plot([x, x-.1], [pair[1], pair[1]], lw=.5, c='.6')
    ax.text(x+.1, (pair[0] + pair[1]) / 2.0 - 0.5, level, ha='left', va='center', c='.6', rotation=0)
    pass

# For motifs 
R = .02

def draw_node(ax, coord, label="", color='k', scale=1.0, fontsize=8, bold=False):
    ax.add_patch(mpl.patches.Circle(coord, radius=.02*scale, fill=bold, facecolor=color, edgecolor=color))
    ax.text(coord[0], coord[1], label, ha='center', va='center', fontsize=fontsize)
    pass

def draw_arrow(ax, coord1, coord2, scale=1):
    radius = 0.02*scale
    dx = coord2[0] - coord1[0]
    dy = coord2[1] - coord1[1]
    th = np.arctan(dy/dx)
    if dx < 0:
        th = th + np.pi 
     
    ax.add_patch(mpl.patches.FancyArrow(coord1[0] + radius*np.cos(th), coord1[1] + radius*np.sin(th), dx - 2*radius*np.cos(th), dy - 2*radius*np.sin(th), lw=0, facecolor='k', width=0.001*scale, head_width=.01*scale, length_includes_head=True))
    pass

def find_ch_locations(n, scale=1.0, rotation=0, start_at='first'):
    node_x_locations = np.arange(n)*scale*0.1
    node_locations = np.array(list(zip(node_x_locations, n*[0])))  
    if start_at == 'first':
        pass
    elif start_at == 'last':
        for i in range(node_locations.shape[0]):
            node_locations[i,:] -= node_locations[-1, :]
    th = rotation * np.pi / 180
    rot_mat = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    node_locations = node_locations @ rot_mat.T
    return node_locations

def shift_locations(nls, center_coord):
    total = np.vstack(nls)
    shift = center_coord - total.mean(axis=0)
    for nl in nls:
        for i in range(nl.shape[0]):
            nl[i, :] += shift
    return nls

def draw_ch(ax, n, center_coord, label_1="", label_2="", color_1='k', color_2='k', scale=1.0, rotation=0, include_first=True, fontsize=8, bold_1=False, bold_2=False):
    n += 1
    nl = find_ch_locations(n, scale=scale, rotation=rotation)
    nl = shift_locations([nl], center_coord)[0]
    for i, loc in enumerate(nl):
        if i == 0:
            if include_first:
                draw_node(ax, loc, label=label_1, color=color_1, bold=bold_1, scale=scale, fontsize=fontsize)
        elif i == n-1:
            draw_node(ax, loc, label=label_2, color=color_2, scale=scale, bold=bold_2,fontsize=fontsize)
            draw_arrow(ax, nl[i-1], loc, scale=scale)
        else:
            draw_node(ax, loc, scale=scale, fontsize=fontsize)
            draw_arrow(ax, nl[i-1], loc, scale=scale)
    pass

def draw_di(ax, n, m, center_coord, label_1="", label_2="", color_1='k', color_2='k', bold_1=False, bold_2=False, scale=1.0, angle=15, rotation=0, fontsize=8):
    n += 1
    m += 1
    nl1 = find_ch_locations(n, scale=scale, rotation=rotation-angle)
    nl2 = find_ch_locations(m, scale=scale, rotation=rotation+angle)
    nl1, nl2 = shift_locations([nl1, nl2], center_coord)
    for i, loc in enumerate(nl1):
        if i == 0:
            draw_node(ax, loc, scale=scale, fontsize=fontsize)
        elif i == n-1:
            draw_node(ax, loc, label=label_1, bold=bold_1, color=color_1, scale=scale, fontsize=fontsize)
            draw_arrow(ax, nl1[i-1], loc, scale=scale)
        else:
            draw_node(ax, loc, scale=scale, fontsize=fontsize)
            draw_arrow(ax, nl1[i-1], loc, scale=scale)
    for i, loc in enumerate(nl2):
        if i == 0:
            continue
        if i == m-1:
            draw_node(ax, loc, label=label_2, bold=bold_2, color=color_2, scale=scale, fontsize=fontsize)
            draw_arrow(ax, nl2[i-1], loc, scale=scale)
        else:
            draw_node(ax, loc, scale=scale, fontsize=fontsize)
            draw_arrow(ax, nl2[i-1], loc, scale=scale)
    pass


def draw_co(ax, n, m, center_coord, label_1="", label_2="", color_1='k', color_2='k', scale=1.0, angle=15, rotation=0, bold_1=False, bold_2=False, fontsize=8):
    n += 1
    m += 1
    nl1 = find_ch_locations(n, scale=scale, rotation=rotation+angle, start_at='last')
    nl2 = find_ch_locations(m, scale=scale, rotation=rotation-angle, start_at='last')
    nl1, nl2 = shift_locations([nl1, nl2], center_coord)
    for i, loc in enumerate(nl1):
        if i == 0:
            draw_node(ax, loc, label=label_1, bold=bold_1, color=color_1, scale=scale, fontsize=fontsize)
        elif i == n-1:
            draw_node(ax, loc, scale=scale, fontsize=fontsize)
            draw_arrow(ax, nl1[i-1], loc, scale=scale)
        else:
            draw_node(ax, loc, scale=scale)
            draw_arrow(ax, nl1[i-1], loc, scale=scale)
    for i, loc in enumerate(nl2):
        if i == 0:
            draw_node(ax, loc, label=label_2, bold=bold_2, color=color_2, scale=scale, fontsize=fontsize)
        elif i == m-1:
            draw_arrow(ax, nl2[i-1], loc, scale=scale)
        else:
            draw_node(ax, loc, scale=scale)
            draw_arrow(ax, nl2[i-1], loc, scale=scale)
    pass

def plot_motif(ax, name, x, y, label_2 = '', color_2='red', scale=1.0, fontsize=8):
    name_list = name.split('_')
    pop = name_list[0]
    nums = name_list[-1]
    if ('tr' in name_list[1]):
        motif_type = name_list[1] + name_list[2]
    else:
        motif_type = name_list[1]
    if pop == 'inT':
        label_1 = "T"
        color_1 = "red"
    elif pop == 'inF':
        label_1 = "NT"
        color_1 = "red"
    elif pop == 'out':
        label_1 = ''
        color_1 = 'red'
    elif pop == "I":
        label_1 = ''
        color_1 = 'blue'
    if 'ch' == motif_type:
        draw_ch(ax, int(nums), (x, y), color_1=color_1, label_1=label_1, color_2=color_2, label_2=label_2, rotation=0, scale=scale, fontsize=fontsize, bold_2=True)
    if 'chR' == motif_type:
        draw_ch(ax, int(nums), (x, y), color_1=color_2, label_1=label_2, color_2=color_1, label_2=label_1, rotation=0, scale=scale, fontsize=fontsize, bold_1=True)
    elif 'di' == motif_type:
        draw_di(ax, int(nums[0]), int(nums[1]), (x, y), color_1=color_1, label_1=label_1, color_2=color_2, label_2=label_2, scale=scale, fontsize=fontsize, bold_2=True)
    elif 'co' == motif_type:
        draw_co(ax, int(nums[0]), int(nums[1]), (x, y), color_1=color_1, label_1=label_1, color_2=color_2, label_2=label_2, scale=scale, fontsize=fontsize, bold_2=True)
    elif 'tr2' == motif_type[:3]:
        if 'ch' == motif_type[-2:]:
            draw_ch(ax, int(nums), (x, y), color_1=color_2, label_1=label_2, color_2=color_2, label_2=label_2, rotation=0, scale=scale, fontsize=fontsize, bold_1=True, bold_2=True)
        elif 'di' == motif_type[-2:]:
            draw_di(ax, int(nums[0]), int(nums[1]), (x, y), color_1=color_2, label_1=label_2, color_2=color_2, label_2=label_2, scale=scale, fontsize=fontsize, bold_1=True, bold_2=True)
        elif 'co' == motif_type[-2:]:
            draw_co(ax, int(nums[0]), int(nums[1]), (x, y), color_1=color_2, label_1=label_2, color_2=color_2, label_2=label_2, scale=scale, fontsize=fontsize, bold_1=True, bold_2=True)
    pass

# def calculate_responsiveness_predictions(model, df_out, p_thresh=0.0001, eta="sum", max_order=2, conds=CONDS, keys=KEYS):
#     df_new = df_out.copy()
#     sig_names = []
#     orders = []

#     for current_order in range(1, max_order+1):
#         for param in dict(model.params).items():
#             name, value = param
#             p = model.pvalues[name]
#             if (eta is True):
#                 if (':' in name):
#                     name = name.split(':')[1]
#                 else:
#                     continue
#             elif (eta is False) and (':' in name):
#                 continue
#             elif (eta == "sum"):
#                 if ':' in name:
#                     name = name.split(':')[1]
#             nums = name.split('_')[-1]
#             order = np.sum([int(n) for n in nums])
#             if (p < p_thresh) and (order==current_order) and (name not in sig_names):
#                 sig_names.append(name)
#                 orders.append(order)
#         n_cumulants = len(sig_names)

#     for name in sig_names:
#         df_new[f"{name}_contrib"] = np.nan
#         for j, co in enumerate(conds):
#             idx = reduce(lambda x,y: x&y, (df_new[l] == m for l,m in zip(keys, co)))
#             df_c = df_new[idx]
#             if eta is True:
#                 df_new.loc[idx, f"{name}_contrib"] = df_c[name] * model.params["eta:" + name]
#             elif eta is False:
#                 df_new.loc[idx, f"{name}_contrib"] = df_c[name] * model.params[name]
#             elif eta == "sum":
#                 df_new.loc[idx, f"{name}_contrib"] = df_c[name] * model.params[name] + df_c[name] * model.params["eta:" + name] * df_c["eta"]
        
#         for current_order in range(1, max_order+1):
#             df_new[f"net_{current_order}"] = 0.0
#             for j, co in enumerate(conds):
#                 idx = reduce(lambda x,y: x&y, (df_new[l] == m for l,m in zip(keys, co)))
#                 df_c = df_new[idx]

#                 for name, order in zip(sig_names, orders):
#                     if order != current_order:
#                         continue
#                     if eta is True:
#                         df_new.loc[idx, f"net_{current_order}"] += df_c[name] * model.params["eta:" + name]
#                     elif eta is False:
#                         df_new.loc[idx, f"net_{current_order}"] += df_c[name] * model.params[name]
#                     elif eta == "sum":
#                         df_new.loc[idx, f"net_{current_order}"] += df_c[name] * model.params[name] + df_c[name] * model.params["eta:" + name] * df_c["eta"]
        
#         df_new[f"net"] = 0.0
#         for col in df_new.columns:
#             if "net_" == col[:4]:
#                 df_new["net"] += df_new[col]

#     return n_cumulants, sig_names, df_new

KEYS = ["IE_stdp", "EE_stdp"]
CONDS = [(False, False), (True, False), (False, True), (True, True)]
COLORS = [plt.cm.Purples(i/5.) for i in range(2,6)]

def plot_motif_contributions(df, prefix = "s_con", current_order=1, xlim=[-4,4], sc=10, fs=5, conds=CONDS, keys=KEYS, labels=None, colors=COLORS, fill=True, show_net=False, sortby="direction", show_only=None, width_ratio=3, plot_style="bar", markersize=1.5):
    assert sortby in ["population", "direction", None]
    assert plot_style in ["bar", "scatter"]
    results = {}
    if labels is None:
        labels = conds

    if prefix == "":
        names = [name for name in df.columns if (name[:1] != "_") and (name[:5] not in  ["s_con", "c_con", "r_con"]) and find_order(name) == current_order]
    else:
        names = [name[len(prefix):] for name in df.columns if (prefix == name[:len(prefix)]) and find_order(name) == current_order and (name[len(prefix):] != f"net_{current_order}")]
    
    # restrict to certain subtypes
    if show_only is not None:
        names = [name for name in names if find_type(name).split('_')[1] in show_only]

    if sortby is not None:
        type_order = {"ch": 3, "di": 2, "chR": 1, "co": 0}
        pop_order = {"inT": 0, "inF": 1, "out": 2, "I": 3}
        if sortby == "population":
            total_order = {pop + "_" + typ: 10*i + j for (pop, i), (typ, j) in it.product(pop_order.items(), type_order.items())}
        elif sortby == "direction":
            total_order = {pop + "_" + typ: 10*j + i for (pop, i), (typ, j) in it.product(pop_order.items(), type_order.items())}
        total_order["net"] = 99
        order_func = lambda x: total_order[find_type(x)]
        names.sort(key=order_func)

    n_cumulants = len(names)
    n_cond = len(conds)

    fig, (ax_m, ax_b) = plt.subplots(figsize=(1 + width_ratio, n_cumulants/3+1), ncols=2, gridspec_kw={'width_ratios': [1,  width_ratio], 'wspace':0}, sharey=False)

    ax_m.set_aspect('equal')
    ax_m.set_xlim([0,2])
    if show_net:
        ax_m.set_ylim([-1, n_cumulants])
    else:
        ax_m.set_ylim([0, n_cumulants])
    ax_m.set_axis_off()
    #ax_m.set_yticks(range(n_cumulants+1))

    ax_b.set_xlim(xlim)
    if show_net:
        ax_b.spines['bottom'].set_position(('data', -1))
        ax_b.set_ylim([-1, n_cumulants])
    else:
        ax_b.spines['bottom'].set_position(('data', 0.0))
        ax_b.set_ylim([0, n_cumulants])
    ax_b.spines['left'].set_position(('data', 0))
    ax_b.set_yticks([])
    #ax_b.axvline(0, c='0.0', lw=.75)

    i = 1
    for name in names:
        results[name] = {}
        if name[0:3] == "net":
            continue
        plot_motif(ax_m, name, 1, i - 0.5, scale=sc, fontsize=fs)
        for j, (co, la) in enumerate(zip(conds, labels)):
            idx = reduce(lambda x,y: x&y, (df[l] == m for l,m in zip(keys, co)))
            data = df[idx][prefix + name]
            results[name][la] = data
            y = i + j/(n_cond+1) - (n_cond-1)/(2*(n_cond+1)) - 0.5
            median = np.percentile(data, 50)
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            if plot_style == "bar":
                ax_b.barh([y], [median], height=1.0/(n_cond+1), align='center', color=colors[j], lw=0, fill=fill, zorder=0)
                ax_b.plot([Q1, Q3], [y,y], color='white', lw=1, zorder=2)
                ax_b.plot([Q1, Q3], [y,y], color=colors[j], lw=.5, zorder=3)
            elif plot_style == "scatter":
                ax_b.scatter([median], [y], color=colors[j], s=(3*markersize)**2, zorder=3)
                ax_b.plot([0, median], [y,y], color=colors[j], lw=.5, linestyle='--', zorder=2)
                ax_b.plot([Q1, Q3], [y,y], color=colors[j], lw=markersize, zorder=2, solid_capstyle='round')
        i += 1


    if show_net:
        ax_m.text(1, 0, "net", va='center', ha='center')
        for j, co in enumerate(conds):
            idx = reduce(lambda x,y: x&y, (df[l] == m for l,m in zip(keys, co)))
            data = df[idx][f"{prefix}net_{current_order}"]
            y = j/(n_cond+1) - n_cond/(2*(n_cond+1)) - 0.5
            median = np.percentile(data, 50)
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            if plot_style == "bar":
                ax_b.barh([y], [median], color=colors[j], lw=0, fill=fill, height=1.0/(n_cond+1), align='center', zorder=0)
                ax_b.plot([Q1, Q3], [y,y], color='white', lw=1, zorder=1)
                ax_b.plot([Q1, Q3], [y,y], color=colors[j], lw=.5, zorder=2)
            elif plot_style == "scatter":
                ax_b.scatter([median], [y], color=colors[j], s=(3*markersize)**2, zorder=3)
                ax_b.plot([0, median], [y,y], color=colors[j], lw=.5, linestyle='--', zorder=2)
                ax_b.plot([Q1, Q3], [y,y], color=colors[j], lw=markersize, zorder=2, solid_capstyle='round')


    return fig, (ax_m, ax_b), results