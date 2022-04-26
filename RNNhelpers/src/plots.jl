module Plots

using CSV
using DataFrames
using PyPlot;
const plt = PyPlot
using ..Math
using ..IO

export init_rate_figure, update_rate_figure,
 init_plasticity_figure, update_plasticity_figure,
 init_current_figure, update_current_figure,
 init_IDom_figure, update_IDom_figure,
 init_raster_figure, update_raster_figure,
 init_output_figure, update_output_figure,
 init_example_unit_figure, update_example_unit_figure,
 init_weights_figure, update_weights_figure

##################
# Color defaults #
##################

c_inh = "blue"
c_exc = "red"
c_default = "black"
c_stimulus = "grey"
c_response = "green"
c_nonresponse = "purple"

######################
# Rendering defaults #
######################

DPI=150

#####################
# Plotting routines #
#####################

function init_rate_figure(p, dpi=DPI)
    fig, ax = plt.subplots(figsize=(2,2), dpi=dpi);
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Population Rates")
    fig.tight_layout();
    fig, ax
end

function update_rate_figure(fig, ax, p, trial, rates; save_fig=false, run="", save_dir=default_dir)
    ax.cla();
    ex_in_tr = pick_from_array(rates, 1)
    ex_non_tr = pick_from_array(rates, 2)
    ex_out_tr = pick_from_array(rates, 3)
    inh_tr = pick_from_array(rates, 4)
    ax.plot(1:trial, ex_in_tr, c=c_stimulus, label="Ex. In");
    ax.plot(1:trial, ex_non_tr, c=c_exc, linestyle = "--", label="Ex. Non");
    ax.plot(1:trial, ex_out_tr, c=c_exc, label="Ex. Out");
    ax.plot(1:trial, inh_tr, c=c_inh, label="Inh.");
    ax.legend(fontsize="x-small", frameon=false)
    ax.set_xlim(0, trial);
    ax.set_ylim(bottom=0);
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Population Rates")
    fig.tight_layout();
    if save_fig
        fig_path = save_dir * "/" * run * "/"
        fig.savefig(fig_path * "population rates.pdf")
        cols = (trial = 1:trial, ex_in=ex_in_tr, ex_non=ex_non_tr, ex_out=ex_out_tr, inh=inh_tr);
        CSV.write(fig_path * "population rates.csv", DataFrame(;cols...));
    end
end

function init_plasticity_figure(p, dpi=DPI)
    fig, ax = plt.subplots(figsize=(4,4), dpi=dpi);
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Total Accumulated Plasticity")
    ax.set_title("Plasticity Attribution")
    fig.tight_layout();
    fig, ax
end

function update_plasticity_figure(fig, ax, p, trial, plasticity; save_fig=false, run="", save_dir=default_dir)
    ax.cla();
    EE_A_tr = plasticity[1,:]
    EE_B_tr = plasticity[2,:]
    EE_β_tr = plasticity[3,:]
    EE_δ_tr = plasticity[4,:]
    IE_η_tr = plasticity[5,:]
    IE_α_tr = plasticity[6,:]
    IE_β_tr = plasticity[7,:]
    IE_δ_tr = plasticity[8,:]
    
    # For plots of change
    # ax.fill_between(1:trial, zeros(trial), EE_A_tr, facecolor=c_exc)
    # ax.fill_between(1:trial, EE_A_tr, EE_A_tr + IE_α_tr, facecolor=c_inh)
    # ax.fill_between(1:trial, EE_A_tr + IE_α_tr, EE_A_tr + IE_α_tr + EE_δ_tr, facecolor=c_exc)
    # ax.fill_between(1:trial, EE_A_tr + IE_α_tr + EE_δ_tr, EE_A_tr + IE_α_tr + EE_δ_tr + IE_β_tr, facecolor=c_inh)    
    # ax.fill_between(1:trial, zeros(trial), -IE_η_tr, facecolor=c_inh)
    # ax.fill_between(1:trial, -IE_η_tr, -IE_η_tr - EE_B_tr, facecolor=c_exc)
    # ax.fill_between(1:trial, -IE_η_tr - EE_B_tr, -IE_η_tr - EE_B_tr - IE_δ_tr, facecolor=c_inh)
    # ax.fill_between(1:trial, -IE_η_tr - EE_B_tr - IE_δ_tr, -IE_η_tr - EE_B_tr - IE_δ_tr - EE_β_tr, facecolor=c_inh)

    ax.plot(1:trial, EE_A_tr, c=c_exc, label="\$EE_A\$");
    ax.plot(1:trial, EE_B_tr, c=c_exc, linestyle = "--", label="\$EE_B\$");
    ax.plot(1:trial, EE_β_tr, c=c_exc, linestyle = "-.", label="\$EE_\\beta\$");
    ax.plot(1:trial, EE_δ_tr, c=c_exc, linestyle = ":", label="\$EE_\\delta\$");
    ax.plot(1:trial, -IE_η_tr, c=c_inh, label="\$IE_\\eta\$");
    ax.plot(1:trial, -IE_α_tr, c=c_inh, linestyle = "--", label="\$IE_\\alpha\$");
    ax.plot(1:trial, -IE_β_tr, c=c_inh, linestyle = "-.", label="\$IE_\\beta\$");
    ax.plot(1:trial, -IE_δ_tr, c=c_inh, linestyle = ":", label="\$IE_\\delta\$");
    ax.legend(fontsize="x-small", frameon=false)
    ax.set_xlim(0, trial);
    #ax.set_ylim(bottom=0);
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Total Accumulated Plasticity")
    ax.set_title("Plasticity Attribution")
    fig.tight_layout();
    if save_fig
        fig_path = save_dir * "/" * run * "/"
        fig.savefig(fig_path * "plasticity attribution.pdf")
        cols = (trial = 1:trial, EE_A = EE_A_tr, EE_B = EE_B_tr, EE_β = EE_β_tr, EE_δ = EE_δ_tr, IE_η = IE_η_tr, IE_α = IE_α_tr, IE_β = IE_β_tr, IE_δ = IE_δ_tr);
        CSV.write(fig_path * "plasticity attribution.csv", DataFrame(;cols...));
    end
end

function init_current_figure(p, dpi=DPI)
    fig, ax = plt.subplots(figsize=(2,2), dpi=dpi);
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Current")
    ax.set_title("External Currents")
    fig.tight_layout();
    fig, ax
end

function update_current_figure(fig, ax, p, trial, currentbias, currentfeedback; save_fig=false, run="", save_dir=default_dir)
    ax.cla();
    #Ibias = pick_from_array(rates, 1)
    ax.plot(1:trial, currentbias, c=c_inh, label="I_bias");
    ax.plot(1:trial, currentfeedback, c=c_response, label="|I_feedback|")
    ax.legend(fontsize="x-small", frameon=false)
    ax.set_xlim(0, trial);
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Current")
    ax.set_title("External Currents")
    fig.tight_layout();
    if save_fig
        fig_path = save_dir * "/" * run * "/"
        fig.savefig(fig_path * "current.pdf")
        cols = (trial = 1:trial, bias=currentbias, feedback=currentfeedback);
        CSV.write(fig_path * "current.csv", DataFrame(;cols...));
    end
end

function init_IDom_figure(p, dpi=DPI)
    fig, ax = plt.subplots(figsize=(2,2), dpi=dpi);
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Inhibitory Dominance")
    fig.tight_layout();
    fig, ax
end

function update_IDom_figure(fig, ax, p, trial, IDom; save_fig=false, run="", save_dir=default_dir)
    ax.cla();
    #Ibias = pick_from_array(rates, 1)
    ax.plot(1:trial, IDom, c=c_inh, label="Inhibitory dominance");
    #ax.legend(fontsize="x-small", frameon=false)
    ax.set_xlim(0, trial);
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Inhibitory Dominance")
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(0, max(1, y_max))
    fig.tight_layout();
    if save_fig
        fig_path = save_dir * "/" * run * "/"
        fig.savefig(fig_path * "inhibitory dominance.pdf")
        cols = (trial = 1:trial, bias=IDom);
        CSV.write(fig_path * "inhibitory dominance.csv", DataFrame(;cols...));
    end
end

function init_raster_figure(p, dpi=DPI)
    fig, ax = plt.subplots(figsize=(3,2), dpi=dpi);
    ax.set_ylim(0, p.N);
    ax.set_xlim(0, p.T);
    ax.axvspan(p.stim_offset, p.stim_offset + p.stim_dur, alpha=0.1, color=c_stimulus, lw=0)
    ax.axvspan(p.resp_offset, p.resp_offset + p.resp_dur, alpha=0.1, color=c_response, lw=0)
    ax.axhline(p.N_in, lw=0.2, c="black")
    ax.axhline(p.NE - p.N_out, lw=0.2, c="black")
    ax.axhline(p.NE, lw=0.2, c="black")
    ax.set_title("Trial # = ??");
    ax.set_ylabel("Neuron Index",fontsize=10);
    ax.set_xlabel("Time (ms)",fontsize=10);
    fig.tight_layout();
    fig, ax
end

function update_raster_figure(fig, ax, p, trial, freq, spike_times, spike_idx; save_fig=false, run="", save_dir=default_dir)
    ax.cla();
    ax.plot(spike_times, spike_idx,",k", ms=100);
    ax.set_ylim(0, p.N);
    ax.set_xlim(0, p.T);
    ax.axvspan(p.stim_offset, p.stim_offset + p.stim_dur, alpha=0.1, color=c_stimulus, lw=0)
    ax.axvspan(p.resp_offset, p.resp_offset + p.resp_dur, alpha=0.1, color=c_response, lw=0)
    ax.axhline(p.N_in, lw=0.2, c="black")
    ax.axhline(p.NE - p.N_out, lw=0.2, c="black")
    ax.axhline(p.NE, lw=0.2, c="black")
    ax.set_title("Trial #$trial, freq = $(2^freq) kHz");
    ax.set_ylabel("Neuron Index",fontsize=10);
    ax.set_xlabel("Time (ms)",fontsize=10);
    fig.tight_layout();
    if save_fig
        fig_path = save_dir * "/" * run * "/"
        fig.savefig(fig_path * "network activity.pdf")
        cols = (unit=spike_idx, spike_time=spike_times);
        CSV.write(fig_path * "network activity.csv", DataFrame(;cols...));

    end
end

function init_output_figure(p, dpi=DPI)
    fig, ax = plt.subplots(ncols=2, figsize=(4,2), dpi=dpi);
    ax[1].set_xlim([0, p.T])
    ax[1].set_ylim([-1, 1])
    ax[1].axvspan(p.stim_offset, p.stim_offset + p.stim_dur, alpha=0.1, color=c_stimulus, lw=0)
    ax[1].axvspan(p.resp_offset, p.resp_offset + p.resp_dur, alpha=0.1, color=c_response, lw=0)
    ax[1].set_title("Output")
    ax[2].set_title("Integrated Output")
    fig.tight_layout();
    fig, ax
end

function update_output_figure(fig, ax, p, trial, t_t, z_out_t, f_out_t, tar_tr, tar_out_tr, foil_tr, foil_out_tr; save_fig=false, run="", save_dir=default_dir)
    ax[1].cla();
    ax[1].set_xlim([0, p.T]);
    ax[1].set_ylim([-1, 1]);
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_title("Output")
    ax[1].axvspan(p.stim_offset, p.stim_offset + p.stim_dur, alpha=0.1, color=c_stimulus, lw=0)
    ax[1].axvspan(p.resp_offset, p.resp_offset + p.resp_dur, alpha=0.1, color=c_response, lw=0)
    ax[1].plot(t_t, z_out_t, c=c_default, linewidth=0.5, label="Network");
    ax[1].plot(t_t, f_out_t, c=c_response, label="Target");
    ax[1].legend(fontsize="x-small", frameon=false);

    ax[2].cla()
    ax[2].plot(tar_tr, tar_out_tr, c=c_response, label="Go")
    ax[2].plot(foil_tr, foil_out_tr, c=c_nonresponse, label="No-go")
    ax[2].set_xlim(0, trial);
    ax[2].set_xlabel("Trial #")
    ax[2].set_title("Integrated Output")
    ax[2].legend(fontsize="x-small", frameon=false)
    fig.tight_layout();
    if save_fig
        fig_path = save_dir * "/" * run * "/";
        fig.savefig(fig_path * "outputs.pdf");
        cols = (time = t_t, z_out=z_out_t, f_out=f_out_t);
        CSV.write(fig_path * "outputs 1.csv", DataFrame(;cols...));
        cols = (trial=tar_tr, target_out=tar_out_tr);
        CSV.write(fig_path * "outputs target.csv", DataFrame(;cols...));
        cols = (trial=foil_tr, foil_out=foil_out_tr);
        CSV.write(fig_path * "outputs foil.csv", DataFrame(;cols...));
        nothing
    end
end

function init_example_unit_figure(p, units, dpi=DPI)
    n = length(units)
    fig, ax = plt.subplots(ncols=n, figsize=(n*2.0,2), dpi=dpi)
    for (a_ax, unit) = zip(ax, units)
        a_ax.axvspan(p.stim_offset, p.stim_offset + p.stim_dur, alpha=0.1, color=c_stimulus, lw=0)
        a_ax.axvspan(p.resp_offset, p.resp_offset + p.resp_dur, alpha=0.1, color=c_response, lw=0)
        a_ax.set_xlabel("Time (ms)")
        a_ax.set_ylabel("Voltage (mV)")
        a_ax.set_title("Neuron #$unit")
    end
    fig.tight_layout();
    fig, ax
end

function update_example_unit_figure(fig, ax, p, units, t_t, I_t, I_rec_minus, I_rec_plus, V_t; I_scale = 1.0, save_fig=false, run="", save_dir=default_dir)
    if save_fig
        fig_path = save_dir * "/" * run * "/"
    end
    for (i, a_ax) = enumerate(ax)
        a_ax.cla();
        V_tt = pick_from_array(V_t, i)
        I_minus_t = pick_from_array(I_rec_minus, i)
        I_plus_t = pick_from_array(I_rec_plus, i)
        I_stimulus_t = pick_from_array(I_t, i)
        # I_max = max(I_plus_t, I_minus_t, abs.(I_stimulus_t))
        a_ax.plot(t_t, V_tt, c=c_default, lw=1.0);
        a_ax.plot(t_t, -I_scale .* I_minus_t, c=c_inh, linewidth=0.5);
        a_ax.plot(t_t, I_scale .* I_plus_t, c=c_exc, linewidth=0.5);
        a_ax.plot(t_t, I_scale .* I_stimulus_t, c=c_stimulus, linewidth=0.5);
        a_ax.axvspan(p.stim_offset, p.stim_offset + p.stim_dur, alpha=0.1, color=c_stimulus, lw=0)
        a_ax.axvspan(p.resp_offset, p.resp_offset + p.resp_dur, alpha=0.1, color=c_response, lw=0)
        a_ax.set_yticks([p.Vᵣ, p.V_th, 0])
        a_ax.set_yticklabels([p.Vᵣ, p.V_th, 0])
        a_ax.set_xlabel("Time (ms)")
        a_ax.set_ylabel("Voltage (mV)")
        a_ax.set_title("Unit #$(units[i])")
        if save_fig
            cols = (time = t_t, V=V_tt, I_minus=I_minus_t, I_plus=I_plus_t, I_stimulus=I_stimulus_t)
            CSV.write(fig_path * "example unit $(units[i]).csv", DataFrame(;cols...));
        end
    end
    fig.tight_layout();
    if save_fig
        fig.savefig(fig_path * "example units.pdf")
    end
end

function init_weights_figure(p, dpi=DPI)
    fig_rect = [0, 0, 1, 0.95]
    (fig, ax) = plt.subplots(nrows=1, ncols=3, figsize=(6,2), dpi=dpi)

    ax[1].set_title("Weight Matrix")
    ax[1].set_xlabel("From (InE, E, OutE, I)")
    ax[1].set_ylabel("To (InE, E, OutE, I)")
    ax[1].set_xlim((0, p.N))
    ax[1].set_ylim((0, p.N))
    ax[1].axhline(p.N_in, color="k", linewidth=.5)
    ax[1].axvline(p.N_in, color="k", linewidth=.5)
    ax[1].axhline(p.NE, color="k", linewidth=.5)
    ax[1].axvline(p.NE, color="k", linewidth=.5)
    ax[1].axhline(p.NE-p.N_out, color="k", linewidth=.5)
    ax[1].axvline(p.NE-p.N_out, color="k", linewidth=.5)

    ax[2].set_title("Recurrent W.")
    ax[2].set_xlabel("Trial #")
    ax[2].set_ylabel("Weights (au)")

    ax[3].set_title("Output W.")
    ax[3].set_xlabel("Trial #")
    ax[3].set_ylabel("Weights (au)")

    fig.tight_layout(pad=1.2)
    fig, ax
end

function update_weights_figure(fig, ax, p, trial, W, WEE_tr, WIE_tr, W_out_tr; save_fig=false, run="", save_dir=default_dir)
    ax[1].cla()
    ax[1].imshow(W, cmap="jet", vmin=0)
    ax[1].set_title("Weight Matrix")
    ax[1].set_xlabel("From (InE, E, OutE, I)")
    ax[1].set_ylabel("To (InE, E, OutE, I)")
    ax[1].set_xlim((0, p.N))
    ax[1].set_ylim((0, p.N))
    ax[1].axhline(p.N_in, color="k", linewidth=.5)
    ax[1].axvline(p.N_in, color="k", linewidth=.5)
    ax[1].axhline(p.NE, color="k", linewidth=.5)
    ax[1].axvline(p.NE, color="k", linewidth=.5)
    ax[1].axhline(p.NE-p.N_out, color="k", linewidth=.5)
    ax[1].axvline(p.NE-p.N_out, color="k", linewidth=.5)

    ax[2].cla()
    ax[2].set_title("Recurrent W.")
    ax[2].set_xlabel("Trial #")
    ax[2].set_ylabel("Weights (au)")
    ax[2].plot(1:trial, WEE_tr, color=c_exc, label="EE")
    ax[2].plot(1:trial, WIE_tr, color=c_inh, label="IE")
    ax[2].ticklabel_format(style="sci", scilimits=(0,0), useMathText=true, axis="y")
    ax[2].legend(fontsize="x-small", frameon=false)

    ax[3].cla()
    ax[3].set_title("Output W.")
    ax[3].set_xlabel("Trial #")
    ax[3].set_ylabel("Weights (au)")
    ax[3].plot(1:trial, W_out_tr, color=c_default)
    ax[3].ticklabel_format(style="sci", scilimits=(0,0), useMathText=true, axis="y")

    fig.tight_layout(pad=1.2)
    if save_fig
        fig_path = save_dir * "/" * run * "/"
        fig.savefig(fig_path * "weights.pdf")
        cols = (trial=1:trial, W_EE=WEE_tr, W_IE=WIE_tr, W_out=W_out_tr)
        CSV.write(fig_path * "weights.csv", DataFrame(;cols...));
    end
end

end
