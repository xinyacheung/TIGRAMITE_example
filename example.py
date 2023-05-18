import numpy as np
import matplotlib.pyplot as plt
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import parcorr, gpdc, cmiknn

def dyn(x1,x2,x3,x4):
    x1 = -0.287*x2 + np.random.normal()
    x2 = 0.4*x2 + 0.287*x1+ np.random.normal()
    x3 = 0.9*x3 + np.random.normal()
    x4 = 0.9 * x2 + np.random.normal()
    return x1,x2,x3,x4


x1_ini,x2_ini,x3_ini,x4_ini = np.random.rand(4,1) #initial state
timestep = 200
data = np.expand_dims(np.concatenate((x1_ini,x2_ini,x3_ini,x4_ini ), axis=0),axis=0) #shape = [1,4]
for step in range(timestep): # run dynamics
    x1_ini,x2_ini,x3_ini,x4_ini = dyn(x1_ini,x2_ini,x3_ini,x4_ini)
    temp = np.expand_dims(np.concatenate((x1_ini,x2_ini,x3_ini,x4_ini), axis=0),axis=0)
    data = np.concatenate((data, temp), axis=0) #shape = [timestep+1,4]
print(data.shape)
var_names = ['x1', 'x2', 'x3', 'x4']
dataframe = pp.DataFrame(data,
                         datatime=np.arange(len(data)),
                         var_names=var_names)
tp.plot_timeseries(dataframe,label_fontsize=20,
                   tick_label_size=10)
plt.savefig(f'timeseries.pdf', dpi=500, bbox_inches='tight')

ParCorr = parcorr.ParCorr(significance='analytic')
pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=ParCorr,
    verbosity=1)

correlations = pcmci.get_lagged_dependencies(tau_max=20, val_only=True)['val_matrix']
lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names':var_names,
                                    'x_base':5, 'y_base':.5})
plt.savefig(f'get_lagged_dependencies.pdf',dpi=500,bbox_inches='tight')

pcmci.verbosity = 1
tau_max=2
results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None)
q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=tau_max, fdr_method='fdr_bh')

tp.plot_graph(graph = results['graph'],
    val_matrix=results['val_matrix'],
    var_names=var_names,
    link_colorbar_label='MCI',
    node_colorbar_label='auto-MCI',
    link_label_fontsize=20,
    label_fontsize=20,
    tick_label_size=14,
    node_label_size=20,
    edge_ticks=0.5,
    node_ticks=0.5,
    node_size=0.5
    )
plt.savefig(f'causal_graph.pdf',dpi=500,bbox_inches='tight')

tp.plot_time_series_graph(
    figsize=(4, 4),
    val_matrix=results['val_matrix'],
    graph = results['graph'],
    var_names=var_names,
    link_colorbar_label='MCI',
    label_fontsize=20,
    tick_label_size=20
)
plt.savefig(f'time-lagged_graph.pdf',dpi=500,bbox_inches='tight')
