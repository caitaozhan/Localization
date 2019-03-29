import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

start_logNFFT = 8
end_logNFFT = 15

sensor_configs = range(start_logNFFT, end_logNFFT + 1)
#sensor_configs = [2 ** config for config in sensor_configs]
energy_cost = np.array([2.9935, 2.9999, 3.0657, 3.2532, 3.5475, 4.0937, 5.1648, 7.6977])
energy_cost_max = np.max(energy_cost)
energy_cost = np.array([energy/energy_cost_max for energy in energy_cost])

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
#def_figsize = 56, 20
#rcParams['figure.figsize'] = 28, 20

plt.style.use('seaborn-dark')
sns.set(context='paper', font_scale=2)
sns.set_style("whitegrid")
rcParams['lines.linewidth'] = 5

# SMALL_SIZE = 0
# MEDIUM_SIZE = 40
# BIGGER_SIZE = 40
#
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title
fig, ax = plt.subplots()
plt.xlabel('Sensor Configuration')
plt.ylabel('Normalized Cost')
plt.plot(sensor_configs, energy_cost)
plt.xlim([sensor_configs[0], sensor_configs[-1]])
plt.ylim([0.3, 1])
ax.set_xticklabels(('256', '512', '1K', '2K', '4K', '8K', '16K', '32K'))
plt.savefig('energy.pdf')