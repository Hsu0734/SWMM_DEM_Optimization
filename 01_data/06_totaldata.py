'''
PySWMM Code Example
输出系统数据 SystemStat data
Author: Hanwen Xu
Version: 1
Date: Nov 01, 2023
'''

from pyswmm import Simulation, SystemStats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sim = Simulation("../GuanyaoshanNJ.inp")
system_routing = SystemStats(sim)
time_stamps = []
S1 = []
S2 = []
S3 = []
S4 = []
S5 = []
S6 = []

for step in sim:
    time_stamps.append(sim.current_time)
    runoff_stats = system_routing.runoff_stats

    infiltration = runoff_stats.__getitem__('infiltration')
    rainfall = runoff_stats.__getitem__('rainfall')
    runoff = runoff_stats.__getitem__('runoff')

    routing_stats = system_routing.routing_stats

    outflow = routing_stats.__getitem__('outflow')
    flooding = routing_stats.__getitem__('flooding')
    wet_weather_inflow = routing_stats.__getitem__('wet_weather_inflow')

    # print(system_routing.routing_stats)
    S1.append(infiltration)
    S2.append(rainfall)
    S3.append(runoff)
    S4.append(outflow)
    S5.append(flooding)
    S6.append(wet_weather_inflow)

print(max(S3))

# Drawing plots
fig = plt.figure(figsize=(12, 8))
fig.suptitle("System Stats of Infiltration, Rainfall, Runoff, Outflow, Flooding, and Wet Weather Inflow")
fig.set_tight_layout(True)

titles = ["Infiltration", "Rainfall", "Runoff", "Outflow", "Flooding", "wet_weather_inflow"]
y_labels = ["Infiltration (cm)", "Rainfall (mm)", "Runoff (mm)", "Outflow (m³)", "Flooding (m³)", "wet_weather_inflow (m³)"]
data = [S1, S2, S3, S4, S5, S6]

for i in range(6):
    subplot = plt.subplot(2, 3, i+1)
    subplot.plot(time_stamps, data[i], label=titles[i])
    subplot.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    subplot.set_ylabel(y_labels[i])
    subplot.set_xlabel("Time")
    subplot.grid("xy")
    subplot.tick_params(axis='x', rotation=30)
    subplot.legend()

plt.legend()
plt.show()
print(max(S3))