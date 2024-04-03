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
sim.start()
system_routing = SystemStats(sim)
time_stamps = []
S1 = []
S2 = []
S3 = []
S4 = []
S5 = []
S6 = []

# Initialize previous values for computing differences
prev_infiltration = 0
prev_rainfall = 0
prev_runoff = 0
prev_outflow = 0
prev_flooding = 0
prev_wet_weather_inflow = 0

for step in sim:
    time_stamps.append(sim.current_time)
    runoff_stats = system_routing.runoff_stats

    # Compute the differences for the current step
    current_infiltration = runoff_stats.__getitem__('infiltration')
    current_rainfall = runoff_stats.__getitem__('rainfall')
    current_runoff = runoff_stats.__getitem__('runoff')

    infiltration_diff = current_infiltration - prev_infiltration
    rainfall_diff = current_rainfall - prev_rainfall
    runoff_diff = current_runoff - prev_runoff

    # Update the previous values
    prev_infiltration = current_infiltration
    prev_rainfall = current_rainfall
    prev_runoff = current_runoff

    S1.append(infiltration_diff)
    S2.append(rainfall_diff)
    S3.append(runoff_diff)

    routing_stats = system_routing.routing_stats

    current_outflow = routing_stats.__getitem__('outflow')
    current_flooding = routing_stats.__getitem__('flooding')
    current_wet_weather_inflow = routing_stats.__getitem__('wet_weather_inflow')

    outflow_diff = current_outflow - prev_outflow
    flooding_diff = current_flooding - prev_flooding
    wet_weather_inflow_diff = current_wet_weather_inflow - prev_wet_weather_inflow

    # Update the previous values for routing stats
    prev_outflow = current_outflow
    prev_flooding = current_flooding
    prev_wet_weather_inflow = current_wet_weather_inflow

    # Append the differences instead of cumulative values

    S4.append(outflow_diff)
    S5.append(flooding_diff)
    S6.append(wet_weather_inflow_diff)

# Drawing plots
fig = plt.figure(figsize=(12, 8))
fig.suptitle("System Stats of Infiltration, Rainfall, Runoff, Outflow, Flooding, and Wet Weather Inflow")
fig.set_tight_layout(True)

titles = ["Infiltration", "Rainfall", "Runoff", "Outflow", "Flooding", "wet_weather_inflow"]
y_labels = ["Infiltration (cm)", "Rainfall (mm)", "Runoff (mm)", "Outflow (CMS)", "Flooding (CMS)", "wet_weather_inflow (CMS)"]
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