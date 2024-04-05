'''
PySWMM + Pymoo Code Practice
A simple optimization case of LID (BR) area
Objective: Min(total Cost) & Max(Reduction of runoff volume) & & Max(Reduction of peak volume)
Author: Hanwen Xu
Version: 1
Date: Nov 01, 2023
'''

from pyswmm import Simulation, Subcatchments, LidGroups, SystemStats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

solutions_df = pd.read_csv('output_solutions.csv')
fig = plt.figure(figsize=(12, 8))

# 获取指定的解
# solution_index = 99  # 可以修改为任意想要输出的解的索引
# solution = solutions_df.iloc[solution_index]

i = 1
for index in range(100, 120):  # range的结束值是不包含的，所以这里是9而不是8
    solution = solutions_df.iloc[index]

    sim = Simulation(r'../GuanyaoshanNJ.inp')
    lid_on_subs = LidGroups(sim)

    lid_on_subs.__getitem__('S1').__getitem__(0).unit_area = solution[0]
    lid_on_subs.__getitem__('S1-1').__getitem__(0).unit_area = solution[1]
    lid_on_subs.__getitem__('S2').__getitem__(0).unit_area = solution[2]
    lid_on_subs.__getitem__('S2-1').__getitem__(0).unit_area = solution[3]
    lid_on_subs.__getitem__('S3').__getitem__(0).unit_area = solution[4]
    lid_on_subs.__getitem__('S3-1').__getitem__(0).unit_area = solution[5]
    lid_on_subs.__getitem__('S4').__getitem__(0).unit_area = solution[6]
    lid_on_subs.__getitem__('S4-1').__getitem__(0).unit_area = solution[7]
    lid_on_subs.__getitem__('S5').__getitem__(0).unit_area = solution[8]
    lid_on_subs.__getitem__('S5-1').__getitem__(0).unit_area = solution[9]
    lid_on_subs.__getitem__('S6').__getitem__(0).unit_area = solution[10]
    lid_on_subs.__getitem__('S6-1').__getitem__(0).unit_area = solution[11]
    lid_on_subs.__getitem__('S6-2').__getitem__(0).unit_area = solution[12]
    lid_on_subs.__getitem__('S7').__getitem__(0).unit_area = solution[13]
    lid_on_subs.__getitem__('S7-1').__getitem__(0).unit_area = solution[14]
    lid_on_subs.__getitem__('S8').__getitem__(0).unit_area = solution[15]
    lid_on_subs.__getitem__('S9').__getitem__(0).unit_area = solution[16]
    lid_on_subs.__getitem__('S10').__getitem__(0).unit_area = solution[17]
    lid_on_subs.__getitem__('S10-1').__getitem__(0).unit_area = solution[8]
    lid_on_subs.__getitem__('S11').__getitem__(0).unit_area = solution[19]
    lid_on_subs.__getitem__('S11-1').__getitem__(0).unit_area = solution[20]
    lid_on_subs.__getitem__('S12').__getitem__(0).unit_area = solution[21]
    lid_on_subs.__getitem__('S12-1').__getitem__(0).unit_area = solution[22]
    lid_on_subs.__getitem__('S12-2').__getitem__(0).unit_area = solution[23]
    lid_on_subs.__getitem__('S12-3').__getitem__(0).unit_area = solution[24]
    lid_on_subs.__getitem__('S12-4').__getitem__(0).unit_area = solution[25]
    lid_on_subs.__getitem__('S13').__getitem__(0).unit_area = solution[26]
    lid_on_subs.__getitem__('S13-1').__getitem__(0).unit_area = solution[27]
    lid_on_subs.__getitem__('S14').__getitem__(0).unit_area = solution[28]
    lid_on_subs.__getitem__('S14-1').__getitem__(0).unit_area = solution[29]

    system_routing = SystemStats(sim)
    prev_runoff = 0
    time_stamps = [] # add time series
    Peak_Runoff = []

    for step in sim:
        time_stamps.append(sim.current_time)
        runoff_stats = system_routing.runoff_stats
        current_runoff = runoff_stats.__getitem__('runoff')
        runoff_diff = current_runoff - prev_runoff
        prev_runoff = current_runoff
        Peak_Runoff.append(runoff_diff)

    fig = plt.subplot(1, 1, 1)
    fig.set_ylabel("Runoff volume (CMS)")  # label标签字号
    fig.set_xlabel("Time")
    fig.plot(time_stamps, Peak_Runoff, linestyle='-', label=f"solution{i}")
    i = i + 1




# default setting
sim = Simulation(r'../GuanyaoshanNJ.inp')
lid_on_subs = LidGroups(sim)

lid_on_subs.__getitem__('S1').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S1-1').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S2').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S2-1').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S3').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S3-1').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S4').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S4-1').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S5').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S5-1').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S6').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S6-1').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S6-2').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S7').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S7-1').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S8').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S9').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S10').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S10-1').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S11').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S11-1').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S12').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S12-1').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S12-2').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S12-3').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S12-4').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S13').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S13-1').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S14').__getitem__(0).unit_area = 0.0
lid_on_subs.__getitem__('S14-1').__getitem__(0).unit_area = 0.0

system_routing = SystemStats(sim)
prev_runoff = 0
time_stamps = []
original_peak_runoff_solution = []

for step in sim:
    time_stamps.append(sim.current_time)
    runoff_stats = system_routing.runoff_stats
    runoff = runoff_stats.__getitem__('runoff')
    runoff_diff = runoff - prev_runoff
    prev_runoff = runoff
    original_peak_runoff_solution.append(runoff_diff)


fig.plot(time_stamps, original_peak_runoff_solution, linestyle='-', label=f"default situation")
fig.grid("xy")
fig.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
fig.tick_params(axis='x', rotation=30)
fig.legend()
plt.show()
sim.close()