'''
PySWMM Code Example
添加for循环语句输出所有Subcatchments名称集合和相关数据
Loop through and output the name collections and related data for all Subcatchments, Nodes, and Links.
Author: Hanwen Xu
Version: 1
Date: Nov 01, 2023
'''

from pyswmm import Simulation, Subcatchments
import matplotlib.pyplot as plt

# Drawing plot
fig = plt.figure(figsize=(12, 8), dpi=300) #Inches Width, Height
fig.suptitle("Subcatchments data output")
fig.set_tight_layout(True)

# Simulation
sim = Simulation("../GuanyaoshanNJ.inp")
time_stamps = []

# 创建Subcatchments，Nodes, Links集合
# 查找所有子汇水区 All Subcatchments
subcatchments = Subcatchments(sim)
subcatchment_names = [subcatchment.subcatchmentid for subcatchment in subcatchments]
# subcatchment_names = ['S1', 'S2'...]
print(subcatchment_names)
runoff_data = {}
for subcatchment_name in subcatchment_names:
    subcatchment = Subcatchments(sim)[subcatchment_name]
    runoff_data[subcatchment_name] = []

# 添加模拟数据
for step in sim:
    time_stamps.append(sim.current_time)
    for subcatchment_name in subcatchment_names:
        subcatchment = Subcatchments(sim)[subcatchment_name]
        runoff_data[subcatchment_name].append(subcatchment.runoff)   # choose your sub data


for subcatchment_name in subcatchment_names:
    # print(runoff_data[subcatchment_name])
    fig01 = plt.subplot(1, 1, 1)
    fig01.set_ylabel("Runoff volume (CMS)")  # label标签字号
    fig01.set_xlabel("Time")
    fig01.plot(time_stamps, runoff_data[subcatchment_name], label=f"{subcatchment_name}")
    fig01.grid("xy")
    fig01.legend()

plt.show()
sim.close()

