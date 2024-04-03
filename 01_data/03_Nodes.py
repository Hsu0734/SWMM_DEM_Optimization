'''
PySWMM Code Example
添加for循环语句输出所有Nodes名称集合和相关数据
Loop through and output the name collections and related data for all Subcatchments, Nodes, and Links.
Author: Hanwen Xu
Version: 1
Date: Nov 01, 2023
'''

from pyswmm import Simulation, Nodes
import matplotlib.pyplot as plt

# Drawing plot
fig = plt.figure(figsize=(12, 8), dpi=300) #Inches Width, Height
fig.suptitle("Nodes data output")
fig.set_tight_layout(True)

# Simulation
sim = Simulation("../GuanyaoshanNJ.inp")
time_stamps = []

# 查找所有节点 All Nodes
nodes = Nodes(sim)
node_names = [node.nodeid for node in nodes]
print(node_names)
inflow_data = {}
for node_name in node_names:
    node = Nodes(sim)[node_name]
    inflow_data[node_name] = []

for step in sim:
    time_stamps.append(sim.current_time)
    for node_name in node_names:
        node = Nodes(sim)[node_name]
        inflow_data[node_name].append(node.total_inflow)   # choose your node data

for node_name in node_names:
    # print(inflow_data[node_name])
    fig02 = plt.subplot(1, 1, 1)
    fig02.set_ylabel("Inflow volume (CMS)")  # label标签字号
    fig02.set_xlabel("Time")
    fig02.plot(time_stamps, inflow_data[node_name], label=f"{node_name}")
    fig02.grid("xy")
    fig02.legend()

plt.show()
sim.close()