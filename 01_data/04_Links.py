'''
PySWMM Code Example
添加for循环语句输出所有Nodes名称集合和相关数据
Loop through and output the name collections and related data for all Subcatchments, Nodes, and Links.
Author: Hanwen Xu
Version: 1
Date: Nov 01, 2023
'''

from pyswmm import Simulation, Links
import matplotlib.pyplot as plt

# Drawing plot
fig = plt.figure(figsize=(12, 8), dpi=300)  # Inches Width, Height
fig.suptitle("Links data output")
fig.set_tight_layout(True)

# Simulation
sim = Simulation("../GuanyaoshanNJ.inp")
time_stamps = []

# 查找所有链接 All Links
links = Links(sim)
link_names = [link.linkid for link in links] # 添加模型中的link name (这个模型node和link取名重合了，应注意避免）
print(link_names)
link_inflow_data = {}
for link_name in link_names:
    link = Links(sim)[link_name]
    link_inflow_data[link_name] = []

for step in sim:
    time_stamps.append(sim.current_time)
    for link_name in link_names:
        link = Links(sim)[link_name]
        link_inflow_data[link_name].append(link.flow)   # choose your link data


for link_name in link_names:
    # print(link_inflow_data[link_name])
    fig03 = plt.subplot(1, 1, 1)
    fig03.set_ylabel("Inflow volume (CMS)")  # label标签字号
    fig03.set_xlabel("Time")
    fig03.plot(time_stamps, link_inflow_data[link_name], label=f"{link_name}")
    fig03.grid("xy")
    fig03.legend()

plt.show()
sim.close()

'''Change the Dates setting'''