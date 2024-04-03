'''
PySWMM Code Example
Author: Hanwen Xu
输出基本信息/ Basic information output
Version: 1
Date: Nov 01, 2023
'''

from pyswmm import swmm5
import pyswmm.toolkitapi as tka

# 打开，运行，关闭swmm模型对象
your_model = swmm5.PySWMM("../GuanyaoshanNJ.inp")

your_model.swmm_open()

your_model.swmm_start(True)
while (True):
    time = your_model.swmm_step()  # or swmm_stride()
    if (time <= 0.0): break
your_model.swmm_end()
your_model.swmm_report()


# 输出模型基本信息 basic model information
print("Simulation Time Info")
print("Start Time")
print(your_model.getSimulationDateTime(tka.SimulationTime.StartDateTime.value))

print("End Time")
print(your_model.getSimulationDateTime(tka.SimulationTime.EndDateTime.value))

print("Report Time")
print(your_model.getSimulationDateTime(tka.SimulationTime.ReportStart.value))

print("Simulation Units")
print(your_model.getSimUnit(tka.SimulationUnits.FlowUnits.value))

print("Simulation Engine Version")
print(your_model.swmm_getVersion())

print("Simulation Allow Ponding Option Selection")
print(your_model.getSimAnalysisSetting(tka.SimAnalysisSettings.AllowPonding.value))

print("Simulation Routing Step")
print(your_model.getSimAnalysisSetting(tka.SimulationParameters.RouteStep.value))

print("Number of Nodes")
print(your_model.getProjectSize(tka.ObjectType.NODE.value))

print("Node ID")
IDS = your_model.getObjectIDList(tka.ObjectType.NODE.value)
print(IDS)
print('ID,Invert,Type')
for ind, idd in enumerate(IDS):
    print(
        ind,
        idd,
        your_model.getNodeParam(idd, tka.NodeParams.invertElev.value),
        your_model.getNodeParam(idd, tka.NodeParams.fullDepth.value),
        your_model.getNodeType(idd), )

print("Link ID")
print('ID,offset1,LinkConnections')
IDS = your_model.getObjectIDList(tka.ObjectType.LINK.value)
print(IDS)
for ind, idd in enumerate(IDS):
    print(
        ind,
        idd,
        your_model.getLinkParam(idd, tka.LinkParams.offset1.value),
        your_model.getLinkConnections(idd), )

print("SUBCATCH ID")
print('ID,area,OutConnections')
IDS = your_model.getObjectIDList(tka.ObjectType.SUBCATCH.value)
print(IDS)
for ind, idd in enumerate(IDS):
    print(
        ind,
        idd,
        your_model.getSubcatchParam(idd, tka.SubcParams.area.value),
        your_model.getSubcatchOutConnection(idd), )

your_model.swmm_start(False)
i = 0
while True:
    if i % 1000 == 0:
        print("test {}".format(your_model.flow_routing_stats()))
    eltime = your_model.swmm_step()
    i += 1
    if eltime == 0:
        break

your_model.swmm_end()
your_model.swmm_close()