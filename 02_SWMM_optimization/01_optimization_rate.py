'''
PySWMM + Pymoo Code Practice
A simple optimization case of LID (BR) area
Objective: Min(total Cost) & Max(Reduction of runoff volume) & & Max(Reduction of peak volume)
Author: Hanwen Xu
Version: 1
Date: Nov 01, 2023
'''


from pyswmm import Simulation, Subcatchments, LidGroups, SystemStats
import numpy as np
# choose problem
from pymoo.core.problem import ElementwiseProblem
# choose algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination


# setting variables


with Simulation(r'D:\PhD career\05 SCI papers\06 Multi-objective optimization\SWMM_DEM_Optimization\GuanyaoshanNJ.inp') as sim:
    lid_on_subs = LidGroups(sim)

# define MOO problem
    class MyProblem(ElementwiseProblem):

        def __init__(self, lid_on_subs, **kwargs):
            super().__init__(n_var=30,
                             n_obj=3,
                             n_ieq_constr=2,
                             xl=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             xu=np.array([13500, 8100, 7600, 9300, 11900, 5100, 11700, 13000, 9100, 6400, 18500, 6600, 5400,
                                          16700, 11400, 17200, 8000, 12400, 10500, 8200, 7600, 14700, 9500, 8200, 19800, 26300, 8100, 19800, 5600, 12300]),
                             **kwargs)
            self.lid_on_subs = lid_on_subs

        def _evaluate(self, x, out, *args, **kwargs):
            S1_BR = x[0]
            S1_1_BR = x[1]
            S2_BR = x[2]
            S2_1_BR = x[3]
            S3_BR = x[4]
            S3_1_BR = x[5]
            S4_BR = x[6]
            S4_1_BR = x[7]
            S5_BR = x[8]
            S5_1_BR = x[9]
            S6_BR = x[10]
            S6_1_BR = x[11]
            S6_2_BR = x[12]
            S7_BR = x[13]
            S7_1_BR = x[14]
            S8_BR = x[15]
            S9_BR = x[16]
            S10_BR = x[17]
            S10_1_BR = x[18]
            S11_BR = x[19]
            S11_1_BR = x[20]
            S12_BR = x[21]
            S12_1_BR = x[22]
            S12_2_BR = x[23]
            S12_3_BR = x[24]
            S12_4_BR = x[25]
            S13_BR = x[26]
            S13_1_BR = x[27]
            S14_BR = x[28]
            S14_1_BR = x[29]

            cost_function = 0.04 * (S1_BR + S1_1_BR + S2_BR + S2_1_BR + S3_BR + S3_1_BR + S4_BR + S4_1_BR + S5_BR + S5_1_BR +
                                    S6_BR + S6_1_BR + S6_2_BR + S7_BR + S7_1_BR + S8_BR + S9_BR + S10_BR + S10_1_BR + S11_BR +
                                    S11_1_BR + S12_BR + S12_1_BR + S12_2_BR + S12_3_BR + S12_4_BR + S13_BR + S13_1_BR + S14_BR + S14_1_BR)
            total_runoff_function, peak_runoff_fuction = compute_runoff(S1_BR, S1_1_BR, S2_BR, S2_1_BR, S3_BR, S3_1_BR, S4_BR, S4_1_BR, S5_BR, S5_1_BR,
                                    S6_BR, S6_1_BR, S6_2_BR, S7_BR, S7_1_BR, S8_BR, S9_BR, S10_BR, S10_1_BR, S11_BR,
                                    S11_1_BR, S12_BR, S12_1_BR, S12_2_BR, S12_3_BR, S12_4_BR, S13_BR, S13_1_BR, S14_BR, S14_1_BR)

            # constrain function
            g1 = 10000 - (S1_BR + S1_1_BR + S2_BR + S2_1_BR + S3_BR + S3_1_BR + S4_BR + S4_1_BR + S5_BR + S5_1_BR + S6_BR + S6_1_BR + \
                 S6_2_BR + S7_BR + S7_1_BR + S8_BR + S9_BR + S10_BR + S10_1_BR + S11_BR + S11_1_BR + S12_BR + S12_1_BR + S12_2_BR + \
                 S12_3_BR + S12_4_BR + S13_BR + S13_1_BR + S14_BR + S14_1_BR)
            g2 =(S1_BR + S1_1_BR + S2_BR + S2_1_BR + S3_BR + S3_1_BR + S4_BR + S4_1_BR + S5_BR + S5_1_BR + S6_BR + S6_1_BR +
                 S6_2_BR + S7_BR + S7_1_BR + S8_BR + S9_BR + S10_BR + S10_1_BR + S11_BR + S11_1_BR + S12_BR + S12_1_BR + S12_2_BR +
                 S12_3_BR + S12_4_BR + S13_BR + S13_1_BR + S14_BR + S14_1_BR) - 50000

            out["F"] = [cost_function, total_runoff_function, peak_runoff_fuction]
            out["G"] = [g1, g2]

    def compute_runoff(S1_BR, S1_1_BR, S2_BR, S2_1_BR, S3_BR, S3_1_BR, S4_BR, S4_1_BR, S5_BR, S5_1_BR,
                       S6_BR, S6_1_BR, S6_2_BR, S7_BR, S7_1_BR, S8_BR, S9_BR, S10_BR, S10_1_BR, S11_BR,
                       S11_1_BR, S12_BR, S12_1_BR, S12_2_BR, S12_3_BR, S12_4_BR, S13_BR, S13_1_BR, S14_BR, S14_1_BR):

        with Simulation(r'D:\PhD career\05 SCI papers\06 Multi-objective optimization\SWMM_DEM_Optimization\GuanyaoshanNJ.inp') as sim:
            system_routing = SystemStats(sim)

            Time_stamps = [] # add time series
            Runoff = [] # creat a set for runoff data collection
            Peak_Runoff = [] # creat a set for runoff data collection
            prev_runoff = 0

            lid_on_subs.__getitem__('S1').__getitem__(0).unit_area = S1_BR
            lid_on_subs.__getitem__('S1-1').__getitem__(0).unit_area = S1_1_BR
            lid_on_subs.__getitem__('S2').__getitem__(0).unit_area = S2_BR
            lid_on_subs.__getitem__('S2-1').__getitem__(0).unit_area = S2_1_BR
            lid_on_subs.__getitem__('S3').__getitem__(0).unit_area = S3_BR
            lid_on_subs.__getitem__('S3-1').__getitem__(0).unit_area = S3_1_BR
            lid_on_subs.__getitem__('S4').__getitem__(0).unit_area = S4_BR
            lid_on_subs.__getitem__('S4-1').__getitem__(0).unit_area = S4_1_BR
            lid_on_subs.__getitem__('S5').__getitem__(0).unit_area = S5_BR
            lid_on_subs.__getitem__('S5-1').__getitem__(0).unit_area = S5_1_BR
            lid_on_subs.__getitem__('S6').__getitem__(0).unit_area = S6_BR
            lid_on_subs.__getitem__('S6-1').__getitem__(0).unit_area = S6_1_BR
            lid_on_subs.__getitem__('S6-2').__getitem__(0).unit_area = S6_2_BR
            lid_on_subs.__getitem__('S7').__getitem__(0).unit_area = S7_BR
            lid_on_subs.__getitem__('S7-1').__getitem__(0).unit_area = S7_1_BR
            lid_on_subs.__getitem__('S8').__getitem__(0).unit_area = S8_BR
            lid_on_subs.__getitem__('S9').__getitem__(0).unit_area = S9_BR
            lid_on_subs.__getitem__('S10').__getitem__(0).unit_area = S10_BR
            lid_on_subs.__getitem__('S10-1').__getitem__(0).unit_area = S10_1_BR
            lid_on_subs.__getitem__('S11').__getitem__(0).unit_area = S11_BR
            lid_on_subs.__getitem__('S11-1').__getitem__(0).unit_area = S11_1_BR
            lid_on_subs.__getitem__('S12').__getitem__(0).unit_area = S12_BR
            lid_on_subs.__getitem__('S12-1').__getitem__(0).unit_area = S12_1_BR
            lid_on_subs.__getitem__('S12-2').__getitem__(0).unit_area = S12_2_BR
            lid_on_subs.__getitem__('S12-3').__getitem__(0).unit_area = S12_3_BR
            lid_on_subs.__getitem__('S12-4').__getitem__(0).unit_area = S12_4_BR
            lid_on_subs.__getitem__('S13').__getitem__(0).unit_area = S13_BR
            lid_on_subs.__getitem__('S13-1').__getitem__(0).unit_area = S13_1_BR
            lid_on_subs.__getitem__('S14').__getitem__(0).unit_area = S14_BR
            lid_on_subs.__getitem__('S14-1').__getitem__(0).unit_area = S14_1_BR

            for step in sim:
                Time_stamps.append(sim.current_time)
                runoff_stats = system_routing.runoff_stats
                runoff_volume = runoff_stats.__getitem__('runoff')

                Runoff.append(runoff_volume)

                runoff_diff = runoff_volume - prev_runoff
                Peak_Runoff.append(runoff_diff)

                prev_runoff = runoff_volume

            sum_runoff_reduction_rate = ((max(Runoff) - 59.4817) / 59.4817) * 100
            peak_runoff_reduction_rate = ((max(Peak_Runoff) - 0.7438) / 0.7438) * 100
            return sum_runoff_reduction_rate, peak_runoff_reduction_rate


    problem = MyProblem(lid_on_subs)


    algorithm = NSGA2(
        pop_size=500,
        n_offsprings=200,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )


    termination = get_termination("n_gen", 200)

    from pymoo.optimize import minimize
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    X = res.X
    F = res.F


import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter

# 3D Visualization
plot = Scatter(tight_layout=True)
plot.add(F, s=10)
plot.show()

# 2D Pairwise Scatter Plots
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=20, facecolors='none', edgecolors='blue')
plt.title("Total runoff (y) and total cost (x)")
plt.grid()
plt.show()

plt.scatter(F[:, 1], F[:, 2], s=20, facecolors='none', edgecolors='blue')
plt.title("Peak volume (y) and Total runoff (x)")
plt.grid()
plt.show()

plt.scatter(F[:, 0], F[:, 2], s=20, facecolors='none', edgecolors='blue')
plt.title("Peak volume (y) and total cost (x)")
plt.grid()
plt.show()

# save the data
import pandas as pd
result_df = pd.DataFrame(F)
result_df.to_csv('output_objectives.csv', index=False)
result_dx = pd.DataFrame(X)
result_dx.to_csv('output_solutions.csv', index=False)