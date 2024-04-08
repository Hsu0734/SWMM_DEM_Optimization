"""
Multi-objective optimization: topographic modification optimization
Author: Hanwen Xu
Version: 1
Date: Nov 10, 2023
"""

import whitebox_workflows as wbw
from pymoo.core.problem import ElementwiseProblem
import numpy as np
import pandas as pd


wbe = wbw.WbEnvironment()
wbe.verbose = False

wbe.working_directory = r'D:\PhD career\05 SCI papers\06 Multi-objective optimization\SWMM_DEM_Optimization\01_data\DEM'
dem = wbe.read_raster('S4.tif')

# creat a blank raster image of same size as the dem
cut_and_fill = wbe.new_raster(dem.configs)

# number of valid grid
n_grid = 0

for row in range(dem.configs.rows):
    for col in range(dem.configs.columns):
        if dem[row, col] == dem.configs.nodata:
            cut_and_fill[row, col] = dem.configs.nodata
        elif dem[row, col] != dem.configs.nodata:
            cut_and_fill[row, col] = 0.0
            n_grid = n_grid + 1
print(n_grid)
n_BRC = 592

# ------------------------------------------ #
# define MOO problem
class MyProblem(ElementwiseProblem):

    def __init__(self, n_grid, **kwargs):
        super().__init__(n_var=int(n_grid),
                         n_obj=3,
                         n_ieq_constr=1,
                         n_eq_constr=0,
                         xl=np.array([0] * n_grid),
                         xu=np.array([1] * n_grid),
                         **kwargs)
        self.n_grid = n_grid

    def _evaluate(self, x, out, *args, **kwargs):

        var_list = [int(value) for value in x]

        earth_volume_function = sum(abs(i) for i in var_list) * 4 * 0.5
        flow_length_function, velocity_function = path_sum_calculation(var_list)

        # notice your function should be <= 0
        g1 = sum(abs(i) for i in var_list) - (int(n_BRC) * 1.2)
        # g2 = (int(n_BRC) * 0.8) - sum(abs(i) for i in var_list)

        out["F"] = [earth_volume_function, flow_length_function, velocity_function]
        out["G"] = [g1]

def path_sum_calculation(var_list):
    i = 0
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            if dem[row, col] == dem.configs.nodata:
                cut_and_fill[row, col] = dem.configs.nodata
            elif dem[row, col] != dem.configs.nodata:
                cut_and_fill[row, col] = var_list[i] * 0.5
                i = i + 1

    # creat dem_pop
    dem_pop = wbe.raster_calculator(expression="'dem' - 'cut_and _fill'", input_rasters=[dem, cut_and_fill])

    # path length calculation
    flow_accum = wbe.d8_flow_accum(dem_pop, out_type='cells')
    slope = wbe.slope(dem_pop, units="percent")

    path_length = wbe.new_raster(flow_accum.configs)
    velocity = wbe.new_raster(flow_accum.configs)

    for row in range(flow_accum.configs.rows):
        for col in range(flow_accum.configs.columns):
            elev = flow_accum[row, col]   # Read a cell value from a Raster
            velo = flow_accum[row, col]
            if elev >= 13.68 and elev != flow_accum.configs.nodata:
                path_length[row, col] = 1.0
            elif elev < 13.68 or elev == flow_accum.configs.nodata:
                path_length[row, col] = 0.0

            if velo == flow_accum.configs.nodata:
                velocity[row, col] = slope.configs.nodata
            elif velo != flow_accum.configs.nodata:
                slope_factor = (slope[row, col] / 100) ** 0.5
                flow_factor = (flow_accum[row, col] * 4 * 0.00001042) ** (2 / 3)
                velocity[row, col] = (slope_factor * flow_factor / 0.03) ** 0.6

    path = []
    for row in range(path_length.configs.rows):
        for col in range(path_length.configs.columns):
            path.append(path_length[row, col])


    velocity_value = []
    for row in range(velocity.configs.rows):
        for col in range(velocity.configs.columns):
            velocity_value.append(velocity[row, col])

    path_sum = sum(path)
    max_velocity = max(velocity_value)
    return path_sum, max_velocity

problem = MyProblem(n_grid)


# choose algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling

algorithm = NSGA2(
    pop_size=200,
    n_offsprings=50,
    sampling=BinaryRandomSampling(),
    crossover=TwoPointCrossover(prob=0.9),  # 适合二元变量的交叉操作
    mutation=BitflipMutation(prob=0.1),
    eliminate_duplicates=True)

termination = get_termination("n_gen", 100)

from pymoo.optimize import minimize
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

X = res.X
F = res.F



# Visualization of Objective space or Variable space
from pymoo.visualization.scatter import Scatter

# 3D Visualization
plot = Scatter(tight_layout=True)
plot.add(F, s=10)
plot.show()

# save the data
result_df = pd.DataFrame(F)
result_df.to_csv('output_S4.csv', index=False)
result_df = pd.DataFrame(X)
result_df.to_csv('output_variable_S4.csv', index=False)

