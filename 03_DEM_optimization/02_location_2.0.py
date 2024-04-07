"""
Multi-objective optimization: DEM optimization
Author: Hanwen Xu
Version: 1
Date: April 10, 2024
"""

import whitebox_workflows as wbw
from pymoo.core.problem import ElementwiseProblem
import numpy as np
import pandas as pd


wbe = wbw.WbEnvironment()
wbe.verbose = False

# file name list
tiff_prefixes = ['S1', 'S1-1', 'S2', 'S2-1', 'S3', 'S3-1', 'S4', 'S4-1', 'S5', 'S5-1',
                 'S6', 'S6-1', 'S6-2', 'S7', 'S7-1', 'S8', 'S9', 'S10', 'S10-1', 'S11',
                 'S11-1', 'S12', 'S12-1', 'S12-2', 'S12-3', 'S12-4', 'S13', 'S13-1', 'S14', 'S14-1']

optimization_grid = [51, 1, 29, 24, 35, 95, 592, 1, 286, 2, 748, 87, 33, 464, 16,
                     395, 210, 48, 110, 68, 190, 803, 169, 10, 154, 54, 583, 0, 8, 60]

dem_n_grid = []

for prefix in tiff_prefixes:
    # 构建文件路径
    file_path = r'D:\PhD career\05 SCI papers\06 Multi-objective optimization\SWMM_DEM_Optimization\01_data\DEM\{}.tif'.format(prefix)
    dem = wbe.read_raster(file_path)
    # 创建与DEM相同大小的空白栅格图像
    cut_and_fill = wbe.new_raster(dem.configs)
    n_grid = 0

    # 计算有效网格数
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            if dem[row, col] != dem.configs.nodata:  # 检查是否为有效数据
                cut_and_fill[row, col] = 0.0  # 将有效网格设置为0.0
                n_grid += 1  # 有效网格数加1
    dem_n_grid.append(n_grid)

    n_round = 0
    # define MOO problem
    class MyProblem(ElementwiseProblem):

        def __init__(self, n_grid, **kwargs):
            super().__init__(n_var=int(n_grid),
                             n_obj=3,
                             n_ieq_constr=2,
                             n_eq_constr=0,
                             xl=np.array([0] * n_grid),
                             xu=np.array([1] * n_grid),
                             **kwargs)
            self.n_grid = n_grid

        def _evaluate(self, x, out, *args, **kwargs):
            var_list = []
            for i in range(n_grid):
                var_list.append(x[i])

            earth_volume_function = sum(abs(i) for i in var_list) * 4 * 0.5
            flow_length_function, velocity_function = path_sum_calculation(var_list)

            # notice your function should be <= 0
            g1 = sum(abs(i) for i in var_list) - (optimization_grid[n_round] * 1.1)
            g2 = (optimization_grid[n_round] * 0.9) - sum(abs(i) for i in var_list)

            out["F"] = [earth_volume_function, flow_length_function, velocity_function]
            out["G"] = [g1, g2]


    def path_sum_calculation(var_list):
        i = 0
        for row in range(dem.configs.rows):
            for col in range(dem.configs.columns):
                if dem[row, col] == dem.configs.nodata:
                    cut_and_fill[row, col] = dem.configs.nodata
                elif dem[row, col] != dem.configs.nodata:
                    cut_and_fill[row, col] = var_list[i]
                    i = i + 1

        # creat dem_pop
        dem_pop = wbe.raster_calculator(expression="'dem' - 'cut_and _fill'", input_rasters=[dem, cut_and_fill])

        # path length calculation
        flow_accum = wbe.d8_flow_accum(dem_pop, out_type='cells')
        slope = wbe.slope(dem_pop, units="percent")

        path_length = wbe.new_raster(flow_accum.configs)
        velocity = wbe.new_raster(flow_accum.configs)
        Flow_accum_value = []

        for row in range(flow_accum.configs.rows):
            for col in range(flow_accum.configs.columns):
                accum = flow_accum[row, col]  # Read a cell value from a Raster
                if accum != flow_accum.configs.nodata:
                    Flow_accum_value.append(accum)

        for row in range(flow_accum.configs.rows):
            for col in range(flow_accum.configs.columns):
                elev = flow_accum[row, col]
                velo = flow_accum[row, col]
                if elev >= max(Flow_accum_value) * 0.02 and elev != flow_accum.configs.nodata:
                    path_length[row, col] = 1.0
                elif elev < max(Flow_accum_value) * 0.02 or elev == flow_accum.configs.nodata:
                    path_length[row, col] = 0.0

                if velo == flow_accum.configs.nodata:
                    velocity[row, col] = slope.configs.nodata
                elif velo != flow_accum.configs.nodata:
                    slope_factor = (slope[row, col] / 100) ** 0.5
                    flow_factor = (flow_accum[row, col] * 4 * 0.00001042) ** (2 / 3)
                    velocity[row, col] = (slope_factor * flow_factor / 0.03) ** 0.6
        # 找到path length和max velocity
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
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.termination import get_termination

    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.optimize import minimize
    from pymoo.problems.single.knapsack import create_random_knapsack_problem

    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=50,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=0.9),  # 适合二元变量的交叉操作
        mutation=BitflipMutation(prob=0.1),
        eliminate_duplicates=True)

    termination = get_termination("n_gen", 50)

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
    import matplotlib.pyplot as plt

    # 3D Visualization
    plot = Scatter(tight_layout=True)
    plot.add(F, s=10)
    # 将图像保存到文件，文件名根据当前的n_round来命名
    plot_figure_path = 'scatter_plot_round_{}.png'.format(n_round)
    plot.save(plot_figure_path)  # 使用save方法而不是show

    # 将目标函数值保存到CSV文件，文件名包含当前的n_round
    output_objectives_file = 'output_objectives_round_{}.csv'.format(n_round)
    result_df = pd.DataFrame(F)
    result_df.to_csv(output_objectives_file, index=False)

    # 将解的变量值保存到CSV文件，文件名包含当前的n_round
    output_solutions_file = 'output_solutions_round_{}.csv'.format(n_round)
    result_dx = pd.DataFrame(X)
    result_dx.to_csv(output_solutions_file, index=False)

    n_round = n_round + 1
