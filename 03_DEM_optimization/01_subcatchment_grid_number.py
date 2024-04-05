"""
Multi-objective optimization: topographic modification optimization
Author: Hanwen Xu
Version: 1
Date: April 10, 2024
"""
import whitebox_workflows as wbw

wbe = wbw.WbEnvironment()
wbe.verbose = False

# file name list
tiff_prefixes = ['S1', 'S1-1', 'S2', 'S2-1', 'S3', 'S3-1', 'S4', 'S4-1', 'S5', 'S5-1',
                 'S6', 'S6-1', 'S6-2', 'S7', 'S7-1', 'S8', 'S9', 'S10', 'S10-1', 'S11',
                 'S11-1', 'S12', 'S12-1', 'S12-2', 'S12-3', 'S12-4', 'S13', 'S13-1', 'S14', 'S14-1']

for prefix in tiff_prefixes:
    # 构建文件路径
    file_path = r'D:\PhD career\05 SCI papers\06 Multi-objective optimization\SWMM_DEM_Optimization\01_data\DEM\{}.tif'.format(prefix)

    # 读取栅格数据
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

    # 打印有效网格数
    print(f'{prefix}: {n_grid}')