import pandas as pd
import rasterio as rs
from rasterio.plot import show
import matplotlib.pyplot as plt
import whitebox_workflows as wbw

# 读取CSV文件
df = pd.read_csv('output_variable_S4.csv')

# 获取第一行的数据
n = 0
row = df.iloc[int(n)]
row_list = row.tolist()


wbe = wbw.WbEnvironment()
wbe.verbose = False

wbe.working_directory = r'D:\PhD career\05 SCI papers\06 Multi-objective optimization\SWMM_DEM_Optimization\01_data\DEM'
dem = wbe.read_raster('S4.tif')


layer = wbe.new_raster(dem.configs)
m = 0
for row in range(dem.configs.rows):
    for col in range(dem.configs.columns):
        if dem[row, col] == dem.configs.nodata:
            layer[row, col] = dem.configs.nodata
        elif dem[row, col] != dem.configs.nodata:
            layer[row, col] = row_list[m]
            m = m + 1

dem_solution = dem - layer

sink = wbe.sink(dem_solution)
sink_area = wbe.new_raster(dem.configs)
for row in range(sink_area.configs.rows):
    for col in range(sink_area.configs.columns):
        area = dem[row, col]
        area_sink = sink[row, col]
        if area != dem.configs.nodata and area_sink == sink.configs.nodata:
            sink_area[row, col] = 0.0
        elif area == dem.configs.nodata:
            sink_area[row, col] = dem.configs.nodata
        elif area_sink != sink.configs.nodata:
            sink_area[row, col] = 1.0

wbe.working_directory = r'D:\PhD career\05 SCI papers\06 Multi-objective optimization\SWMM_DEM_Optimization\05_visualization'
wbe.write_raster(sink_area, 'DEM_sink.tif', compress=True)

# visualization
path_01 = '../05_visualization/DEM_sink.tif'
data_01 = rs.open(path_01)

fig, ax = plt.subplots(figsize=(16, 16))
ax.tick_params(axis='both', which='major', labelsize=20)
show(data_01, title='DEM_sink', ax=ax)
plt.ticklabel_format(style='plain')
plt.show()