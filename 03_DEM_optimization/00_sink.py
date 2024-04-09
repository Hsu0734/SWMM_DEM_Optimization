import whitebox_workflows as wbw
import rasterio as rs
from rasterio.plot import show
import matplotlib.pyplot as plt

wbe = wbw.WbEnvironment()
wbe.verbose = False
wbe.working_directory = r'D:\PhD career\05 SCI papers\06 Multi-objective optimization\SWMM_DEM_Optimization\01_data\DEM'

# web read DEM data
dem = wbe.read_raster('S4.tif')

sink = wbe.sink(dem)
wbe.write_raster(sink, 'DEM_sink.tif', compress=True)

# visualization
path_01 = '../01_data/DEM/DEM_sink.tif'
data_01 = rs.open(path_01)

fig, ax = plt.subplots(figsize=(16, 16))
ax.tick_params(axis='both', which='major', labelsize=20)
show(data_01, title='DEM_demo_d8', ax=ax)

plt.ticklabel_format(style='plain')
# ax.get_xaxis().get_major_formatter().set_scientific(False)  # 关闭科学计数法
# ax.get_yaxis().get_major_formatter().set_scientific(False)  # 关闭科学计数法
# grid and show plot
ax.grid(True, linestyle='--', color='grey')

# 添加颜色条
cbar_ax = fig.add_axes([0.92, 0.19, 0.03, 0.3])  # 调整颜色条的位置和大小
cbar = plt.colorbar(ax.images[0], cax=cbar_ax)  # 使用 ax.images[0] 获取图像数据用于颜色条

# 调整颜色条上刻度标签的字体大小
cbar.ax.tick_params(labelsize=20)
plt.show()

# flow accumulation value
Sink_value = []
for row in range(sink.configs.rows):
    for col in range(sink.configs.columns):
        elev = sink[row, col]
        if elev != sink.configs.nodata:
            Sink_value.append(elev)

# print(Flow_accum_value)
print(Sink_value)
