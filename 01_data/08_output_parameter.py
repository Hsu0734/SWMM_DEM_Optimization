'''output all the area'''


from pyswmm import Simulation, Subcatchments

# 指定SWMM模型文件的路径
input_file = "../GuanyaoshanNJ.inp"

# 初始化SWMM模拟
with Simulation(input_file) as sim:
    # 获取所有分水区
    subcatchments = Subcatchments(sim)

    # 遍历所有分水区并获取它们的面积
    for subcatchment in subcatchments:
        # 获取当前分水区的ID和面积
        subcatchment_id = subcatchment.subcatchmentid
        area = subcatchment.area  # 面积单位通常是英亩或公顷，具体取决于模型的设置

        # 打印分水区ID和面积
        print(f"Subcatchment ID: {subcatchment_id}, Area: {area} square units")
