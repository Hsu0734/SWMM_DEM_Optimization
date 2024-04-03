'''
PySWMM Code Example
LIDGroups.unitarea属性修改与传递（swmm5)
Author: Hanwen Xu
Version: 1
Date: Nov 01, 2023
'''

from pyswmm import Simulation, LidGroups


sim = Simulation("../GuanyaoshanNJ.inp")
lid_on_subs = LidGroups(sim)

lid_on_subs1 = lid_on_subs.__getitem__('S1')
Bio_retention = lid_on_subs1.__getitem__(0)
Bio_retention_area_S1 = Bio_retention.unit_area
print("S1_BR = ", end='')
print(Bio_retention_area_S1)

lid_on_subs1_1 = lid_on_subs.__getitem__('S1-1')
Bio_retention = lid_on_subs1_1.__getitem__(0)
Bio_retention_area_S1_1 = Bio_retention.unit_area
print("S1_1_BR = ", end='')
print(Bio_retention_area_S1_1)

lid_on_subs2 = lid_on_subs.__getitem__('S2')
Bio_retention = lid_on_subs2.__getitem__(0)
Bio_retention_area_S2 = Bio_retention.unit_area
print("S2_BR = ", end='')
print(Bio_retention_area_S2)

lid_on_subs2_1 = lid_on_subs.__getitem__('S2-1')
Bio_retention = lid_on_subs2_1.__getitem__(0)
Bio_retention_area_S2_1 = Bio_retention.unit_area
print("S2_1_BR = ", end='')
print(Bio_retention_area_S2_1)

lid_on_subs3 = lid_on_subs.__getitem__('S3')
Bio_retention = lid_on_subs3.__getitem__(0)
Bio_retention_area_S3 = Bio_retention.unit_area
print("S3_BR = ", end='')
print(Bio_retention_area_S3)

lid_on_subs3_1 = lid_on_subs.__getitem__('S3-1')
Bio_retention = lid_on_subs3_1.__getitem__(0)
Bio_retention_area_S3_1 = Bio_retention.unit_area
print("S3_1_BR = ", end='')
print(Bio_retention_area_S3_1)

lid_on_subs4 = lid_on_subs.__getitem__('S4')
Bio_retention = lid_on_subs4.__getitem__(0)
Bio_retention_area_S4 = Bio_retention.unit_area
print("S4_BR = ", end='')
print(Bio_retention_area_S4)

lid_on_subs4_1 = lid_on_subs.__getitem__('S4-1')
Bio_retention = lid_on_subs4_1.__getitem__(0)
Bio_retention_area_S4_1 = Bio_retention.unit_area
print("S4_1_BR = ", end='')
print(Bio_retention_area_S4_1)

lid_on_subs5 = lid_on_subs.__getitem__('S5')
Bio_retention = lid_on_subs5.__getitem__(0)
Bio_retention_area_S5 = Bio_retention.unit_area
print("S5_BR = ", end='')
print(Bio_retention_area_S5)

lid_on_subs5_1 = lid_on_subs.__getitem__('S5-1')
Bio_retention = lid_on_subs5_1.__getitem__(0)
Bio_retention_area_S5_1 = Bio_retention.unit_area
print("S5_1_BR = ", end='')
print(Bio_retention_area_S5_1)

lid_on_subs6 = lid_on_subs.__getitem__('S6')
Bio_retention = lid_on_subs6.__getitem__(0)
Bio_retention_area_S6 = Bio_retention.unit_area
print("S6_BR = ", end='')
print(Bio_retention_area_S6)

lid_on_subs6_1 = lid_on_subs.__getitem__('S6-1')
Bio_retention = lid_on_subs6_1.__getitem__(0)
Bio_retention_area_S6_1 = Bio_retention.unit_area
print("S6_1_BR = ", end='')
print(Bio_retention_area_S6_1)

lid_on_subs6_2 = lid_on_subs.__getitem__('S6-2')
Bio_retention = lid_on_subs6_2.__getitem__(0)
Bio_retention_area_S6_2 = Bio_retention.unit_area
print("S6_2_BR = ", end='')
print(Bio_retention_area_S6_2)

lid_on_subs7 = lid_on_subs.__getitem__('S7')
Bio_retention = lid_on_subs7.__getitem__(0)
Bio_retention_area_S7 = Bio_retention.unit_area
print("S7_BR = ", end='')
print(Bio_retention_area_S7)

lid_on_subs7_1 = lid_on_subs.__getitem__('S7-1')
Bio_retention = lid_on_subs7_1.__getitem__(0)
Bio_retention_area_S7_1 = Bio_retention.unit_area
print("S7_1_BR = ", end='')
print(Bio_retention_area_S7_1)

lid_on_subs8 = lid_on_subs.__getitem__('S8')
Bio_retention = lid_on_subs8.__getitem__(0)
Bio_retention_area_S8 = Bio_retention.unit_area
print("S8_BR = ", end='')
print(Bio_retention_area_S8)

lid_on_subs9 = lid_on_subs.__getitem__('S9')
Bio_retention = lid_on_subs9.__getitem__(0)
Bio_retention_area_S9 = Bio_retention.unit_area
print("S9_BR = ", end='')
print(Bio_retention_area_S9)

lid_on_subs10 = lid_on_subs.__getitem__('S10')
Bio_retention = lid_on_subs10.__getitem__(0)
Bio_retention_area_S10 = Bio_retention.unit_area
print("S10_BR = ", end='')
print(Bio_retention_area_S10)

lid_on_subs10_1 = lid_on_subs.__getitem__('S10-1')
Bio_retention = lid_on_subs10_1.__getitem__(0)
Bio_retention_area_S10_1 = Bio_retention.unit_area
print("S10_1_BR = ", end='')
print(Bio_retention_area_S10_1)

lid_on_subs11 = lid_on_subs.__getitem__('S11')
Bio_retention = lid_on_subs11.__getitem__(0)
Bio_retention_area_S11 = Bio_retention.unit_area
print("S11_BR = ", end='')
print(Bio_retention_area_S11)

lid_on_subs11_1 = lid_on_subs.__getitem__('S11-1')
Bio_retention = lid_on_subs11_1.__getitem__(0)
Bio_retention_area_S11_1 = Bio_retention.unit_area
print("S11_1_BR = ", end='')
print(Bio_retention_area_S11_1)

lid_on_subs12 = lid_on_subs.__getitem__('S12')
Bio_retention = lid_on_subs12.__getitem__(0)
Bio_retention_area_S12 = Bio_retention.unit_area
print("S12_BR = ", end='')
print(Bio_retention_area_S12)

lid_on_subs12_1 = lid_on_subs.__getitem__('S12-1')
Bio_retention = lid_on_subs12_1.__getitem__(0)
Bio_retention_area_S12_1 = Bio_retention.unit_area
print("S12_1_BR = ", end='')
print(Bio_retention_area_S12_1)

lid_on_subs12_2 = lid_on_subs.__getitem__('S12-2')
Bio_retention = lid_on_subs12_2.__getitem__(0)
Bio_retention_area_S12_2 = Bio_retention.unit_area
print("S12_2_BR = ", end='')
print(Bio_retention_area_S12_2)

lid_on_subs12_3 = lid_on_subs.__getitem__('S12-3')
Bio_retention = lid_on_subs12_3.__getitem__(0)
Bio_retention_area_S12_3 = Bio_retention.unit_area
print("S12_3_BR = ", end='')
print(Bio_retention_area_S12_3)

lid_on_subs12_4 = lid_on_subs.__getitem__('S12-4')
Bio_retention = lid_on_subs12_4.__getitem__(0)
Bio_retention_area_S12_4 = Bio_retention.unit_area
print("S12_4_BR = ", end='')
print(Bio_retention_area_S12_4)

lid_on_subs13 = lid_on_subs.__getitem__('S13')
Bio_retention = lid_on_subs13.__getitem__(0)
Bio_retention_area_S13 = Bio_retention.unit_area
print("S13_BR = ", end='')
print(Bio_retention_area_S13)

lid_on_subs13_1 = lid_on_subs.__getitem__('S13-1')
Bio_retention = lid_on_subs13_1.__getitem__(0)
Bio_retention_area_S13_1 = Bio_retention.unit_area
print("S13_1_BR = ", end='')
print(Bio_retention_area_S13_1)

lid_on_subs14 = lid_on_subs.__getitem__('S14')
Bio_retention = lid_on_subs14.__getitem__(0)
Bio_retention_area_S14 = Bio_retention.unit_area
print("S14_BR = ", end='')
print(Bio_retention_area_S14)

lid_on_subs14_1 = lid_on_subs.__getitem__('S14-1')
Bio_retention = lid_on_subs14_1.__getitem__(0)
Bio_retention_area_S14_1 = Bio_retention.unit_area
print("S14_1_BR = ", end='')
print(Bio_retention_area_S14_1)