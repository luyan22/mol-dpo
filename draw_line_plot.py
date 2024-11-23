import matplotlib.pyplot as plt

x = [1, 10, 100]
# x = [5,7,11,17,19,25]#点的横坐标
# mol_stable = [86.7, 89.8, 88.8]
# lumo = [21, 16.7, 17.3]
# # plt.plot(x,mol_stable,'s-',color = 'r',label="Nucleation time")#s-:方形
# plt.plot(x, lumo,'o-',color='g',label="Nucleation time")#o-:圆形
# plt.xlabel("Time step")#横坐标名字
# # plt.ylabel("Molecule Stability↑")#纵坐标名字
# # 控制坐标轴显示的范围
# plt.ylim(15,25)
# plt.ylabel("MAE(lumo, meV)↓")
# plt.legend(loc = "best")#图例
# # save as svg
# # plt.savefig("mol_stable.svg")
# plt.savefig("LUMO.svg")



mol_stable = [86.7, 89.8, 88.8]
plt.plot(x,mol_stable,'s-',color = 'r',label="Nucleation time")#s-:方形
# plt.plot(x, lumo,'o-',color='g',label="Nucleation time")#o-:圆形
plt.xlabel("Time step")#横坐标名字
plt.ylabel("Molecule Stability↑")#纵坐标名字
# 控制坐标轴显示的范围
plt.ylim(85, 92)
# plt.ylabel("MAE(lumo, meV)↓")
plt.legend(loc = "best")#图例
# save as svg
plt.savefig("mol_stable.svg")
# plt.savefig("LUMO.svg")
