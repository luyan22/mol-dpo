import matplotlib.pyplot as plt


fig , ax1 = plt.subplots()
ax2 = ax1.twinx()
 
x = [1, 10, 100]
mol_stable = [86.7, 89.8, 88.8]
lumo = [21, 16.7, 17.3]


ax2.plot(x,mol_stable,'s-',color = 'r',label="molecule stability")#s-:方形
ax1.plot(x, lumo,'o-',color='g',label="lumo")#o-:圆形
ax1.set_xlabel("Time step", fontsize=16)#横坐标名字
ax2.set_ylim(85, 92)
ax2.set_ylabel("Molecule Stability↑", fontsize=16)#纵坐标名字
# 控制坐标轴显示的范围
# plt.ylim(15,25)
ax1.set_ylim(15, 25)
ax1.set_ylabel("MAE(lumo, meV)↓", fontsize=16)

# save as svg
# ax1.xticks(fontsize=12)
# ax1.yticks(fontsize=12)
# ax2.xticks(fontsize=12)

ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)

ax1.legend(loc = 0, fontsize=16)
ax2.legend(loc = 4, fontsize=16)
# save as svg
# plt.savefig("mol_stable.svg")
plt.savefig("line_graph.svg")