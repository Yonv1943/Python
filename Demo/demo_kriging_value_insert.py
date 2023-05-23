import numpy as np
import matplotlib.pyplot as plt
import kriging

"""
已经在2023年4月12日 14:25:28 发给 pan1hao4lin2 了

在 Kriging 插值中，x 和 y 表示空间位置的坐标，z 表示在该位置上的数值或属性。
在这个例子中，x, y 和 z 是随机生成的 15 个点的坐标和属性，使用 Kriging 插值预测了在 100x100 的网格上的属性值。
通过生成网格点，可以可视化 Kriging 插值的结果，这里使用了 plt.contourf() 函数绘制了二维等高线图。
"""

np.random.seed(1)

x = np.random.uniform(0, 100, 15)
y = np.random.uniform(0, 100, 15)
z = np.random.uniform(0, 100, 15)
xy = np.stack((x, y), axis=1)
print(xy)

kri = kriging.Kriging()
kri.fit(xy, z)

xls = np.linspace(0, 100, 100)
yls = np.linspace(0, 100, 100)
xgrid, ygrid = np.meshgrid(xls, yls)

zgridls = kri.predict(np.c_[xgrid.ravel(), ygrid.ravel()])
zgrid = zgridls.reshape(*xgrid.shape)

fig = plt.figure(figsize=(7, 4))
plt.contourf(xgrid, ygrid, zgrid, cmap='jet')
plt.show()
