
import numpy as np
import matplotlib.pyplot as plt
y = []
x = []
for i in range(11):
    data = np.load('/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/Assemble_task/data/routine_pictures/reality/t4/cost_'+str(i)+'.npy')
    y.append(data[0])
    x.append(i+1)
x.append(12)
y.append(0.24)
y = np.array(y)

x = np.array(x)

# 示例数据
# x = np.linspace(0, 10, 100)  # 创建一个包含100个点的从0到10的等差数列
# y = np.sin(x)               # 计算每个点的正弦值

# 绘制折线图
plt.plot(x, y)

# 设置标题和轴标签
plt.title('Gradient Descent')
plt.xlabel('times')
plt.ylabel('cost')

# # 显示网格线
# plt.grid(True)

# 显示图形
plt.show()
