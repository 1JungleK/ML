import numpy as np
import matplotlib.pyplot as plt

# 数据预处理
x, y = [], []

for sample in open("../Data/price.txt", "r"):
    _x, _y = sample.split(",")

    x.append(float(_x))
    y.append(float(_y))

x, y = np.array(x), np.array(y)

# 标准化
x = (x - x.mean()) / x.std()

plt.figure()
plt.scatter(x, y, c="g", s=6)
plt.show()

# 模型确实
# 根据数据预处理结果选定要选择的模型

# 在(-2, 4)这个区间上选100个点作为画图基础
x0 = np.linspace(-2, 4, 100)


# 返回一个函数，接收input_x, 输出线性回归多项式的值 y
def get_model(degree):
    return lambda input_x=x0: np.polyval(np.polyfit(x, y, degree), input_x)


# 返回Loss Function 的值
def get_cost(degree, input_x, input_y):
    return 0.5 * ((get_model(degree)(input_x) - input_y) ** 2).sum()


# 设定测试集， 使用原数据进行比较选择degree的值（画出不同degree下进行拟合的图像）
plt.scatter(x, y, c="g", s=20)
test_set = [1, 2, 3, 4, 10]

for deg in test_set:
    # 打印Loss值 （当deg = 10 时， Loss值最小）
    print(get_cost(deg, x, y))

    plt.plot(x0, get_model(deg)(), label="degree = {}".format(deg))

plt.xlim(-2, 4)
plt.ylim(1e5, 8e5)

plt.legend()
# 依据图像， deg = 10 为过拟合
plt.show()
