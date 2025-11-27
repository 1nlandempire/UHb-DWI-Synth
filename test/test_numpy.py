import numpy as np

# 针对ln后的数据进行异常值清理
def clean_log(data):
    # negative
    data = np.where(data < 0, 0, data)
    # nan -> 0
    data = np.where(np.isnan(data), 0, data)
    # inf
    data = np.where(np.isinf(data), 0, data)
    # 其他的异常值
    data = np.where(np.isreal(data), data, 0)
    return data

x = np.array([5000, 10000, 700, 400, 300, 200, 0, -0.01])



print(np.log(x))
print(clean_log(np.log(x)))