from rpy2.robjects.packages import importr
from rpy2.robjects import DataFrame, Formula
from rpy2.robjects import r

C50 = importr('C50')

# 样本数据
# 用你自己的数据集代替
# 最后一列是目标变量
data = [
    [1, 2, 3, 'A'],
    [4, 5, 6, 'B'],
    [7, 8, 9, 'A'],
    # Add more rows as needed
]

# 定义列名
# 替换为您自己的列名
columns = ['feature1', 'feature2', 'feature3', 'target']

# 将 Python 列表转换为 R DataFrame
rdata = DataFrame(data, columns=columns)

# 使用 R 的 C5.0 函数来训练模型
# 注意：这里假设 'target' 是你的目标变量
formula = Formula('target ~ .')
model = C50.C5_0(rdata, formula)

# 进行预测
# 用自己的测试数据替换 test_data 中的值
# 同样需要转换为 R DataFrame，并确保没有目标列
test_data = [
    [10, 11, 12],
    # Add more rows as needed
]
rtestdata = DataFrame(test_data, columns=['feature1', 'feature2', 'feature3'])

# 使用 R 的 predict 函数进行预测
predictions = r.predict(model, newdata=rtestdata)

# 显示预测结果
print("Predictions:", predictions)