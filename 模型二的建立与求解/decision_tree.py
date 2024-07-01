from c50 import C50

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
## 替换为您自己的列名
columns = ['feature1', 'feature2', 'feature3', 'target']

# Create a C5.0 decision tree
tree = C50()

# Fit the model
tree.fit(data, columns)

# 进行预测
## 用自己的测试数据替换 test_data 中的值
test_data = [
    [10, 11, 12],
    # Add more rows as needed
]

predictions = tree.predict(test_data)

# Display predictions
print("Predictions:", predictions)