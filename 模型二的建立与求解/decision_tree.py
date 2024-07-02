# 导入所需库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

# 启用pandas和R之间的数据框转换
pandas2ri.activate()

# 安装并导入C50库
utils = importr('utils')
utils.install_packages('C50')
C50 = importr('C50')

# 假设我们有一些数据
# X是自变量（特征），y是因变量（标签）
X = np.array([[...], [...], ...])  # 替换为实际数据
y = np.array([...])  # 替换为实际数据

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 将训练集和测试集转换为pandas数据框
train_df = pd.DataFrame(X_train)
train_df['target'] = y_train

test_df = pd.DataFrame(X_test)
test_df['target'] = y_test

# 将pandas数据框转换为R数据框
r_train_df = pandas2ri.py2rpy(train_df)
r_test_df = pandas2ri.py2rpy(test_df)

# 在R中使用C5.0函数训练模型
C5_0_model = C50.C5_0(r_train_df.rx2('target'), r_train_df.drop('target', axis=1))

# 在测试集上进行预测
predictions = C50.predict(C5_0_model, r_test_df.drop('target', axis=1))

# 将R向量转换为numpy数组
predictions = np.array(predictions)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print(f'测试集准确率: {accuracy * 100:.2f}%')

# 检查决策树的稳定性和准确性
if accuracy >= 0.90:
    print('最终决策树模型已建立')
else:
    print('需要进行修剪和调整')
