# 将训练集的企业的实力 Y1、信贷风险 Y2、违约行为 Y3数据代入建立企业的信誉评级的 Fisher 多分类
# 模型

# 计算出贝叶斯判别函数系数。根据该表，可将企业实力 Y1、风险 Y2、违约 Y3数据代
# 入贝叶斯判别函数比较对应的函数值，按照函数值大小将样本数据的信誉评级归类

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 加载数据集（例如，使用鸢尾花数据集，这是一个经典的多分类数据集）
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化LDA分类器
lda = LinearDiscriminantAnalysis()

# 在训练集上训练LDA分类器
lda.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = lda.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("LDA分类器的准确率:", accuracy)