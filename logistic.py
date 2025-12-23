import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 定义文件路径（训练集和测试集，若测试集文件名不同请自行修改）
train_filename = r"C:\Users\张焜\Documents\GitHub\logsitic\horseColicTraining.txt"
test_filename = r"C:\Users\张焜\Documents\GitHub\logsitic\horseColicTraining.txt"  # 若有独立测试集请替换路径

filename=r"c:\Users\Administrator\Desktop\机器学习\lesson4\testSet.txt"
#=====================
# 1. 数据读取函数
#=====================
def load_dataset(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]   # 特征
    y = data[:, -1]    # 标签
    return X, y
#=====================
# 2. 缺失值处理函数
#   （缺失值替换为该列均值）
#=====================
def replace_nan_with_mean(X):
    for i in range(X.shape[1]):
        col = X[:, i]
        # 选择非0的数作为有效特征
        # 选择非0的数作为有效特征（此处假设0代表缺失值）
        valid = col[col != 0]
        if len(valid) > 0:
            mean_val = np.mean(valid)
            col[col == 0] = mean_val
            X[:, i] = col
    return X
#=====================
# 3. 主流程
#=====================
# 读取训练集

X_train, y_train = load_dataset(train_filename)
# 处理训练集缺失值
X_train = replace_nan_with_mean(X_train)

# 读取测试集

X_test, y_test = load_dataset(test_filename)
# 处理测试集缺失值（使用训练集对应列的均值，避免数据泄露，优化后版本）
for i in range(X_test.shape[1]):
    # 用训练集的有效数据计算均值，用于填充测试集
    train_col = X_train[:, i]
    train_valid = train_col[train_col != 0]
    if len(train_valid) > 0:
        mean_val = np.mean(train_valid)
        test_col = X_test[:, i]
        test_col[test_col == 0] = mean_val
        X_test[:, i] = test_col

#=====================
# 4. 构建并训练逻辑回归模型
#=====================

# 初始化逻辑回归模型（设置随机种子保证结果可复现）
lr_model = LogisticRegression(random_state=42, max_iter=1000)
# 训练模型
lr_model.fit(X_train, y_train)

#=====================
# 5. 测试集预测
#=====================

y_pred = lr_model.predict(X_test)

#=====================
# 6. 计算准确率
#=====================

accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")