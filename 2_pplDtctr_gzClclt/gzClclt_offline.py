import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor

# 1. 读取原始数据
df = pd.read_csv("eye_dataset_qt/session_20250520_230541/raw_eye_data.csv")
print(df)
# 2. 处理旋转（90°顺时针拍摄，需要反旋）
H, W = 480, 640
df['pupil_x_rot'] = df['pupil_y']
df['pupil_y_rot'] = W - df['pupil_x']

df['glint_center_x'] = (df['glint1_x'] + df['glint2_x']) / 2
df['glint_center_y'] = (df['glint1_y'] + df['glint2_y']) / 2
df['vec_dx'] = df['pupil_x'] - df['glint_center_x']
df['vec_dy'] = df['pupil_y'] - df['glint_center_y']
# 增加 angle、distance（角度和距离可能更稳定）
df['glint_dist'] = np.linalg.norm(df[['glint1_x', 'glint1_y']].values - df[['glint2_x', 'glint2_y']].values, axis=1)
df['pupil_to_glint_dist'] = np.linalg.norm(df[['pupil_x', 'pupil_y']].values - df[['glint_center_x', 'glint_center_y']].values, axis=1)

# 简单异常值剔除：过滤 gaze 或 pupil/glint 超出合理范围的行
df = df[(df['gaze_x'] > 0) & (df['gaze_x'] < 2000)]
df = df[(df['gaze_y'] > 0) & (df['gaze_y'] < 1200)]
df = df[(df['pupil_x'] > 0) & (df['pupil_x'] < 1000)]
df = df[(df['pupil_y'] > 0) & (df['pupil_y'] < 1000)]
df = df[(df['glint1_x'] > 0) & (df['glint2_x'] > 0)]

print("Data shape after outlier removal:", df.shape)

# 3. 构建多种模型
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV

models = {
    'Linear': make_pipeline(StandardScaler(), LinearRegression()),
    'Poly2': make_pipeline(StandardScaler(), PolynomialFeatures(2), RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])),
    'Poly3': make_pipeline(StandardScaler(), PolynomialFeatures(3), RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])),
    # 'RF': make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100)), # 分类算法，不适合
    # 'SVR': make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(kernel='rbf'))),
    'MLP': make_pipeline(StandardScaler(), MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000)))
}

# 4. 训练模型并评估误差
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# X = df[['pupil_x', 'pupil_y', 'glint1_x', 'glint1_y', 'glint2_x', 'glint2_y']].values
# X = df[['vec_dx', 'vec_dy']].values
X = df[['pupil_x', 'pupil_y', 'glint1_x', 'glint1_y', 'glint2_x', 'glint2_y','vec_dx', 'vec_dy', 'pupil_to_glint_dist']].values
# X = df[['vec_dx', 'vec_dy', 'pupil_to_glint_dist']].values
y = df[['gaze_x', 'gaze_y']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error = np.linalg.norm(y_test - y_pred, axis=1)
    results[name] = {
        'model': model,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'error': error,
        'y_pred': y_pred
    }

# 5. 输出评估结果和绘图
for name, res in results.items():
    print(f"{name} RMSE: {res['rmse']:.2f}")

    plt.figure()
    plt.title(f"{name} Prediction (RMSE={res['rmse']:.2f})")
    plt.scatter(y_test[:,0], y_test[:,1], label='Ground Truth', c='green', alpha=0.5)
    plt.scatter(res['y_pred'][:,0], res['y_pred'][:,1], label='Prediction', c='red', alpha=0.5)
    for i in range(len(y_test)):
        plt.plot([y_test[i,0], res['y_pred'][i,0]], [y_test[i,1], res['y_pred'][i,1]], 'gray', alpha=0.3)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True)
plt.show()
