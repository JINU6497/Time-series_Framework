import numpy as np
import torch
import tqdm

def window_transform(raw_data, win_len, stride):
    # Feature 수
    n_features = raw_data.shape[1]
    # 총 data 수 - Window size + 1
    k = (len(raw_data) - win_len) // stride + 1
    # K가 음수인 경우는 Win_len이 raw_data보다 긴 경우이므로, 이런 경우 방지
    assert k >= 1
    # (k, window size, feature 수) 크기의 0벡터 선언한 후 한칸씩 데이터 넣음
    ts_array = np.zeros((k, win_len, n_features))
    for i in range(k):
        ts_array[i, :, :] = raw_data[i * stride:i * stride + win_len, :]
    return ts_array

def calculate_correlation_matrix(window):
    n = window.shape[1]
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        x_i = window[:,i]
        for j in range(n):
            x_j = window[:,j]
            corr_matrix[i,j] = np.inner(x_i, x_j)/len(x_i)
    return corr_matrix

def calculate_signature_matrix_dataset(X: np.array, label:np.array, lags=[60, 30, 10], stride=1, num_timesteps=5):
    max_lag = max(lags)
    # (k, window size, feature 수)
    X_w = window_transform(X, max_lag, stride)
    result = []
    # k (window size로 구성된 Input 수)
    for i in range(len(X_w)):
        matrix_list = []
        for lag in sorted(lags, reverse=True):
            current_slice = X_w[i, -lag:, :]
            corr_matrix = calculate_correlation_matrix(current_slice)
            # [(featrue,feature,1), (featrue,feature,1), (featrue,feature,1)] -> [(lag(60)),(lag(30)),(lag(10))]
            matrix_list.append(np.expand_dims(corr_matrix, axis=2))
        signature_matrix = np.concatenate(matrix_list, axis=2)
        result.append(np.expand_dims(signature_matrix, 0))

    matrix_num = len(result) - num_timesteps + 1

    input_matrix_series = []

    for j in range(matrix_num):
        matrix_series = np.expand_dims(np.concatenate(result[j: j + num_timesteps], axis=0), axis=0)
        input_matrix_series.append(matrix_series)

    input_matrix_series = np.concatenate(input_matrix_series, axis=0)
    target = input_matrix_series[:, -1, :, :]
    
    input_matrix_series = torch.Tensor(input_matrix_series.transpose(0, 4, 1, 2, 3))
    target = torch.Tensor(target.transpose(0, 3, 1, 2))
    
    test_label = label[max(lags) + num_timesteps - 2: len(X)].flatten()

    return input_matrix_series, target, test_label

# input_matrix_series = ((k-time step), (time step), (feature), (feature), (lag 길이))
# target = ((k-time step), (feature), (feature), (lag 길이)) -> lag에서 t-1 시점 (lag 10인 matrix)

