def get_mean_var_std(arr):
    import numpy as np

    arr_mean = np.mean(arr)
    arr_var = np.var(arr)
    arr_std = np.std(arr, ddof=1)
    print("average:%f" % arr_mean)
    print("std:%f" % arr_std)

    return arr_mean, arr_var, arr_std


if __name__ == '__main__':
    arr = [0.3912, 0.392, 0.3885, 0.3919, 0.3925]
    print(get_mean_var_std(arr))
