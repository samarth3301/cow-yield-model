def rmse(y_true, y_pred):
    import numpy as np
    return ((y_true - y_pred) ** 2).mean() ** 0.5
