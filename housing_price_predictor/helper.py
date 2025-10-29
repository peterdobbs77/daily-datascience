
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, root_mean_squared_log_error, r2_score

def evaluate_model_outputs(y_actual, y_pred):
    """"""
    rmse_log_error = root_mean_squared_log_error(y_actual, y_pred)
    print('RMSE Log score : {0:.3f}'.format(rmse_log_error))
    _rf_score = r2_score(y_actual, y_pred)
    print('R^2 score : {0:.3f}'.format(_rf_score))
    pass