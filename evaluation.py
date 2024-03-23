import numpy as np

def eval_score(truth, preditect):
    threshold = 0.00001
    points_gt = np.asarray(truth.points)
    points_predicted = np.asarray(preditect.points)

    mae = np.mean(np.abs(points_predicted - points_gt))
    print("MAE: ", mae)

    rmse = np.sqrt(np.mean(np.power(points_predicted - points_gt, 2)))
    print("RMSE: ", rmse)

    intersection = np.sum(np.logical_and(points_predicted > threshold, points_gt > threshold))
    precision = intersection / np.sum(points_gt > threshold)
    print("precision: ", precision)
    recall = intersection / np.sum(points_predicted > threshold)
    print("recall: ", recall)
    f_score =  2 * (precision * recall) / (precision + recall)
    print("F-Score: ", f_score)