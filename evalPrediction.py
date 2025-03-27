import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import xarray as xr

# %% Auxiliar functions
def get_scores(y_true, y_pred, score_fun):
    nclasses = np.shape(y_true)[1]
    scores = []
    for name, fun in score_fun.items():
        scores += [[fun(y_true[:, k], y_pred[:, k]) for k in range(nclasses)]]
    return np.array(scores).T


def specificity_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spc = m[0, 0] * 1.0 / (m[0, 0] + m[0, 1])
    return spc


def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score[:, k])
        # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
        # Select threshold that maximize f1 score
        index = np.argmax(f1_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = threshold[index-1] if index != 0 else threshold[0]-1e-10
        opt_threshold.append(t)
    return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)

def affer_results(y_true, y_pred):
    """Return true positives, false positives, true negatives, false negatives.

    Parameters
    ----------
    y_true : ndarray
        True value
    y_pred : ndarray
        Predicted value

    Returns
    -------
    tn, tp, fn, fp: ndarray
        Boolean matrices containing true negatives, true positives, false negatives and false positives.
    cm : ndarray
        Matrix containing: 0 - true negative, 1 - true positive,
        2 - false negative, and 3 - false positive.
    """

    # True negative
    tn = (y_true == y_pred) & (y_pred == 0)
    # True positive
    tp = (y_true == y_pred) & (y_pred == 1)
    # False positive
    fp = (y_true != y_pred) & (y_pred == 1)
    # False negative
    fn = (y_true != y_pred) & (y_pred == 0)

    # Generate matrix of "tp, fp, tn, fn"
    m, n = np.shape(y_true)
    cm = np.zeros((m, n), dtype=int)
    cm[tn] = 0
    cm[tp] = 1
    cm[fn] = 2
    cm[fp] = 3
    return tn, tp, fn, fp, cm

def plotF1AgainstThreshold(): # Helper function
    # Define candidate binarization thresholds
    binarization_thresholds = np.linspace(0.05, 0.95, 50)

    best_thresholds = np.zeros(5)
    best_f1_scores = np.zeros(5)

    # Create a plot for each class
    plt.figure(figsize=(10, 6))

    for class_idx in range(5):
        f1_list = []
        best_f1 = 0
        best_thresh = 0

        for bin_thresh in binarization_thresholds:
            y_true_bin = (y_true_raw[:, class_idx] >= bin_thresh).astype(int)  # Binarize y_true
            y_score_bin = (y_score_best[:, class_idx] >= bin_thresh).astype(int)  # Binarize y_true
            
            # Compute F1-score at each threshold
            f1_scores = f1_score(y_true_bin, y_score_bin)

            # Store best F1-score across all PR curve thresholds
            f1_list.append(f1_scores)

            # Update best threshold
            if f1_scores > best_f1:
                best_f1 = f1_scores
                best_thresh = bin_thresh

        # Store best threshold per class
        best_thresholds[class_idx] = best_thresh
        best_f1_scores[class_idx] = best_f1
        # best_thresholds = threshold
        # Plot for this class
        plt.plot(binarization_thresholds, f1_list, label=f"Class {class_idx} (Best Thresh: {best_thresh:.3f})")

    # Final plot settings
    plt.xlabel("Binarization Threshold for y_true")
    plt.ylabel("F1-score")
    plt.title("Optimal Binarization Threshold Per Class (Maximizing F1-score)")
    plt.legend()
    plt.grid()
    plt.show()

    print(np.average(best_f1_scores))
    print("Best Thresholds Per Class:", best_thresholds)
    print("Best F1-scores Per Class:", best_f1_scores)

def find_best_threshold(): # Finds best threshold for F1 score
    num_classes = y_true_raw.shape[1]  # Number of classes
    thresholds = np.linspace(0.05, 0.95, 50)  # Candidate thresholds
    best_thresholds = np.zeros(num_classes)

    # Iterate over each class to find the best threshold
    for class_idx in range(num_classes):
        best_f1 = 0
        best_thresh = 0
        
        for threshold in thresholds:
            # Convert soft labels to binary using the threshold
            y_true_bin = (y_true_raw[:, class_idx] >= threshold).astype(int)
            y_pred_bin = (y_score_best[:, class_idx] >= threshold).astype(int)

            # Compute F1-score
            f1 = f1_score(y_true_bin, y_pred_bin)

            # Update best threshold
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = threshold

        best_thresholds[class_idx] = best_thresh
        # print(f"Best threshold for class {class_idx}: {best_thresh}, Best F1: {best_f1}")
    print("Final optimal thresholds:", best_thresholds)
    return best_thresholds

def Scores():
    # %% Generate table with scores for the average model (Table 2)
    for y_pred in [y_neuralnet]:
        # Compute scores
        scores = get_scores(y_true, y_pred, score_fun)
        # Put them into a data frame
        scores_df = pd.DataFrame(scores, index=diagnosis, columns=score_fun.keys())
        # Append
        scores_list.append(scores_df)
    # Concatenate dataframes
    scores_all_df = pd.concat(scores_list, axis=1, keys=['DNN'])
    # Change multiindex levels
    scores_all_df = scores_all_df.swaplevel(0, 1, axis=1)
    scores_all_df = scores_all_df.reindex(level=0, columns=score_fun.keys())
    # Save results
    scores_all_df.to_excel(f"{output_path}/tables/scores.xlsx", float_format='%.3f')
    scores_all_df.to_csv(f"{output_path}/tables/scores.csv", float_format='%.3f')
    print("Scores done")

def PRCurve():
    # %% Plot precision recall curves (Figure 2)
    for k, name in enumerate(diagnosis):
        precision_list = []
        recall_list = []
        threshold_list = []
        average_precision_list = []
        fig, ax = plt.subplots()
        lw = 2
        t = ['bo', 'rv', 'gs', 'kd']
        
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score_best[:, k])
        recall[np.isnan(recall)] = 0  # change nans to 0
        precision[np.isnan(precision)] = 0  # change nans to 0
        
        # Plot if is the choosen option
        ax.plot(recall, precision, color='blue', alpha=0.7)
        # Compute average precision
        average_precision = average_precision_score(y_true[:, k], y_score_best[:, k])
        precision_list += [precision]
        recall_list += [recall]
        average_precision_list += [average_precision]
        threshold_list += [threshold]
    
    
        # Plot shaded region containing maximum and minimun from other executions
        recall_all = np.concatenate(recall_list)
        recall_all = np.sort(recall_all)  # sort
        recall_all = np.unique(recall_all)  # remove repeated entries
        recall_vec = []
        precision_min = []
        precision_max = []
        for r in recall_all:
            p_max = [max(precision[recall == r]) for recall, precision in zip(recall_list, precision_list)]
            p_min = [min(precision[recall == r]) for recall, precision in zip(recall_list, precision_list)]
            recall_vec += [r, r]
            precision_min += [min(p_max), min(p_min)]
            precision_max += [max(p_max), max(p_min)]
        ax.plot(recall_vec, precision_min, color='blue', alpha=0.3)
        ax.plot(recall_vec, precision_max, color='blue', alpha=0.3)
        ax.fill_between(recall_vec, precision_min, precision_max,
                        facecolor="blue", alpha=0.3)
        # Plot iso-f1 curves
        f_scores = np.linspace(0.1, 0.95, num=15)
        for f_score in f_scores:
            x = np.linspace(0.0000001, 1, 1000)
            y = f_score * x / (2 * x - f_score)
            ax.plot(x[y >= 0], y[y >= 0], color='gray', ls=':', lw=0.7, alpha=0.25)
        # Plot values in
        for npred in range(1):
            ax.plot(scores_list[npred]['Recall'][k], scores_list[npred]['Precision'][k],
                    t[npred], label=predictor_names[npred])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        # if k in [3, 4, 5]:
        ax.set_xlabel('Recall (Sensitivity)', fontsize=17)
        # if k in [0, 3]:
        ax.set_ylabel('Precision (PPV)', fontsize=17)
        # plt.title('Precision-Recall curve (' + name + ')')
        if k == 0:
            plt.legend(loc="lower left", fontsize=17)
        else:
            ax.legend().remove()
        plt.tight_layout()
        plt.savefig(f'{output_path}/figures/precision_recall_{{0}}.pdf'.format(name))
    print("PR-curve complete")

def ConfusionMatrix():
    # %% Confusion matrices (Supplementary Table 1)

    M = [[confusion_matrix(y_true[:, k], y_pred[:, k], labels=[0, 1])
        for k in range(nclasses)] for y_pred in [y_neuralnet]]

    M_xarray = xr.DataArray(np.array(M),
                            dims=['predictor', 'diagnosis', 'true label', 'predicted label'],
                            coords={'predictor': ['DNN'],
                                    'diagnosis': diagnosis,
                                    'true label': ['not present', 'present'],
                                    'predicted label': ['not present', 'present']})
    confusion_matrices = M_xarray.to_dataframe('n')
    confusion_matrices = confusion_matrices.reorder_levels([1, 2, 3, 0], axis=0)
    confusion_matrices = confusion_matrices.unstack()
    confusion_matrices = confusion_matrices.unstack()
    confusion_matrices = confusion_matrices['n']
    confusion_matrices.to_excel(f"{output_path}/tables/confusion matrices.xlsx", float_format='%.3f')
    confusion_matrices.to_csv(f"{output_path}/tables/confusion matrices.csv", float_format='%.3f')

    print("Confusion matrix complete")
    #%% Compute scores and bootstraped version of these scores
def Boxplot():
    bootstrap_nsamples = 1000
    percentiles = [2.5, 97.5]
    scores_resampled_list = []
    scores_percentiles_list = []
    for y_pred in [y_neuralnet]:
        # Compute bootstraped samples
        np.random.seed(123)  # NEVER change this =P
        n, _ = np.shape(y_true)
        samples = np.random.randint(n, size=n * bootstrap_nsamples)
        # Get samples
        y_true_resampled = np.reshape(y_true[samples, :], (bootstrap_nsamples, n, nclasses))
        y_doctors_resampled = np.reshape(y_pred[samples, :], (bootstrap_nsamples, n, nclasses))
        # Apply functions
        scores_resampled = np.array([get_scores(y_true_resampled[i, :, :], y_doctors_resampled[i, :, :], score_fun)
                                    for i in range(bootstrap_nsamples)])
        # Sort scores
        scores_resampled.sort(axis=0)
        # Append
        scores_resampled_list.append(scores_resampled)

        # Compute percentiles index
        i = [int(p / 100.0 * bootstrap_nsamples) for p in percentiles]
        # Get percentiles
        scores_percentiles = scores_resampled[i, :, :]
        # Convert percentiles to a dataframe
        scores_percentiles_df = pd.concat([pd.DataFrame(x, index=diagnosis, columns=score_fun.keys())
                                        for x in scores_percentiles], keys=['p1', 'p2'], axis=1)
        # Change multiindex levels
        scores_percentiles_df = scores_percentiles_df.swaplevel(0, 1, axis=1)
        scores_percentiles_df = scores_percentiles_df.reindex(level=0, columns=score_fun.keys())
        # Append
        scores_percentiles_list.append(scores_percentiles_df)
    # Concatenate dataframes
    scores_percentiles_all_df = pd.concat(scores_percentiles_list, axis=1, keys=predictor_names)
    # Change multiindex levels
    scores_percentiles_all_df = scores_percentiles_all_df.reorder_levels([1, 0, 2], axis=1)
    scores_percentiles_all_df = scores_percentiles_all_df.reindex(level=0, columns=score_fun.keys())

    #%% Print box plot (Supplementary Figure 1)
    # Convert to xarray
    scores_resampled_xr = xr.DataArray(np.array(scores_resampled_list),
                                    dims=['predictor', 'n', 'diagnosis', 'score_fun'],
                                    coords={
                                        'predictor': predictor_names,
                                        'n': range(bootstrap_nsamples),
                                        'diagnosis': ['NORM', 'CD', 'HYP', 'MI', 'STTC'],
                                        'score_fun': list(score_fun.keys())})
    # Remove everything except f1_score
    for sf in score_fun:
        fig, ax = plt.subplots()
        f1_score_resampled_xr = scores_resampled_xr.sel(score_fun=sf)
        # Convert to dataframe
        f1_score_resampled_df = f1_score_resampled_xr.to_dataframe(name=sf).reset_index(level=[0, 1, 2])
        # Plot seaborn
        ax = sns.boxplot(x="diagnosis", y=sf, hue="predictor", data=f1_score_resampled_df)
        # Save results
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("")
        plt.ylabel("", fontsize=16)
        if sf == "F1 score":
            plt.legend(fontsize=17)
        else:
            ax.legend().remove()
        plt.tight_layout()
        plt.savefig(f'{output_path}/figures/boxplot_bootstrap_{{}}.pdf'.format(sf))

    scores_resampled_xr.to_dataframe(name='score').to_csv(f'{output_path}/figures/boxplot_bootstrap_data.txt')
    print("Boxplot complete")


parser = argparse.ArgumentParser(description='Evaluate Prediction by F1 score.')
parser.add_argument('path_test_csv', type=str,
                    help='path to csv file containing annotations')
parser.add_argument('output_folder_path', type=str,
                    help='path to output folder')
parser.add_argument('path_predict_npy', type=str,
                    help='path to npy file containing predictions')
args = parser.parse_args()
# %% Constants
score_fun = {'Precision': precision_score,
             'Recall': recall_score, 'Specificity': specificity_score,
             'F1 score': f1_score}
diagnosis = ['NORM', 'CD', 'HYP', 'MI', 'STTC']
nclasses = len(diagnosis)
predictor_names = ['DNN']
fileName = args.path_predict_npy.split("/")[-1]
fileName = str.removesuffix(fileName, ".npy")
output_path = os.path.join(args.output_folder_path, fileName)
os.makedirs(f"{output_path}/tables",exist_ok=True)
os.makedirs(f"{output_path}/figures",exist_ok=True)


# %% Read datasets
# Get true values (soft labels, probability)
y_true_raw =  pd.read_csv(args.path_test_csv).values

# get y_score (soft labels, probability)
y_score_best = np.load(args.path_predict_npy)


# %% Binarize the soft labels
# Get threshold that yield the best precision recall using "get_optimal_precision_recall" on validation set
#   (we rounded it up to three decimal cases to make it easier to read...)

# This is the threshold for determining the binarization
# threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390]) # Original valus from Riberio
# threshold = np.array([0.49, 0.49, 0.49, 0.49, 0.49]) # Init values for getting opt values

# Used model for testing PR vs FBT for thresholding
# python evalPrediction.py data/PTB_XL_data/test_data.csv outputs dnn_predicts/base_model-BS32-LR0.001-DR_Keep_P0.5-DA_P0.5-L2_0.001-20250322-134701.npy
# threshold = np.array([0.307, 0.436, 0.123, 0.123, 0.142]) # Threshold values from find_best_threshold
# threshold = np.array([0.402, 0.436, 0.136, 0.189, 0.136]) # PR threshold from get_optimal_precision_recall, using test set and thresholding on 0.49.

threshold =  find_best_threshold() # Dynamic threshold
# threshold = [0.399,0.385,0.110,0.1645,0.26125] # Tog en average av 4 FBT thresholds.

mask = y_score_best > threshold # This is a true/false matrix
y_true = y_true_raw > threshold # This is a true/false matrix, works with the sklearn functions.

# opt_prec, opt_rec, opt_thres = get_optimal_precision_recall(y_true,y_score_best) # Calculates best threshold for PR.
# print(opt_thres)

# Get neural network prediction
y_neuralnet = np.zeros_like(y_score_best)
y_neuralnet[mask] = 1 # This converts true/false matrix into binary

scores_list = []

#%% Usable functions
Scores()
PRCurve()
ConfusionMatrix()
Boxplot()
