from collections import defaultdict
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics


# Useful to uniquely identify metrics across functions
class Metric(Enum):
    ACC = 'Accuracy'
    SEN = 'Sensitivity'
    SPE = 'Specificity'
    PPV = 'PPV/Precision'  # precision / positive predictive value
    NPV = 'NPV'  # negative predictive value
    PEOPLE = 'People included'


def plot_mean_std(joined_df: pd.DataFrame, diagnosis_arr: list) -> None:
    plt.subplots(figsize=(10, 5))

    for diagnosis in diagnosis_arr:
        tmp_df = joined_df[joined_df.diagnosis == diagnosis]

        plt.scatter(tmp_df['std'], tmp_df['mean'], label=diagnosis)

    plt.legend()
    plt.xlabel('Std')
    plt.ylabel('Mean')
    plt.show()
    plt.close()


def plot_all_roc_curves(mcdrop_df: pd.DataFrame, singlpass_df: pd.DataFrame) -> None:
    def plot_roc_stuff(tmp_df, label, color):
        preds = tmp_df['mean']
        fpr, tpr, thresholds = metrics.roc_curve(tmp_df['diagnosis'], preds)
        roc_auc = metrics.auc(fpr, tpr)

        # Youden’s J statistic
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        print(f'Best treshold for {label}: {best_thresh}')

        plt.plot(fpr, tpr, 'b', label=f'AUC {label} = %0.2f' % roc_auc, color=color)
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label=f'Best {label}')

    plt.subplots(figsize=(20, 10))
    plt.title('Receiver Operating Characteristic')
    plot_roc_stuff(mcdrop_df, 'MC-Drop', 'orange')
    plot_roc_stuff(singlpass_df, 'Single Pass', 'green')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.close()


def plot_all_pr_curves(mcdrop_df, singlpass_df):
    def plot_pr_stuff(tmp_df, label, color):
        preds = tmp_df['mean']
        precision, recall, thresholds = metrics.precision_recall_curve(tmp_df['diagnosis'], preds)

        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        print(f'Best Threshold for {label}=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
        plt.scatter(recall[ix], precision[ix], marker='o', color='black', label=f'Best {label}')
        plt.plot(recall, precision, marker='.', label=f'{label}', color=color)

    plt.subplots(figsize=(20, 10))
    plt.title('PR-Curve')
    plot_pr_stuff(mcdrop_df, 'MC-Drop', 'orange')
    plot_pr_stuff(singlpass_df, 'Single Pass', 'green')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()
    plt.close()


def populate_arrs_for_df(df, metrics_dict, threshold=0.5):
    probs = df['mean'].copy().values
    probs[probs < threshold] = 0
    probs[probs >= threshold] = 1

    tn, fp, fn, tp = metrics.confusion_matrix(df['diagnosis'].values, probs).ravel()
    specificity = tn / (tn + fp)
    sensitivty = tp / (tp + fn)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    metrics_dict[Metric.ACC.name].append(metrics.accuracy_score(df['diagnosis'].values, probs))
    metrics_dict[Metric.SEN.name].append(sensitivty)
    metrics_dict[Metric.SPE.name].append(specificity)
    metrics_dict[Metric.PEOPLE.name].append(df.shape[0])
    metrics_dict[Metric.PPV.name].append(ppv)
    metrics_dict[Metric.NPV.name].append(npv)


def plot_across_metrics(x_vals_mc, x_vals_1, mcdrop_metrics, single_metrics, x_label):
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    for i, val in enumerate(Metric):
        if val.name == Metric.PEOPLE.name:
            continue
        axs[i].plot(x_vals_mc, mcdrop_metrics[val.name], label=f'{val.value} - MC Drop')
        axs[i].plot(x_vals_1, single_metrics[val.name], label=f'{val.value} - Single')

    for ax in axs:
        ax.set_xlabel(x_label)
        ax.set_ylabel('Performance achieved')
        ax.legend()

    plt.tight_layout()
    plt.show()


def print_title(title_str: str) -> None:
    print('#####################################################################################')
    print(f'################# {title_str}')
    print('#####################################################################################')


def plot_all_comparisons(joined_df: pd.DataFrame, single_pass: pd.DataFrame,
                         threshold: float, starting_num_people: int) -> None:
    print_title('MC-Drop with uncertainty thresholding')

    people_std_mcdrop_metrics = defaultdict(list)

    joined_df = joined_df.sort_values(by=['std'])

    for i in np.arange(starting_num_people, len(joined_df) + 1, 1):
        tmp_df = joined_df.iloc[:i, :]
        populate_arrs_for_df(tmp_df, people_std_mcdrop_metrics, threshold=threshold)

    # Plotting it
    fig, ax = plt.subplots(figsize=(15, 5))
    for val in Metric:
        if val.name == Metric.PEOPLE.name:
            continue
        ax.plot(people_std_mcdrop_metrics[Metric.PEOPLE.name], people_std_mcdrop_metrics[val.name], label=val.value)
    ax.set_xlabel(Metric.PEOPLE.value)
    ax.set_ylabel('Performance achieved - MC Drop')
    ax.legend()

    plt.show()

    ###############################
    print_title('MC-Drop vs Single-pass')

    joined_df = joined_df.sort_values(by='extremes')
    single_pass = single_pass.sort_values(by='extremes')

    people_delta_mcdrop_metrics, people_delta_1_metrics = defaultdict(list), defaultdict(list)

    for i in np.arange(starting_num_people, len(joined_df) + 1, 1):
        tmp_df = joined_df.iloc[:i, :]
        populate_arrs_for_df(tmp_df, people_delta_mcdrop_metrics, threshold=threshold)

        tmp_df = single_pass.iloc[:i, :]
        populate_arrs_for_df(tmp_df, people_delta_1_metrics, threshold=threshold)

    plot_across_metrics(people_delta_mcdrop_metrics[Metric.PEOPLE.name],
                        people_delta_1_metrics[Metric.PEOPLE.name],
                        people_delta_mcdrop_metrics,
                        people_delta_1_metrics,
                        'People included')

    ###############################
    delta_delta_mcdrop_metrics, delta_delta_1_metrics = defaultdict(list), defaultdict(list)

    # Finding a minimal delta to start the for loop
    ini_val = 0
    for delta_val in np.arange(0.01, 0.51, 0.001):
        t1_df = joined_df.loc[(joined_df['mean'] < delta_val) | (joined_df['mean'] > 1 - delta_val), :]
        t2_df = single_pass.loc[(single_pass['mean'] < delta_val) | (single_pass['mean'] > 1 - delta_val), :]
        if t1_df.shape[0] >= 4 and t2_df.shape[0] >= 4:
            ini_val = round(delta_val, 3)
            break

    # Now getting for different values of delta
    for delta_val in np.arange(ini_val, 0.51, 0.001):
        tmp_df = joined_df.loc[(joined_df['mean'] < delta_val) | (joined_df['mean'] > 1 - delta_val), :]
        populate_arrs_for_df(tmp_df, delta_delta_mcdrop_metrics, threshold=threshold)

        tmp_df = single_pass.loc[(single_pass['mean'] < delta_val) | (single_pass['mean'] > 1 - delta_val), :]
        populate_arrs_for_df(tmp_df, delta_delta_1_metrics, threshold=threshold)

    plot_across_metrics(np.arange(ini_val, 0.51, 0.001),
                        np.arange(ini_val, 0.51, 0.001),
                        delta_delta_mcdrop_metrics,
                        delta_delta_1_metrics,
                        'Extreme deltas used to filter')

    ###############################
    print_title('All 3 approaches together')

    for val in Metric:
        if val.name == Metric.PEOPLE.name:
            continue
        plt.subplots(figsize=(20, 5))
        plt.plot(people_delta_mcdrop_metrics[Metric.PEOPLE.name], people_delta_mcdrop_metrics[val.name], #'o-',
                 label='Considering delta - MC Drop')
        plt.plot(people_delta_1_metrics[Metric.PEOPLE.name], people_delta_1_metrics[val.name], #'|-',
                 label='Considering delta - Single')
        plt.plot(people_std_mcdrop_metrics[Metric.PEOPLE.name], people_std_mcdrop_metrics[val.name], #'x-',
                 label='Considering uncertainty')
        plt.xlabel('People included')
        plt.ylabel(val.value)
        plt.legend()
        plt.show()
