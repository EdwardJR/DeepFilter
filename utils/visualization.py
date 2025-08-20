#============================================================
#
#  Deep Learning BLW Filtering
#  Data Visualization
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import os

# Global output directory - can be set by calling scripts
output_dir = "output_images"

def check_and_create_dir(directory):
    """Creates a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_violinplots(np_data, description, ylabel, log):
    global output_dir
    check_and_create_dir(output_dir)
    
    # Process the results and store in Panda objects
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)

    # Set up the matplotlib figure
    f, ax = plt.subplots()
    sns.set(style="whitegrid")
    ax = sns.violinplot(data=pd_df, palette="Set3", bw=.2, cut=1, linewidth=1)

    if log:
        ax.set_yscale("log")

    ax.set(xlabel='Models/Methods', ylabel=ylabel)
    ax = sns.despine(left=True, bottom=True)
    
    plt.savefig(os.path.join(output_dir, f'violinplot_{ylabel.replace(" ", "_")}.png'))
    plt.show()


def generate_barplot(np_data, description, ylabel, log):
    global output_dir
    check_and_create_dir(output_dir)
    
    # Process the results and store in Panda objects
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)

    # Set up the matplotlib figure
    f, ax = plt.subplots()
    sns.set(style="whitegrid")
    ax = sns.barplot(data=pd_df)

    if log:
        ax.set_yscale("log")

    ax.set(xlabel='Models/Methods', ylabel=ylabel)
    ax = sns.despine(left=True, bottom=True)

    plt.savefig(os.path.join(output_dir, f'barplot_{ylabel.replace(" ", "_")}.png'))
    plt.show()


def generate_boxplot(np_data, description, ylabel, log):
    global output_dir
    check_and_create_dir(output_dir)
    
    # Process the results and store in Panda objects
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)

    # Set up the matplotlib figure
    f, ax = plt.subplots()
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=pd_df)

    if log:
        ax.set_yscale("log")

    ax.set(xlabel='Models/Methods', ylabel=ylabel)
    ax = sns.despine(left=True, bottom=True)

    plt.savefig(os.path.join(output_dir, f'boxplot_{ylabel.replace(" ", "_")}.png'))
    plt.show()


def generate_hboxplot(np_data, description, ylabel, log, set_x_axis_size=None):
    global output_dir
    check_and_create_dir(output_dir)
    
    # Process the results and store in Panda objects
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)

    # Set up the matplotlib figure
    sns.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(15, 6))
    ax = sns.boxplot(data=pd_df, orient="h", width=0.4)

    if log:
        ax.set_xscale("log")

    if set_x_axis_size != None:
        ax.set_xlim(set_x_axis_size)

    ax.set(ylabel='Models/Methods', xlabel=ylabel)
    ax = sns.despine(left=True, bottom=True)
    
    plt.savefig(os.path.join(output_dir, f'hboxplot_{ylabel.replace(" ", "_")}.png'))
    plt.show()


def ecg_view(ecg, ecg_blw, ecg_dl, ecg_f, signal_name=None, beat_no=None):
    global output_dir
    check_and_create_dir(output_dir)

    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(ecg_blw, 'k', label='ECG + BLW')
    plt.plot(ecg, 'g', label='ECG orig')
    plt.plot(ecg_dl, 'b', label='ECG DL Filtered')
    plt.plot(ecg_f, 'r', label='ECG IIR Filtered')
    plt.grid(True)

    plt.ylabel('au')
    plt.xlabel('samples')

    leg = ax.legend()
    
    if signal_name is not None and beat_no is not None:
        title = f'Signal {signal_name} beat {beat_no}'
        filename = f'ecg_view_{signal_name}_beat_{beat_no}.png'
    else:
        title = 'ECG signal for comparison'
        filename = 'ecg_view_default.png'
    
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()


def ecg_view_diff(ecg, ecg_blw, ecg_dl, ecg_f, signal_name=None, beat_no=None):
    global output_dir
    check_and_create_dir(output_dir)

    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(ecg, 'g', label='ECG orig')
    plt.plot(ecg_dl, 'b', label='ECG DL Filtered')
    plt.plot(ecg_f, 'r', label='ECG IIR Filtered')
    plt.plot(ecg - ecg_dl, color='#0099ff', lw=3, label='Difference ECG - DL Filter')
    plt.plot(ecg - ecg_f, color='#cb828d', lw=3, label='Difference ECG - IIR Filter')
    plt.grid(True)

    plt.ylabel('Amplitude (au)')
    plt.xlabel('samples')

    leg = ax.legend()

    if signal_name is not None and beat_no is not None:
        title = f'Signal Diff {signal_name} beat {beat_no}'
        filename = f'ecg_view_diff_{signal_name}_beat_{beat_no}.png'
    else:
        title = 'ECG signal for comparison'
        filename = 'ecg_view_diff_default.png'

    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()


def generate_table(metrics, metric_values, Exp_names):
    # Print tabular results in the console, in a pretty way
    tb = PrettyTable()
    ind = 0

    for exp_name in Exp_names:

        tb.field_names = ['Method/Model'] + metrics

        tb_row = []
        tb_row.append(exp_name)

        for metric in metric_values:   # metric_values[metric][model][beat]
            m_mean = np.mean(metric[ind])
            m_std = np.std(metric[ind])
            tb_row.append('{:.3f}'.format(m_mean) + ' (' + '{:.3f}'.format(m_std) + ')')

        tb.add_row(tb_row)
        ind += 1

    print(tb)


def generate_table_time(column_names, all_values, Exp_names, gpu=True):
    # Print tabular results in the console, in a pretty way

    # The FIR and IIR are the last on all_values
    # We need circular shift them to the right
    all_values[0] = all_values[0][-2::] + all_values[0][0:-2]
    all_values[1] = all_values[1][-2::] + all_values[1][0:-2]

    tb = PrettyTable()
    ind = 0

    if gpu:
        device = 'GPU'
    else:
        device = 'CPU'

    for exp_name in Exp_names:
        tb.field_names = ['Method/Model'] + [column_names[0] + '(' + device + ') h:m:s:ms'] + [
            column_names[1] + '(' + device + ') h:m:s:ms']

        tb_row = []
        tb_row.append(exp_name)
        tb_row.append(all_values[0][ind])
        tb_row.append(all_values[1][ind])

        tb.add_row(tb_row)

        ind += 1

    print(tb)

    if gpu:
        print('* For FIR and IIR Filters is CPU since scipy filters are CPU based implementations')
