import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import sys
from matplotlib.image import imread
from tempfile import NamedTemporaryFile


def get_size(fig, dpi=100):
    fig.savefig('temp/file.png', bbox_inches='tight', dpi=dpi)
    height, width, _channels = imread('temp/file.png').shape
    return width / dpi, height / dpi


def set_size(fig, size, dpi=100, eps=1e-2, give_up=2, min_size_px=10):
    target_width, target_height = size
    set_width, set_height = target_width, target_height # reasonable starting point
    deltas = [] # how far we have
    while True:
        fig.set_size_inches([set_width, set_height])
        actual_width, actual_height = get_size(fig, dpi=dpi)
        set_width *= target_width / actual_width
        set_height *= target_height / actual_height
        deltas.append(abs(actual_width - target_width) + abs(actual_height - target_height))
        if deltas[-1] < eps:
            return True
        if len(deltas) > give_up and sorted(deltas[-give_up:]) == deltas[-give_up:]:
            return False
        if set_width * dpi < min_size_px or set_height * dpi < min_size_px:
            return False


def plotter(results_filename, output_dir, name, argv=None):
    output_dir = os.path.join(output_dir, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sns.set(style="darkgrid")
    df = pd.read_csv(results_filename)
    df = df[df['terminal'] == True]
    x = ['episode']
    y_reward = ['total_reward']
    reward_df = pd.melt(df[x + y_reward], id_vars=x, value_vars=y_reward)
    dpi = 150
    fig, (ax1) = plt.subplots(1, sharex=True, facecolor='w', edgecolor='k', figsize=(1920/dpi, 1080/dpi),dpi=dpi)
    fig.suptitle(name + ":\n" + "Total Reward Per Episode")
    sns.lineplot(x='episode', y='value', hue='variable', data=reward_df, ax=ax1)
    ax1.set_ylim(-1000, 100000)
    ax1.legend().set_visible(False)
    ax1.set_ylabel('Total Reward')
    plt.xlabel('Episode')
    plt.savefig(os.path.join(output_dir, 'reward.png'))
    plt.close()



    sns.set(style="darkgrid")
    df = pd.read_csv(results_filename)
    df = df[df['episode'] == 0]
    df = df[df['time'] <= 86400]


    x = ['time']
    y_temp = [
              'basement_temperature',
              'main_temperature',
              'attic_temperature']
    y_outsideTemp = [
        'air_temperature',
        'ground_temperature'
    ]
    y_action = ['action']

    temperature_df = pd.melt(df[x + y_temp], id_vars=x, value_vars=y_temp)
    outside_temperature_df = pd.melt(df[x + y_outsideTemp],id_vars=x, value_vars=y_outsideTemp)
    action_df = pd.melt(df[y_action + x], id_vars=x, value_vars=y_action)

    dpi = 150
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, facecolor='w', edgecolor='k', figsize=(1920/dpi, 1080/dpi),dpi=dpi)
    fig.suptitle(name + ":\n" + "Temperatures over a day")



    sns.scatterplot(data=action_df, x='time', y='value', ax=ax1)
    ax1.set_ylim(0,5)
    ax1.legend().set_visible(False)
    ax1.set_ylabel('Actions Taken')

    sns.lineplot(x='time', y='value', hue='variable', data=temperature_df, ax=ax2)
    ax2.set_ylim(10, 30)
    ax2.legend().set_visible(True)
    ax2.set_ylabel('Home \nTemperatures (C)')


    sns.lineplot(x='time', y='value', hue='variable', data=outside_temperature_df, ax=ax3)
    ax3.set_ylim(-20, 50)
    ax3.legend().set_visible(True)
    ax3.set_ylabel('Outside Environment \nTemperatures (C)')



    plt.xlabel('Time (s)')
    plt.savefig(os.path.join(output_dir, 'temperatures.png'))
    plt.close()


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp_ylim_lower', type=float, default=-5)
    parser.add_argument('--temp_ylim_upper', type=float, default=40)
    parser.add_argument('--reward_ylim_lower', type=float, default=-10)
    parser.add_argument('--reward_ylim_upper', type=float, default=250)
    parser.add_argument('--xlim_left', type=float, default=0)
    parser.add_argument('--xlim_right', type=float, default=672*900)
    args = parser.parse_args(argv)
    return vars(args)


def parse_config_file(config_file_name):
    with open(config_file_name, 'r') as config_file:
        argv = config_file.read().replace('\n', '').split(' ')
    return parse_args(argv)


def __main__(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('output_dir')
    parser.add_argument('episode_upper', type=int)
    parser.add_argument('episode_lower', type=int, default=0)
    parser.add_argument('inputResultsFile', type=str, default="inputs/oldResults.csv")
    args = parser.parse_args(argv)
    vargs = vars(args)
    plotter(
            os.path.join(vargs['inputResultsFile']),
            vargs['output_dir'],
            vargs['name'])


# if __name__ == '__main__':
#     __main__(sys.argv[1:])



name="Non-Learning, Updated Model from Example"
outputDir = "outputs"
episodeUpper = "250"
episodeLower = "0"
inputfile = "inputs/NonLearningModelFromExampleQuickresults.csv"
argv = [name, outputDir, episodeUpper, episodeLower, inputfile]
__main__(argv)


name="Learning, Updated Model Long Form"
outputDir = "outputs"
episodeUpper = "15000"
episodeLower = "0"
inputfile = "inputs/NormalNewVeryLongresults.csv"
argv = [name, outputDir, episodeUpper, episodeLower, inputfile]
__main__(argv)


name="Learning, Non-Updated Model Long Form"
outputDir = "outputs"
episodeUpper = "15000"
episodeLower = "0"
inputfile = "inputs/NormalOldVeryLongresults.csv"
argv = [name, outputDir, episodeUpper, episodeLower, inputfile]
__main__(argv)
