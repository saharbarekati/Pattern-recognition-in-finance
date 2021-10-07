#imports
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.kernel_regression import KernelReg
from collections import defaultdict
import csv

#find min and max(high and low)

def plot_extrema(data):
    x = data['High']
    x = np.float64(x.to_numpy())
    print(argrelextrema(x, np.greater))

    y = data['Low']
    y = np.float64(y.to_numpy())
    print(argrelextrema(y, np.less))

    local_maxs = argrelextrema(data['High'].values, np.greater)[0]
    local_mins = argrelextrema(data['Low'].values, np.less)[0]
    print("local_maxs")
    print(local_maxs)
    print(local_mins)

    highs = data.iloc[local_maxs, :]
    lows = data.iloc[local_mins, :]

    print(highs)
    print(lows)

    highslows = pd.concat([highs, lows])
    print(highslows['High'].values)


def find_extrema(s, bw='cv_ls'):
    """
    Input:
        s: prices as pd.series
        bw: bandwith as str or array like
    Returns:
        prices: with 0-based index as pd.series
        extrema: extrema of prices as pd.series
        smoothed_prices: smoothed prices using kernel regression as pd.series
        smoothed_extrema: extrema of smoothed_prices as pd.series
    """
    # Copy series so we can replace index and perform non-parametric
    # kernel regression.
    prices = s.copy()
    prices = prices.reset_index()

    prices = prices['High']

    kr = KernelReg([prices.values], [prices.index.to_numpy()],
                   var_type='c', bw=bw)
    f = kr.fit([prices.index])

    # Use smoothed prices to determine local minima and maxima
    smooth_prices = pd.Series(data=f[0], index=prices.index)
    smooth_local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    smooth_local_min = argrelextrema(smooth_prices.values, np.less)[0]
    local_max_min = np.sort(np.concatenate(
        [smooth_local_max, smooth_local_min]))
    smooth_extrema = smooth_prices.loc[local_max_min]

    # Iterate over extrema arrays returning datetime of passed
    # prices array. Uses idxmax and idxmin to window for local extrema.
    price_local_max_dt = []
    for i in smooth_local_max:
        if (i > 1) and (i < len(prices)-1):
            price_local_max_dt.append(prices.iloc[i-2:i+2].idxmax())

    price_local_min_dt = []
    for i in smooth_local_min:
        if (i > 1) and (i < len(prices)-1):
            price_local_min_dt.append(prices.iloc[i-2:i+2].idxmin())

    maxima = pd.Series(prices.loc[price_local_max_dt])
    minima = pd.Series(prices.loc[price_local_min_dt])
    extrema = pd.concat([maxima, minima]).sort_index()
    prices = pd.concat([prices, s['Date']], axis=1)
    extrema = pd.concat([extrema, s['Date']], axis=1)
    smooth_prices = pd.concat([smooth_prices, s['Date']], axis=1)
    smooth_extrema = pd.concat([smooth_extrema, s['Date']], axis=1)

    # Return series for each with bar as index
    return prices, extrema, smooth_prices, smooth_extrema


def plot_window(prices, extrema, smooth_prices, smooth_extrema, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    prices.Date = pd.to_datetime(prices['Date'], format='%Y%m%d')
    extrema.Date = pd.to_datetime(extrema.Date, format='%Y%m%d')
    smooth_prices.Date = pd.to_datetime(smooth_prices.Date, format='%Y%m%d')
    smooth_extrema.Date = pd.to_datetime(smooth_extrema.Date, format='%Y%m%d')

    prices.plot(x='Date', y='High', ax=ax, color='dodgerblue')
    ax.scatter(extrema.Date, extrema.High, color='red')
    smooth_prices.plot(x='Date', y=0, ax=ax, color='lightgrey')
    ax.scatter(smooth_extrema.Date, smooth_extrema[0], color='lightgrey')

#pattern identification

def find_patterns(extrema, max_bars=35):
    """
    Input:
        extrema as pd.series with bar number as index
        max_bars: max bars for pattern to play out
    Returns:
        patterns: patterns as a defaultdict list of tuples
        containing the start and end bar of the pattern
    """
    patterns = defaultdict(list)

    # Need to start at five extrema for pattern generation
    # per 5 steps
    for i in range(5, len(extrema)):
        window = extrema.iloc[i-5:i]

        # A pattern must play out within max_bars (default 35)
        if (window.index[-1] - window.index[0]) > max_bars:
            continue

        # Using the notation from the paper to avoid mistakes
        e1 = window.iloc[0]
        e2 = window.iloc[1]
        e3 = window.iloc[2]
        e4 = window.iloc[3]
        e5 = window.iloc[4]

        # top triangle

        # Head and Shoulders

        if (e1 > e2) and (e3 > e1) and (e3 > e5) and \
            (abs(e1 - e5) <= 0.03*np.mean([e1, e5])) and \
                (abs(e2 - e4) <= 0.03*np.mean([e1, e5])):
            patterns['HS'].append((window.index[0], window.index[-1]))


    return patterns


if __name__ == '__main__':
    data = pd.read_csv("./DRZK1-a.csv")

    fi = open("./derazak.csv","w")
    writer=csv.writer(fi)
    # writer.writerow(["name","start","end"])
    prices, extrema, smooth_prices, smooth_extrema = find_extrema(data, bw=[1.5])
    patterns = find_patterns(extrema['High'].dropna())
    for name, pattern_periods in patterns.items():
        rows = int(np.ceil(len(pattern_periods)/2))
        f, axes = plt.subplots(rows, 2, figsize=(20, 5*rows))
        for i in range(rows):
            axes[i, 0].set_title("title"+str(2*i))
            axes[i, 1].set_title("title"+str(2*i+1))
        axes = axes.flatten()
        i = 0
        print(pattern_periods)
        for start, end in pattern_periods:
            s = prices.index[start-1]
            e = prices.index[end+1]

            # print('calling plot window')
            plot_window(prices[s:e], extrema.loc[s:e],
                        smooth_prices[s:e],
                        smooth_extrema.loc[s:e], ax=axes[i])
            axes[i].set_title(name)
            i += 1
            Date = pd.to_datetime(prices[s:e]['Date'], format='%Y%m%d')

            writer.writerow([name,Date.max().strftime(' %Y/%m/%d'),Date.min().strftime(' %Y/%m/%d')])

        plt.show()

    fi.close()