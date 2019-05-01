import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

startTime = -1.5  # earliest time that an event can start to be included
endTime = 1.5  # latest time that an event can start to be included
fmin = 15  # NOTE: this timing behaviour is different than the prestim code
fmax = 30

saveFig = True

dataDir = '/home/timb/camcan/spectralEvents/'
figureDir = '/home/timb/camcan/figures/spectralEvents/MEG0711'

csvFile = os.path.join(dataDir, 'MEG0711_spectral_events_-1.0to1.0s.csv')
df = pd.read_csv(csvFile)

eventChars = ['Event Duration', 'Event Offset Time', 'Event Onset Time', 'Frequency Span',
       'Lower Frequency Bound', 'Normalized Peak Power', 'Peak Frequency', 'Peak Power', 'Peak Time',
       'Trial', 'Upper Frequency Bound']

df1 = df[df['Outlier Event']]
# Freq range of interest
df2 = df1.drop(df1[df1['Lower Frequency Bound'] < fmin].index)
df3 = df2.drop(df2[df2['Upper Frequency Bound'] > fmax].index)
df4 = df3.drop(df3[df3['Event Onset Time'] > endTime].index)
newDf = df4.drop(df4[df4['Event Onset Time'] < startTime].index)

print('Including ' + str(len(newDf)) + ' events')

for e in eventChars:

    if saveFig:
        lbl = e.replace(" ", "")
        figureFile = os.path.join(figureDir, "".join([lbl, "_distplot.pdf"]))
        ax = sns.distplot(newDf[e])
        fig = ax.get_figure()
        plt.title('Distribution of Event Characteristic - 598 Subjects')
        fig.savefig(figureFile)
        plt.close()
    else:
        ax = sns.distplot(newDf[e])
        fig = ax.get_figure()
        plt.title('Distribution of Event Characteristic - 598 Subjects')
        plt.show()


if saveFig:
    figureFile = os.path.join(figureDir, "peakTimePeakFreq_jointplot.pdf")
    fig = sns.jointplot(data=df, x='Peak Time', y="Peak Frequency", kind='hex')
    fig.savefig(figureFile)
    plt.close()
else:
    fig = sns.jointplot(data=df, x='Peak Time', y="Peak Frequency", kind='hex')
    plt.show()