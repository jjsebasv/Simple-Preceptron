import plotly.offline as offline
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import pdb
from scipy.interpolate import griddata
import glob
import os

path = "points/*.csv"

oranges = [
    [0, '#fc210c'],
    [0.1, '#fa3112'],
    [0.2, '#f94018'],
    [0.3, '#f74e1e'],
    [0.4, '#f65b24'],
    [0.5, '#f4682a'],
    [0.6, '#f3742f'],
    [0.7, '#f18035'],
    [0.8, '#f08a3a'],
    [0.9, '#ee9540'],
    [1.0, '#ed9e45']
]


blues = [
    [0, '#384ee4'],
    [0.1, '#3a57e3'],
    [0.2, '#3b61e2'],
    [0.3, '#3c6be1'],
    [0.4, '#3e74e1'],
    [0.5, '#3f7ee0'],
    [0.6, '#4088df'],
    [0.7, '#4192de'],
    [0.8, '#429bde'],
    [0.9, '#44a5dd'],
    [1.0, '#45afdc']
]

for fname in glob.glob(path):
    plot_name = os.path.basename(fname)
    print plot_name

    #Load CSV
    df = pd.read_csv(fname, header=None, delimiter=';')
    df.head()
    data = []

    s = np.linspace(df[0].min(), df[0].max(), len(df[0].unique()))
    t = np.linspace(df[1].min(), df[1].max(), len(df[1].unique()))
    tGrid, sGrid = np.meshgrid(s, t)

    x = tGrid
    y = sGrid
    z_1 = griddata((df[0], df[1]), df[2], (tGrid, sGrid), method='cubic')
    z_2 = griddata((df[0], df[1]), df[3], (tGrid, sGrid), method='cubic')

    surface_1 = go.Surface(x=x, y=y, z=z_1, colorscale=blues)
    surface_2 = go.Surface(x=x, y=y, z=z_2, colorscale=oranges)

    data.append(surface_1)
    data.append(surface_2)

    layout = go.Layout(
        width=1200,
        height=800,
        autosize=False,
        title=fname,
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
        )
    )


    fig = go.Figure(data=data, layout=layout)
    offline.plot(fig, filename='surface_plots/' + plot_name + '.html')
