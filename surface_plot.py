import plotly.offline as offline
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import pdb
from scipy.interpolate import griddata
import glob
import os

path = "points/*.csv"

for fname in glob.glob(path):
    plot_name = os.path.basename(fname)

    #Load CSV
    df = pd.read_csv('points/points_8_1.csv', header=None, delimiter=';')
    df.head()
    data = []

    s = np.linspace(df[0].min(), df[0].max(), len(df[0].unique()))
    t = np.linspace(df[1].min(), df[1].max(), len(df[1].unique()))
    tGrid, sGrid = np.meshgrid(s, t)

    x = tGrid
    y = sGrid
    z_1 = griddata((df[0], df[1]), df[2], (tGrid, sGrid), method='cubic')
    z_2 = griddata((df[0], df[1]), df[3], (tGrid, sGrid), method='cubic')

    surface_1 = go.Surface(x=x, y=y, z=z_1, colorscale='Greens')
    surface_2 = go.Surface(x=x, y=y, z=z_2, colorscale='Reds')

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
