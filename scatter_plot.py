# Learn about API authentication here: https://plot.ly/pandas/getting-started
# Find your api_key here: https://plot.ly/settings/api

import plotly.offline as offline
import plotly.graph_objs as go
import pandas as pd
import pdb
df = pd.read_csv('points/points_1_1.csv', header=None, delimiter=';')
df.head()

data = []
clusters = []

trace_1 = dict(
    name = 'Valor obtenido',
    x = df[0], y = df[1], z = df[2],
    type = "scatter3d",
    mode = 'markers',
    marker = dict( size=5, color='#D73361', line=dict(width=0) ) )

trace_2 = dict(
    name = 'Valor real',
    x = df[0], y = df[1], z = df[3],
    type = "scatter3d",
    mode = 'markers',
    marker = dict( size=5, color='#28CC9E', line=dict(width=0) ) )

data.append(trace_1)
data.append(trace_2)

layout = dict(
    width=1200,
    height=800,
    autosize=False,
    title='points_1.csv',
    scene=dict(
        xaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='#F4F4EC'
        ),
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='#F4F4EC'
        ),
        zaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='#F4F4EC'
        ),
        aspectratio = dict( x=1, y=1, z=0.7 ),
        aspectmode = 'manual'
    ),
)

fig = dict(data=data, layout=layout)

# IPython notebook
# py.iplot(fig, filename='pandas-3d-iris', validate=False)

url = offline.plot(fig, filename='scatter.html', validate=False)
