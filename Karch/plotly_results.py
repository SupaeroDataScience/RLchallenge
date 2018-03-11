import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import numpy as np

comment = '300000_alpha00001'
scores = np.load('results/scores' + comment + '.npy')
final_score = np.reshape(np.load('results/cumulated.npy'), (1, 20))
scores = np.concatenate((scores, final_score))
avg_cost = np.load('results/avg_cost' + comment + '.npy')
max_cost = np.load('results/max_cost' + comment + '.npy')

avglist = []
maxlist = []
minlist = []
for test in scores:
    avglist.append(np.mean(test))
    minlist.append(np.min(test))
    maxlist.append(np.max(test))

x_score = 25000 * np.linspace(1, 12, 12)
x_cost = 200 * np.linspace(0, 1500, 1503)

app = dash.Dash()

app.layout = html.Div([
    html.H1('Plotting the scores'),
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Max Score', 'value': 'Max Score'},
            {'label': 'Average Score', 'value': 'Average Score'},
            {'label': 'Min Score', 'value': 'Min Score'}
        ],
        value='COKE'
    ),
    dcc.Graph(id='my-graph')
])


@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    if selected_dropdown_value == 'Max Score':
        y = maxlist
    elif selected_dropdown_value == 'Average Score':
        y = avglist
    elif selected_dropdown_value == 'Min Score':
        y = minlist
    return {
        'data': [{
            'x': x_score,
            'y': y,
            'line': dict(
                color=('rgb(167, 46, 22)'),
                width=4,
                dash='dot'),
            'marker': dict(color=('rgb(2,40, 100)'), size=20)
        }]
    }


app.run_server()
