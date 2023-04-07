import dash
from dash import dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import base64

##########################################DATA PREPROCESSING############################################################

colorscale = [[0, 'rgb(255, 255, 255)'], [1, 'rgb(255, 0, 0)']]

df_oscars = pd.read_csv('data/every_nom_wins.csv')

df_imdb = pd.read_csv('data/movies_rating_genre.csv')
roles = ['Director','Actor']
data_agg = []

df_oscars['win_percentage'] = ((df_oscars['winner']/df_oscars['Nominations'])*100).round(2)
def get_color(row):
    if row['Role'] == 'Director':
        return '#CD853F'
    elif row['Role'] == 'Actor':
        return '#deb887'
    else:
        return ''

df_oscars['Color'] = df_oscars.apply(get_color, axis=1)
sorted = df_oscars.sort_values(by='winner',ascending = False)
top5directors = sorted[sorted['Role'] == 'Director']
top5actors = sorted[sorted['Role'] == 'Actor']

data_directors = [
    {"img": "images/empty.png", "name": 'Name', "value": "Win Percentage"},
    {"img": "images/steven-spielberg.jpeg", "name": top5directors['Name'].iloc[0], "value": str(top5directors['win_percentage'].iloc[0])+"%"},
    {"img": "images/director2.jpg", "name": top5directors['Name'].iloc[1], "value": str(top5directors['win_percentage'].iloc[1])+"%"},
    {"img": 'images/director3.jpg', "name": top5directors['Name'].iloc[2], "value": str(top5directors['win_percentage'].iloc[2])+"%"},
    {"img": "images/director4.jpg", "name": top5directors['Name'].iloc[3], "value": str(top5directors['win_percentage'].iloc[3])+"%"},
    {"img": "images/director5.jpg", "name": top5directors['Name'].iloc[4], "value": str(top5directors['win_percentage'].iloc[4])+"%"},
]

data_actors = [
    {"img": "images/empty.png", "name": 'Name', "value": "Win Percentage"},
    {"img": "images/actor1.jpg", "name": top5actors['Name'].iloc[0], "value": str(top5actors['win_percentage'].iloc[0])+"%"},
    {"img": "images/actor2.jpg", "name": top5actors['Name'].iloc[1], "value": str(top5actors['win_percentage'].iloc[1])+"%"},
    {"img": 'images/actor3.jpg', "name": top5actors['Name'].iloc[2], "value": str(top5actors['win_percentage'].iloc[2])+"%"},
    {"img": "images/actor4.jpg", "name": top5actors['Name'].iloc[3], "value": str(top5actors['win_percentage'].iloc[3])+"%"},
    {"img": "images/actor5.jpg", "name": top5actors['Name'].iloc[4], "value": str(top5actors['win_percentage'].iloc[4])+"%"},
]

all_genres = pd.unique(df_imdb[['Genre1', 'Genre2', 'Genre3']].values.ravel())
genres_unique = list(filter(lambda x: x is not None and not pd.isna(x), all_genres))
#################################################### Stacked bar plot ##################################################

df_ratio = pd.read_csv('data/ratio_Wins_Nominations_per_Genre.csv')
grouped = df_ratio.groupby('Genre').agg({'winner': 'sum', 'Genre': 'count'})
grouped.columns = ['total_winners', 'total_rows']
average = df_ratio.groupby('Genre')['winner'].mean()
grouped['%'] = average

# sort the dataframe by the percentage of winners in descending order
grouped = grouped.sort_values('%', ascending=True)

# sample data
labels = grouped.index
data1 = grouped['total_winners']
data2 = grouped['total_rows']

# calculate row totals and normalize the data
row_totals = [sum(row) for row in zip(data1, data2)]
data1_norm = [val/total for val, total in zip(data1, row_totals)]
data2_norm = [val/total for val, total in zip(data2, row_totals)]

# Create a dataframe with normalized data
df = pd.DataFrame({'labels': labels,
                   'Wins': data1_norm,
                   'Nominations': data2_norm})

# Create a stacked horizontal bar chart
fig_stacked = px.bar(df, x=['Wins', 'Nominations'], y='labels', orientation='h',
              color_discrete_sequence=['#CD853F', '#deb887'])

fig_stacked.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
fig_stacked.update_xaxes(title_text='Win Ratio')
fig_stacked.update_yaxes(title_text='Movie Titles')
fig_stacked.update_layout(margin=dict(l=30, r=5, b=30, t=30, pad=5))
fig_stacked.update_layout(legend_title="")
fig_stacked.update_traces(marker_line_color='DarkSlateGrey', marker_line_width=1.5)
fig_stacked.update_traces(hovertemplate="<b>Ratio: </b>%{x}<br>" +
                  "<b>Genre:</b> %{y}")



################################################ LINE PLOT ##########################################################

df_oscars1 = pd.read_csv('data/file_name.csv')
temp = df_oscars1.iloc[:, [1, 14]]
temp_by_year = temp.groupby(['year_film']).agg(['mean'])
gross = temp_by_year[('Gross', 'mean')].tolist()
years_no_sort = df_oscars1['year_film'].unique()
years = np.sort(years_no_sort).tolist()

fig_lineplot = px.line(temp, x=years, y=gross)

fig_lineplot.update_layout(
                   legend_title="Evolution of the Gross Worldwide within Oscar nominees for Best Picture"
)
fig_lineplot.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
fig_lineplot.update_traces(line_color='#886308')
fig_lineplot.update_xaxes(title_text='Years')
fig_lineplot.update_yaxes(title_text='Gross')
fig_lineplot.update_layout(margin=dict(l=30, r=30, b=30, t=15, pad=10))
fig_lineplot.update_traces(hovertemplate=
                                        "<b>Year:</b> %{x}<br>" +
                                        "<b>Average Gross:</b> %{y:.2f}$<br>")

################################################ SCATTER PLOT ##########################################################
for role in roles:
    data_agg.append(dict(type='scatter',
                        x=df_oscars[df_oscars['Role'] == role]['Nominations'],
                        y=df_oscars[df_oscars['Role'] == role]['winner'],
                        name=role + 's',
                        mode='markers',
                        marker=dict(size=8, symbol="diamond", color=df_oscars[df_oscars['Role'] == role]['Color'], line=dict(width=1, color="DarkSlateGrey")),
                        hovertemplate=
                         '<b>Name</b>: %{text}<br>' +
                         '<b>Wins</b>: %{y}' +
                         '<br><b>Nominations</b>: %{x}<br>',
                        text=df_oscars[df_oscars['Role'] == role]['Name']
                        )
                    )

layout_agg = dict(
                    yaxis=dict(title='Oscars Wins'),
                    xaxis=dict(title='Nominations'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                 )

fig_scatterplot = go.Figure(data=data_agg,layout=layout_agg)

################################################################ APP ####################################################

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1('Oscars and IMDB Statistics'),
            html.Img(src='data:image/png;base64,{}'.format(base64.b64encode(open('images/oscars_trophy.png', 'rb').read()).decode('ascii')),
                     style={
                        'width':'10%',
                        'height':'10%'
                     }
            )
        ],className='rowTitle'),
        html.Br(),
        html.H3('Get to know the Oscars nominated movies and their details, namely: Gross Worldwide, IMDB Ratings, Genre and Celebrities.',
                style={
                    'margin-bottom':'0px'
                }
        ),
        html.Br()
    ]),

    html.Div([
        html.Div([
        ],className='vertical_line'),

        html.Div([
                html.Div([
                    html.H2(children='Explore the evolution of the Gross Worldwide for Best Picture Nominated Movies'),
                    dcc.Graph(
                        id = 'lineplot',
                        figure = fig_lineplot
                    ),
                    html.H5(children=". ")
                ],className='column1')
        ],className='column4'),

        html.Div([
        ],className='vertical_line'),

        html.Div([
            html.Div([
                html.H2(children='Choose a genre and get the TOP 5 IMDb Rated Movies:'),
                dcc.Dropdown(
                        id='genre-dropdown',
                        options=genres_unique,
                        value='Drama',
                        multi=False,
                        style={
                        'font-family':'Helvetica',
                        'font-size' : '12px'
                        }
                ),
                html.Br(),
                dcc.Graph(id='genre-graph')
            ],className='column1')
        ],className='column4'),

        html.Div([

        ],className='vertical_line')
    ],className='row'),

    html.Div([
        html.Div([
        ],className='vertical_line'),

        html.Div([
            html.Div([
                html.H2(children='Find the Genres with the highest Win/Nominee Ratio'),
                dcc.Graph(
                    id='stacked',
                    figure=fig_stacked
                ),
            ],className='column1'),

            html.Br(),

            html.Div([
                html.H2(children='Note:', style={'margin-top': '40px',
                                                 'color':'black'}),
                html.H2(
                    children='When building the visualizations, a nomination or win was considered for any category.',style={'color':'black'}),
                html.H2(
                    children='When counting actors and directors nominations and wins, any nomination and win for any category for the movie they directed/starred counted towards their total sum',
                    style={'margin-bottom': '10px',
                           'color':'black'}),

                html.Img(src='data:image/png;base64,{}'.format(base64.b64encode(open('images/final_oscar.jpg', 'rb').read()).decode('ascii')),
                     style={
                        'width':'35%',
                        'height':'30%',
                         'text-align':'center',
                         'margin':'10px',
                         'padding':'10px',
                         'border-radius': '20px'
                     }
                )


            ],className='column1',style={'align-items':'center',
                                         'text-align':'center'})

        ],className='column4'),

        html.Div([
        ],className='vertical_line'),

        html.Div([
            html.Div([
                html.H2(children='Discover the biggest names in Oscar history by exploring the Nominations and Wins per Directors and Actors:'),
                dcc.Graph(id='fig_scatter2',figure=fig_scatterplot),
                html.Div([
                    html.Div([
                        html.H6(children='Top 5 outstanding Directors'),
                        html.Table([
                            html.Tr([
                                html.Td(
                                    html.Img(src='data:image/png;base64,{}'.format(
                                        base64.b64encode(open(row['img'], 'rb').read()).decode('ascii')), style={
                                        'height': '50px'}), style={'text-align' :'center'}),
                                html.Div([
                                ],className='vertical_line_top5'),

                                html.Td(row["name"],
                                        style={
                                            'text-align' : 'left',
                                            'font-size' : '18px',
                                            'font-family' : 'Helvetica'}
                                ),

                                html.Div([
                                ],className='vertical_line_top5'),

                                html.Td(row["value"],
                                        style={
                                                'text-align' : 'right',
                                                'font-size' : '18px',
                                                'font-family' : 'Helvetica'}
                                )
                            ], style = {
                                'text-align' : 'center'
                                }
                            )
                            for row in data_directors
                        ],style= {
                            'text-align' : 'center'
                            }
                        )

                    ],className='column4'),

                    html.Div([
                    ],className='vertical_line_top5'),

                    html.Div([
                        html.H6(children='Top 5 outstanding Actors'),
                        html.Table([
                            html.Tr([
                                html.Td(
                                    html.Img(src='data:image/png;base64,{}'.format(
                                        base64.b64encode(open(row['img'], 'rb').read()).decode('ascii')), style={
                                        'height': '50px'}), style={'text-align' :'center'}),
                                html.Div([

                                ], className='vertical_line_top5'),

                                html.Td(row["name"], style={
                                    'text-align': 'left',
                                    'font-size': '18px',
                                    'font-family': 'Helvetica'
                                }),

                                html.Div([

                                ], className='vertical_line_top5'),

                                html.Td(row["value"], style={
                                    'text-align': 'right',
                                    'font-size': '18px',
                                    'font-family': 'Helvetica'
                                })
                            ], style={
                                'text-align': 'center'
                            })
                            for row in data_actors
                        ], style={
                            'text-align': 'center',
                            'margin-top':'5px'
                        })
                    ],className='column4')
                ],className='row')
            ],className='column1')
        ],className='column4'),

        html.Div([
        ],className='vertical_line')

    ],className='row'),
    html.Br(),
    html.Div([
        html.H4('Students: Ana Sofia Mendes 20220687, David Castro 20220688, Sofia Vieira 20220676, Vasco Fontoura 20220556')
    ], className='column_students'),
])

# Bar Plot with responsive filter
# 1- Bar Plot
# Filters: Genres

@app.callback(
    Output('genre-graph', 'figure'),
    Input('genre-dropdown', 'value')
)
def update_graph(selected_genre):

    genre_df = df_imdb.loc[
        (df_imdb['Genre1'].str.contains(selected_genre)) | (df_imdb['Genre2'].str.contains(selected_genre)) | (
            df_imdb['Genre3'].str.contains(selected_genre))].head(5).sort_values(by='IMDB_Rating', ascending=True)
    movies_df = genre_df[['film', 'IMDB_Rating']]

    data = [go.Bar(x=movies_df['IMDB_Rating'],
                y=movies_df['film'],
                orientation='h',
                hovertemplate=
                         '<b>Movie Title</b>: %{y}<br>' +
                         '<b>IMDb Rating</b>: %{x}<extra></extra>',
                marker=dict(
                    color='#886308',
                    colorscale=colorscale,
                )
                )]

    layout = go.Layout(
                       xaxis_title='IMDB Rating',
                       yaxis_title='Movie Title',
                       plot_bgcolor='rgba(184, 135, 11, 0.0)',
                       paper_bgcolor='rgba(184, 135, 11, 0.0)',

                       title=dict(text=f"IMDb Rating for Movies in {selected_genre} Genre", font=dict(family='Helvetica', size=20, color='#666'),x=0.5),
                       xaxis=dict(title='X Axis Label', titlefont=dict(family='Helvetica', size=18, color='#666')),
                       yaxis=dict(title='Y Axis Label', titlefont=dict(family='Helvetica', size=18, color='#666'))
                       )

    return go.Figure(data= data,layout= layout)


if __name__ == '__main__':
    app.run_server(debug=True)