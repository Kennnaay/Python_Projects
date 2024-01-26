"""
Kenny Nguyen, kennyang@usc.edu
Fall 2023
Wednesday 6-7:50pm
Final Project
Displays Free Throw Data for ALL NBA players 
"""
from flask import Flask, redirect, render_template, request, url_for, send_file 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from flask import Response 
import io 
from io import BytesIO
import sqlite3 as sl
from sklearn.linear_model import LinearRegression
import base64

app = Flask(__name__)

#------------------- Connect to database --------------------#
database_connection = sl.connect('players.db')
database_cursor = database_connection.cursor()

#------------Ensure that the 'nba' table does not exist, and then create it ------ #
database_cursor.execute('DROP TABLE IF EXISTS nba') 
database_cursor.execute('CREATE TABLE IF NOT EXISTS nba (player TEXT, year INTEGER, pts INTEGER, fg_pct REAL)')

df = pd.read_csv('players.csv')

#------------------- Fill missing values in numeric columns with their mean --------------------#
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())


database_cursor.executemany('INSERT INTO nba(player, year, pts, fg_pct) VALUES (?, ?, ?, ?)', 
[(df.loc[i, 'Player'], df.loc[i, 'Year'], df.loc[i, 'PTS'], df.loc[i, 'FG%']) for i in range(len(df))])
database_connection.commit()

player_data = {}
player_models = {}
model_ft_pct = LinearRegression()

# Loop through each row in the dataset
for idx, row in df.iterrows():
    player_name = row['Player']
    player_year = row['Year']
    player_ft_attempts = row['FTA']
    player_ft_pct = row['FT%']

    if player_name not in player_data:
        # Initialize data structures for the player
        player_data[player_name] = {'FT Attempts': {player_year: player_ft_attempts}, 'FT%': {player_year: player_ft_pct}}
        player_models[player_name] = {'FT Attempts': LinearRegression(), 'FT%': LinearRegression()}
        
        # Train the models for FT Attempts and FT%
        X = [row['FG%'], row['3P%'], row['FT%'], row['G']]
        player_models[player_name]['FT Attempts'].fit([X], [player_ft_attempts])
        player_models[player_name]['FT%'].fit([X], [player_ft_pct])
        
        # Train the model for FT%
        model_ft_pct.fit([X], [player_ft_pct])

    else:
        player_data[player_name]['FT Attempts'][player_year] = player_ft_attempts 
        player_data[player_name]['FT%'][player_year] = player_ft_pct 


@app.route('/')
def home():
    
    # Display the options 
    graph_types = {'Free Throws Attempted': 'histogram1', 
                   'Free Throw % (includes prediction)': 'histogram2'}

    return render_template('home.html', players=db_get_players(), graph=graph_types, message="Analyze any NBA player since 1993.")

# Chooses which graph to make 
@app.route('/analyze_player', methods=['POST'])
def analyze_player():
    selected_player = request.form['player']
    selected_graph = request.form['graph']
    
    if selected_graph == 'Free Throws Attempted':
        fig = create_graph(selected_player, 'histogram1')
    elif selected_graph == 'Free Throw % (includes prediction)':
        fig = create_graph(selected_player, 'histogram2')

    # Save the graph to an image and convert to base64 for rendering in HTML
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    fig_data = base64.b64encode(img.getvalue()).decode()

    return render_template('stats.html', fig=fig_data, selected_player=selected_player) # Renders the stats.html template and passes in the selected player and visual )


# Gets all unique player names from data 
def db_get_players():
    all_players = []
    for player in player_data.keys():
        if player not in all_players:
            all_players.append(player)
    return all_players

# Predicts the free throw percentage of a player
def projection(player, data):
    ft_pct_prediction = player_models[player]['FT%'].predict([data])
    return ft_pct_prediction, ft_pct_prediction

# Create graph
@app.route("/fig/<data_request>/<locale>")
def fig(player, type):
    fig = create_graph(player, type)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype="image/png")

# Which graph to make and display to the user
# Create graph
def create_graph(player, type):
    ft_attempts_data = player_data[player]['FT Attempts']
    ft_data = player_data[player]['FT%']

    # ----------------- Displays graph for FREE THROW ATTEMPTS ------------ #
    if type == 'histogram1': 
        fig = Figure() 
        axis = fig.subplots()
        axis.set_title(f"{player}'s Free Throw Attempts")
        axis.set_xlabel('Year')
        axis.set_ylabel('FTA')
        axis.plot(list(ft_attempts_data.keys()), list(ft_attempts_data.values()), marker='o')
        axis.legend()
        return fig

    # ----------------- Displays graph for FREE THROW PERCENTAGE ------------ #
    elif type == 'histogram2': 
        X = [row['FG%'], row['3P%'], row['FT%'], row['G']]
        ft_pct_pred, _ = projection(player, X)
        
        fig = Figure() 
        axis = fig.subplots()
        axis.set_title(f"{player}'s Free Throw Percentage")
        axis.set_xlabel('Year')
        axis.set_ylabel('FT%')
        axis.plot(list(ft_data.keys()), list(ft_data.values()), marker='o')
        axis.axhline(y=ft_pct_pred, color='b', linestyle='dashed', label='Predicted FT%')
        axis.legend()
        return fig

    else:
        fig = Figure()
        return fig

if __name__ == '__main__':
    app.run(debug=True)
