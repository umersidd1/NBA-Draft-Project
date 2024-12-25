# Umer Siddiqi, usiddiqi@usc.edu
# ITP 216, Fall 2024
# Section: 32080
# Final Project
# Description:
# In this final project, I used Flask, MatplotLib, Scikit-learn, pandas, and numpy to make an app that asks the user
# which draft pick they want information on. When the user inputs something (like 1 for example), the program
# outputs a plot that shows the historical data - last 20 years - of each player drafted at that positions ppg.
# Moreover, on the /results.html page, there is an option for the user to predict the points per game for a specific
# year, which takes them to a /predictions.html page that shows what the program predicts the average ppg will be for
# that pick at user inputted year.


import base64
import io
import os
import sqlite3 as sl

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)
db = "nba.db"

@app.route("/")
def home():
    return render_template("home.html", pickOptions=get_pick_options())

def get_pick_options():
    return sorted(set(range(1,61)))

def generate_plot(pick_number):
    conn = sl.connect('nba.db')

    # Query to get draft picks for the specified number
    query = '''
    SELECT DraftYr, Player, PPG 
    FROM nba 
    WHERE Pk = ? 
    ORDER BY DraftYr
    '''

    # Read the data
    df = pd.read_sql_query(query, conn, params=(pick_number,))

    # Check if any data exists
    if df.empty:
        return None

    # Create the plot
    plt.figure(figsize=(15, 7))
    plt.plot(df['DraftYr'], df['PPG'], marker='o')

    # Annotate each point with the player name
    for i, row in df.iterrows():
        plt.annotate(row['Player'],
                     (row['DraftYr'], row['PPG']),
                     xytext=(5, 5),
                     textcoords='offset points', rotation=45)

    plt.title(f'Points Per Game for Draft Position {pick_number} (1990-2021)')
    plt.xlabel('Draft Year')
    plt.ylabel('Points Per Game')
    plt.grid(True, alpha=0.7)

    plt.xticks(df['DraftYr'], rotation=45)

    plt.tight_layout()

    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Encode the image to base64
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png).decode('utf-8')
    buffer.close()

    return graph

# Add route for picks
@app.route("/results", methods=['POST'])
def results_route():
    pick_number = request.form.get('draft_position')

    # Generate plot
    plot = generate_plot(int(pick_number))

    # Get data for the specific draft pick
    conn = sl.connect('nba.db')
    query = '''
    SELECT Player, PPG, DraftYr 
    FROM nba 
    WHERE Pk = ? 
    ORDER BY DraftYr
    '''
    df = pd.read_sql_query(query, conn, params=(pick_number,))

    # Render template with plot and data
    return render_template('results.html',
                           plot=plot,
                           data=df.to_dict('records'),
                           pick_number=pick_number)

def generate_predictive_plot(pick_number, predict_year):
    # Connect to the database
    conn = sl.connect('nba.db')

    # Query to get draft picks for the specified number
    query = '''
    SELECT DraftYr, Player, PPG
    FROM nba
    WHERE Pk = ?
    ORDER BY DraftYr
    '''

    # Read the data
    df = pd.read_sql_query(query, conn, params=(pick_number,))

    # Check if any data exists
    if df.empty:
        return None

    # Prepare data for machine learning
    X = df['DraftYr'].values.reshape(-1, 1)
    y = df['PPG'].values

    # Create polynomial features to capture non-linear trends
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # Train the model on all data
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict future points up to the user's specified year
    future_years = np.array(range(df['DraftYr'].min(), predict_year + 1)).reshape(-1, 1)
    future_years_poly = poly.transform(future_years)
    predicted_ppg = model.predict(future_years_poly)

    if predict_year:
        predict_year_poly = poly.transform(np.array([[predict_year]]))
        custom_prediction = model.predict(predict_year_poly)[0]

    # Create the plot
    plt.figure(figsize=(15, 7))

    # Plot original data points
    plt.scatter(X, y, color='blue', label='Actual Data')

    # Plot prediction line
    plt.plot(future_years, predicted_ppg, color='red', label=f'Prediction up to {predict_year}')

    # Annotate original data points
    for i, row in df.iterrows():
        plt.annotate(row['Player'],
                     (row['DraftYr'], row['PPG']),
                     xytext=(5, 5),
                     textcoords='offset points', rotation=45)

    plt.title(f'Points Per Game Prediction for Draft Position {pick_number}')
    plt.xlabel('Draft Year')
    plt.ylabel('Points Per Game')
    plt.legend()
    plt.grid(True, alpha=0.7)

    # Improve x-axis readability
    plt.xticks(future_years.flatten(), rotation=45)

    plt.tight_layout()

    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Encode the image to base64
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png).decode('utf-8')
    buffer.close()

    return graph, predicted_ppg.tolist(), future_years.flatten().tolist(), custom_prediction

@app.route("/prediction", methods=['POST'])
def prediction_route():
    pick_number = request.form.get('draft_position')

    # Get the year from the user input
    predict_year = request.form.get('predict_year')
    if predict_year:
        predict_year = int(predict_year)
    if not predict_year:
        return render_template('prediction.html',
                               error="Please enter a valid year to predict.")

    # Generate plot with prediction up to the user's specified year
    plot_result = generate_predictive_plot(int(pick_number), predict_year)

    # Get data for the specific draft pick
    conn = sl.connect('nba.db')
    query = '''
    SELECT Player, PPG, DraftYr
    FROM nba
    WHERE Pk = ?
    ORDER BY DraftYr
    '''
    df = pd.read_sql_query(query, conn, params=(pick_number,))

    # If plot generation was successful
    if plot_result:
        plot, predicted_ppg, future_years, custom_prediction = plot_result
        return render_template('prediction.html',
                               plot=plot,
                               data=df.to_dict('records'),
                               pick_number=pick_number,
                               predicted_ppg=predicted_ppg,
                               future_years=future_years,
                               predict_year=predict_year,
                               custom_prediction=custom_prediction)
    else:
        return render_template('prediction.html',
                               plot=None,
                               data=df.to_dict('records'),
                               pick_number=pick_number)

# Any issues go back to home page
@app.route('/<path:path>')
def catch_all(path):
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)