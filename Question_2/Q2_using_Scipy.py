'''APPROACH 1: Scipy Library'''
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def waveform_template(time, masses):
    # For simplicity, let's assume a sinusoidal waveform
    m1, m2 = masses
    frequency = 2 * np.sqrt(m1 + m2) / (2 * np.pi)
    waveform = np.sin(2 * np.pi * frequency * time)
    return waveform

def cost_function(masses, time, strain):
    # Compute the difference between the data and the model
    model = waveform_template(time, masses)
    cost = np.sum((strain - model)**2)
    return cost

def predict_masses(input_csv):
    # Load strain vs. time data from CSV
    df = pd.read_csv(input_csv)
    time = df["time"].values
    strain = df["strain"].values

    # Initial guess for masses
    initial_guess = [30.0, 45.0]

    # Perform optimization to find the masses that minimize the cost function
    result = minimize(cost_function, initial_guess, args=(time, strain), method='L-BFGS-B')

    # Extract the estimated masses
    estimated_masses = result.x
    print("Estimated Masses for "+ input_csv)
    print(estimated_masses)

# Example usage with a CSV file containing time and strain columns
predict_masses("data1.csv")
predict_masses("data2.csv")
predict_masses("data3.csv")
print("Attempt1")
