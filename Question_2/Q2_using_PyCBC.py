'''ATTEMPT 3: Importing PyCBC library: Python Library for Gravitational Wave Data Analysis'''

import pandas as pd
import numpy as np
from pycbc.waveform import get_fd_waveform
from pycbc.filter import sigma
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import matched_filter
from pycbc.types import TimeSeries

from pycbc import pnutils

def predictionFunction(inputCSVfile):
    df = pd.read_csv(inputCSVfile)
    time = df["time"].values
    strain = df["strain"].values

    strain_data = TimeSeries(strain.values,delta_t = time[1]- time[0])

    #the masses are unknown, lets begin with an assumption
    #as m1< m2, and m1<60, m2<80 in Solar Masses
    #initialising the template with guesses of unknown masses
    m1 = 20.0
    m2 = 50.0
    distance = 100
    #f_lower: lower cutoff frequency for waveform, freq below this values are typically not used
    hp, _ = get_fd_waveform(approximant="IMRPhenomD", mass1=m1, mass2=m2, delta_f=1.0/128.0, f_lower=20.0)
    
    
    #psd = power spectral density- power of a signal distributed across different frequencies
    psd = strain_data.psd(2)
    #'delta_f' is the frequency spacing in the original PSD
    #estimating PSD at frequencies that may not be directly sampled in the original data.
    psd = interpolate(psd, hp.delta_f)
    #truncating to remove high-frequency components dominated by noise
    psd = inverse_spectrum_truncation(psd, int(2 * psd.sample_rate), low_frequency_cutoff=15.0)
    #white data is the strain data, whitened to have a flat power spectral density (PSD)
    white_data = (strain_data.to_frequencyseries() / psd**0.5).to_timeseries()
    #whitening indicates distinguishing the GWS from background noise
    #dividing the data by the square root of PSD

    #Performing the masses parameter estimation
    m1, m2 = pnutils.mass1_mass2_from_mchirp_q(estimated_parameters['mass1'], estimated_parameters['mass2'])

    #Updating the initialised waveform template of guessed masses with estimated masses
    hp_estimate, _ = get_fd_waveform(approximant="IMRPhenomD", mass1=m1, mass2=m2, delta_f=1.0/128.0, f_lower=20.0)

    #snr_estimate = TimeSeries object having signal-to-noise ration function of time
    snr_estimate = matched_filter(hp_estimate, white_data)
    #high SNR = more significan signal relative to noise
    #low SNR = weaker signal,difficult to distinguish from noise

    #extracting peak SNR value     
    estimated_parameters = {'mass1': m1, 'mass2': m2, 'distance': distance, 'snr': snr_estimate.abs_max(), 'peak_time': snr_estimate.sample_times[snr_estimate.abs_argmax()]}

    print("Attempt 3")
    print("Updated Estimated Parameters:-")
    print(estimated_parameters)


csv1 = pd.read_csv("data1.csv")
csv2 = pd.read_csv("data2.csv")
csv3 = pd.read_csv("data3.csv")

predictionFunction(csv1)
predictionFunction(csv2)
predictionFunction(csv3)
