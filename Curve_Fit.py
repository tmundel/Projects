#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:21:35 2020

@author: administrator
"""

from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from jdcal import gcal2jd
from scipy import stats

#Reat text file
temp_full = ascii.read('/Users/administrator/Documents/Phys221/spokane_temperature_1973_2018.txt')
temp = np.array(temp_full['Temp'])
date = temp_full['Date']
hrmn = temp_full['HrMn']
year = np.floor(date/1e4)
month = np.floor((date-year*1e4)/1e2)
day = np.floor(date-year*1e4-month*1e2)
minutes, hours = np.modf(np.array(hrmn)/100)
t0 = 41700
t_final = 58400

#Convert to modified julian date
i = 0
mjd = np.empty(len(year))
while(i < len(year)):
    offset, mjd[i] = gcal2jd(year[i],month[i],day[i])
    i += 1
    
mjd = mjd + hours/24 + minutes/(24*60)

#Define double sin function
def function(mdj,a_1, omega_1, phi_1, a_2, omega_2, phi_2, c):
    '''
    This function takes the neccessary paramaters for a double sin wave equation and returns an equation representing the function with the specified parameters.

    Parameters
    ----------
    mdj : NUMPY ARRAY
        ARRAY HOLDING TIME VALUES FROM THE SPOKANE TEMPERATURE DATA (X VALUES).
    a_1 : FLOAT
        REPRESENTS THE VALUE FOR THE AMPLITUDE PARAMETER OF THE FIRST SIN WAVE.
    omega_1 : FLOAT
        REPRESENTS THE VALUE FOR THE ANGULAR FREQUENCY PARAMETER OF THE FIRST SIN WAVE.
    phi_1 : FLOAT
        REPRESENTS THE VALUE FOR THE PHASE CONSTANT PARAMETER OF THE FIRST SIN WAVE.
    a_2 : FLOAT
        REPRESENTS THE VALUE FOR THE AMPLITUDE PARAMETER OF THE SECOND SIN WAVE.
    omega_2 : FLOAT
        REPRESENTS THE VALUE FOR THE ANGULAR FREQUENCY OF THE SECOND SIN WAVE.
    phi_2 : FLOAT
        REPRESENTS THE VALUE FOR THE PHASE CONSTANT PARAMETER OF THE FIRST SIN WAVE.
    c : FLOAT
        REPRESENTS THE VALUE FOR THE CONSTANT PARAMETER OF THE EQUATION.

    Returns
    -------
    ARRAY
        ARRAY HOLDING NEW EQUATION FOR DOUBLE SIN WAVE WITH SPECIFIED PARAMETERS.

    '''
    return ((a_1 * np.sin(omega_1 * mdj + phi_1)) + (a_2 * np.sin(omega_2 *mdj + phi_2)) + c)

#Initialize variables
omega_1 = 2 * np.pi
omega_2 = (2 * np.pi) / 365
a_1 = np.mean(temp)
a_2 = np.mean(temp)
c = 0
phi_1 = 0
phi_2 = 0 
wave = function(mjd, a_1, omega_1, phi_1, a_2, omega_2, phi_2, c)

#Run curve fit program
p0 = np.array([a_1, omega_1, phi_1, a_2, omega_2, phi_2, c])
ppot, pcovt = curve_fit(function, mjd, temp, p0)


#Plot diurnal data vs. model
plt.plot(mjd, temp, lw=0.5)
plt.axis([53900,53940,0,40])
plt.xlabel('MJD')
plt.ylabel('T ['+u'\u00b0'+'C]')
plt.plot(mjd, function(mjd,*ppot))
plt.xlabel("MJD Day")
plt.ylabel("Temperature (K)")
plt.show()

#Plot annual data vs. model
plt.plot(mjd, temp, lw=0.5)
plt.xlabel('MJD')
plt.ylabel('T ['+u'\u00b0'+'C]')
plt.plot(mjd, function(mjd, *ppot))
plt.xlabel("MJD Day")
plt.ylabel("Temperature (K)")
plt.show()

#1C: Calculate residuals
def residuals(t0, t_final,mjd, temp, ppot):
    '''
    This function calculates the residual (data - model) for a set of data and the slope and intercept of the linear model fit line.
    
    Parameters
    ----------
    t0 : INTEGER
        REPRESENTS THE STARTING VALUE OF THE DATA TO BE INCLUDED IN RESIDUAL FUNCTION    
    t_final : INTEGER
        REPRESENTS THE ENDING VALUE OF THE DATA TO BE INCLUDED IN RESIDUAL FUNCTION    
    mdj : NUMPY ARRAY
        ARRAY HOLDING VALUES FOR THE INDEPENDENT DATA CALCULATED IN RESIDUALS (X VALUES).
    temp : NUMPY ARRAY
        ARRAY HOLDING VALUES FOR THE DEPENDENT DATA CALCULATED IN RESIDUALS
    ppot: NUMPY ARRAY
        ARRAY HOLDING PARAMETERS NECCESARY TO CALL MODEL FUNCTION
    Returns
    -------
    slope: FLOAT
        FLOAT REPRESENTING THE SLOPE OF THE LINEAR MODEL FOR THE RESIDUAL DATA
    intercept: FLOAT
        FLOAT REPRESENTING THE INTERCEPT OF THE LINEAR MODEL FOR THE RESIDUAL DATA
    residual: NUMPY ARRAY
        ARRAY HOLDING VALUES OF THE RESIDUAL AT EACH DATA POINT
    '''
    start_index = (np.abs(mjd - t0)).argmin()
    end_index = (np.abs(mjd - t_final)).argmin()
    residual = temp[start_index : end_index] - function(mjd[start_index : end_index], *ppot)
    x = np.linspace(t0, t_final, np.size(residual))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,residual)
    
    return slope, intercept, residual

slope, intercept, residual = residuals(t0, t_final, mjd, temp, ppot)

 #Plot residuals and overlay linear model 
x = np.linspace(t0, t_final, np.size(residual))
plt.figure()
plt.plot(x, residual)
plt.plot(x, ((slope * x) + intercept))
plt.xlabel("MJD Day")
plt.ylabel("Residual")

print("The slope of the linear fit to the residuals of the data is {: .2e}, and the intercept is {: .2f} \n" .format(slope, intercept))

#1D: Smooth

def smooth(mjd, t0, t_final, residual):
    '''
    This function calculates the smooth running average for a set of data and plots the smoothed data with a linear model fit line.
    
    Parameters
    ----------
    mdj : NUMPY ARRAY
        ARRAY HOLDING VALUES FOR THE INDEPENDENT DATA CALCULATED IN RESIDUALS (X VALUES).
    t0 : INTEGER
        REPRESENTS THE STARTING VALUE OF THE DATA TO BE INCLUDED IN RESIDUAL FUNCTION    
    t_final : INTEGER
        REPRESENTS THE ENDING VALUE OF THE DATA TO BE INCLUDED IN RESIDUAL FUNCTION    
    residual : NUMPY ARRAY
        ARRAY HOLDING VALUES FOR THE RESIDUALS OF THE DATA
        
    Returns
    -------

    smooth_residual: NUMPY ARRAY
        ARRAY HOLDING VALUES OF THE SMOOTHED AVERAGE DATA
    '''
    start_index = (np.abs(mjd - t0)).argmin()
    end_index = (np.abs(mjd - t_final)).argmin()
    i = start_index
    smooth_residuals = np.zeros(0)
    while i < end_index:
        smooth_data = residual[i - 2000 : i + 2000]
        mean = np.mean(smooth_data)
        smooth_residuals = np.append(smooth_residuals, mean)
        i += 4000
    #Plot smooth function
    plt.figure()
    y = np.linspace(t0, t_final, np.size(smooth_residuals))
    plt.plot(y, smooth_residuals)
    plt.plot(y, ((slope * y) + intercept))
    plt.xlabel("MJD Day")
    plt.ylabel("Residual (Data - Model)")
    
    return smooth_residuals
#Call Smoothing function
smooth_residuals = smooth(mjd, t0, t_final, residual)
  
#1E: Calculate difference of residual mean between earlier (MJD 41683 to 50082) times and later (MJD 50083 to 58483) times
t0_early = 41683
t_final_early = 50082
t0_late = 50083
t_final_late = 58483

#Call residual functions for different intervals
slope_early, intercept_early, residual_early = residuals(t0_early, t_final_early, mjd, temp, ppot)
slope_late, incercept_late, residual_late = residuals(t0_late, t_final_late, mjd, temp, ppot)
#I think the runtime error is from the early residual function since t0_early corresponsds to an index of 0, and slicing an index of 0 is frowned upon.

#Calculate mean of both residual arrays and calculate difference
early_residual_mean = np.mean(residual_early)
late_residual_mean = np.mean(residual_late)
residual_mean_difference = (late_residual_mean - early_residual_mean)
print("The difference between the earlier residual mean and the later residual mean is {: .2f} \n" .format(residual_mean_difference))

#Calculate uncertainties for each mean
early_uncertainty = np.std(residual_early) / np.sqrt(np.size(residual_early))
late_uncertainty = np.std(residual_late) / np.sqrt(np.size(residual_late))
difference_uncertainty = np.sqrt(early_uncertainty**2 + late_uncertainty**2)
print("The uncertainty of the difference between the earlier and later residuals is{: .2f} \n" .format(difference_uncertainty))

#Calculate significance
significance_warming = residual_mean_difference / difference_uncertainty
print("The significance level of the warming detection is {: .2f}, which shows a significant change\n" .format(significance_warming))


 
#2A: Predict temperature on December 4th

#Convert date into a Modified Julian Date
prediction_year = 2020
prediction_month = 12
prediction_day = 4
prediction_hour = 12
time_difference = 8
offset, prediction_mjd = gcal2jd(prediction_year, prediction_month, prediction_day)
prediction_mjd += ((prediction_hour + time_difference)/ 24)

#Run model function with mjd date
prediction_temp = (function(prediction_mjd, *ppot))
prediction_farenheight = prediction_temp * (9/5) + 32
print("The predicted temperature in Spokane at 12 in the afternoon on December 4th is{: .2f} degrees celsius or{: .2f} degrees farenheight. \n" .format(prediction_temp, prediction_farenheight))

#2B: Mean Temperature on December 4th

#Convert date into a Modified Julian Date and initialize variables for loop
offset, prediction_start = gcal2jd(np.min(year), prediction_month, prediction_day)
prediction_start += ((prediction_hour + time_difference) / 24)
i = prediction_start
predictions = np.zeros(0)

#Loop that loops through one year at a time to calculate temp on December 4th
while(i < 58483):
    prediction_temp = function(i, *ppot)
    predictions = np.append(predictions, prediction_temp)
    i += 365.25
    
mean_prediction = np.mean(predictions)
mean_farenheight = mean_prediction * (9/5) + 32
print("The mean temperature on December 4th is {: .2f} degrees celsius and {: .2f} degrees farenheight\n" .format(mean_prediction, mean_farenheight))

#2C: Confidence Interval

confidence_interval = .95
stdev_prediction = np.std(predictions, ddof = 1)
#Use formula for gaussian distribution
confidence_lower = mean_prediction -  (1.96 * stdev_prediction) / np.sqrt(np.size(predictions))
confidence_upper = mean_prediction + (1.96 * stdev_prediction) / np.sqrt(np.size(predictions))
print("The 95% confidence interval calculated from a gaussian distribution is {: .2f} to {: .2f} \n" .format(confidence_lower, confidence_upper))

#2D: Find 95% confidence interval


def bootstrap(predictions, confidence_interval, number_of_samples):
    '''
    Function to calculate the confidence interval of a set of data using a nonparametric bootstrap
    
    Parameters
    ----------
    predictions : NUMPY ARRAY
        ARRAY HOLDING VALUES THE MEAN VALUES OF THE PREDITCED TEMPERATURES.
    confidence_interval : FLOAT
        REPRESENTS THE DESIRED CONFIDENCE INTERVAL FOR THE FUNCTION TO CALCULATE   
    number_of_samples : INTEGER
        REPRESENTS THE NUMBER OF SAMPLES TO BE USED WHEN CALCLUTING BOOTSTRAP  
        
    Returns
    -------
    
    means: NUMPY ARRAY
        NUMPY ARRAY HOLDING VALUES WITHIN THE SPECIFIED CONFIDENCE INTERVAL
    
    '''
    
    if(np.size(predictions) > 10):
        
        i = 0
        means = np.zeros(0)
        while(i < number_of_samples):
            sample = np.random.choice(predictions, np.size(predictions))
            means = np.append(means, np.mean(sample))
            i += 1

        low_index = int((np.size(means) * (1 - confidence_interval)) / 2)
        high_index = int(np.size(means) - low_index)
        np.ndarray.sort(means)
        means = means[low_index : high_index]
        return means

#Initialize variables and call function
number_of_samples = 1e5
means = bootstrap(predictions, confidence_interval, number_of_samples)
low_value = np.min(means)
upper_value = np.max(means)
print("The 95% confidence interval calculated form the bootstrap is {: .2f} to {: .2f}\n" .format(low_value, upper_value))

