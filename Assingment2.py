# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:13:45 2024

@author: julia
"""

# Import necessary libraries
import gurobipy as gp  
from gurobipy import *  
from gurobipy import GRB  
import numpy as np 
import math  
import copy  
import pandas as pd  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
import networkx as nx
from collections import defaultdict

#name model
m = Model('AirlinePlanningAssingment1')

#import data Suze

#import data Maaike

#import data Julia
airport_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\AirportData.xlsx"
fleet_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\FleetType.xlsx"
demand_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\Group1.xlsx"


#Defining Parameters Fleet
fleet       = pd.read_excel(fleet_data, skiprows=0, header=0, index_col=0)
speed       = fleet.iloc[0].tolist()
capacity    = fleet.iloc[1].tolist()
TAT         = fleet.iloc[2].tolist()
R           = fleet.iloc[3].tolist()
runway_AC   = fleet.iloc[4].tolist()
Cl          = fleet.iloc[5].tolist()
Cf          = fleet.iloc[6].tolist()
Ch          = fleet.iloc[7].tolist()
cfuel       = fleet.iloc[8].tolist()
fleet       = fleet.iloc[9].tolist()

#Defining Parameters Airport
airport     = pd.read_excel(airport_data, skiprows=0, header=0, index_col=0)
IATA        = airport.iloc[0].tolist()
latitude    = airport.iloc[1].tolist()
longitude   = airport.iloc[2].tolist()
runway_AP   = airport.iloc[3].tolist()

Re = 6371 

#Defining Sets
n = len(IATA)
N = range(len(IATA))
K = range(len(speed))

# Haversine Distance Calculation
lat_rad = np.radians(latitude)
lon_rad = np.radians(longitude)
d = np.zeros((n, n))  # Distance matrix
for i in range(n):
    for j in range(n):
        delta_lat = lat_rad[i] - lat_rad[j]
        delta_lon = lon_rad[i] - lon_rad[j]
        a = math.sin(delta_lat / 2)**2 + math.cos(lat_rad[i]) * math.cos(lat_rad[j]) * math.sin(delta_lon / 2)**2
        sigma = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d[i, j] = Re * sigma



