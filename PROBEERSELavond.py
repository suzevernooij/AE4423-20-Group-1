# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:26:15 2025

@author: julia
"""

# Import necessary libraries
  
import numpy as np 
import math  
import copy  
import pandas as pd  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
import networkx as nx
from collections import defaultdict

#import data Suze
# airport_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\AirportData.xlsx"
# fleet_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\FleetType.xlsx"
# demand_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\Group1.xlsx"

#import data Maaike




#import data Julia
airport_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\AirportData.xlsx"
fleet_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\FleetType.xlsx"
demand_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\Group1.xlsx"


# Read Excel Files
df_airports = pd.read_excel(airport_data)
df_fleet = pd.read_excel(fleet_data)
df_demand = pd.read_excel(demand_data, header=None)

# print(df_airports)
# print(df_fleet)
# print(df_demand)

#Defining Parameters Fleet
fleet       = pd.read_excel(fleet_data, skiprows=0, header=0, index_col=0)
speed       = np.array(fleet.iloc[0])
capacity    = np.array(fleet.iloc[1])
TAT         = np.array(fleet.iloc[2])
R           = np.array(fleet.iloc[3])
runway_AC   = np.array(fleet.iloc[4])
Cl          = np.array(fleet.iloc[5])
Cf          = np.array(fleet.iloc[6])
Ch          = np.array(fleet.iloc[7])
cfuel       = np.array(fleet.iloc[8])
fleet       = np.array(fleet.iloc[9])

#Defining Parameters Airport
airport     = pd.read_excel(airport_data, skiprows=0, header=0, index_col=0)
IATA        = np.array(airport.iloc[0])
latitude    = airport.iloc[1].tolist()
longitude   = airport.iloc[2].tolist()
runway_AP   = np.array(airport.iloc[3])

Re = 6371 
f = 1.42
Yield = 0.26

#Defining Sets
n = len(IATA)
N = range(len(IATA)) #Amount of airports = 20
K = range(len(speed)) #Amount of ACs
T = range(120*10) #Total amount of time steps is 120 hour * 10 steps per hour

def demand(dep, arr, tw):
    return float(df_demand[(df_demand[1] == dep) & (df_demand[2] == arr)][tw+3])

step_demand = np.zeros((2,20,1200))
#initalize hub = 3
for i in N:
    for t in T:
        step_demand[0][i][t] = demand(IATA[3], IATA[i], int(t/40))
for i in N:
    for t in T:
        step_demand[1][i][t] = demand(IATA[i], IATA[3], int(t/40))

# def total_demand(dep, arr):
    # total = 0
    # for tw in range(30):  # Aangenomen dat je 40 tijdsvensters hebt (van 0 tot 39)
    #     total_demand += demand(dep, arr, tw)  # Voeg de vraag toe voor het specifieke tijdvenster
    # return total_demand

#%%
# Voorbeeld van hoe je de totale vraag zou berekenen voor specifieke luchthavens
# dep_airport = IATA[3]  # bijvoorbeeld vertrek luchthaven
# arr_airport = IATA[1]  # bijvoorbeeld aankomst luchthaven

# total = total_demand(IATA[1], IATA[3])
# #%%
# print(total)



#%%
# Haversine Distance Calculation
lat_rad = np.radians(latitude)
lon_rad = np.radians(longitude)

IATA_index = list(IATA)
def distance(dep, arr):
    delta_lat = lat_rad[IATA_index.index(dep)] - lat_rad[IATA_index.index(arr)]
    delta_lon = lon_rad[IATA_index.index(dep)] - lon_rad[IATA_index.index(arr)]
    a = math.sin(delta_lat / 2)**2 + math.cos(lat_rad[IATA_index.index(dep)]) * math.cos(lat_rad[IATA_index.index(arr)]) * math.sin(delta_lon / 2)**2
    sigma = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = Re * sigma
    return float(distance)

#%%
#Define operating_costs
def costs(dep, arr, k):
    if dep == arr:
        operating_costs = 0
    else:
        CT = Ch[k] * (distance(dep, arr) / speed[k])
        CFuel = cfuel[k] * f * distance(dep, arr) / 1.5
        operating_costs = CT + Cf[k] + CFuel  
    return operating_costs

def revenue(dep, arr, k, t):
    flow = 0
    Yield = 0.26  # Zet een standaardwaarde voor Yield
    timew = int(t / 40)
    if dep == IATA[3]:
        if step_demand[0][IATA_index.index(arr)][t] <= capacity[k]:
            flow = step_demand[0][IATA_index.index(arr)][t]
        else:
            flow = capacity[k]
    
    elif arr == IATA[3]:
        if step_demand[1][IATA_index.index(dep)][t] <= capacity[k]:
            flow = step_demand[1][IATA_index.index(dep)][t]
        else:
            flow = capacity[k]
    flow = flow / 1000
    revenue = Yield * distance(dep, arr) * flow
    return revenue

# Define total profit
M = -10000000000 
def profit(dep, arr, k, t):
    if R[k] < distance(dep, arr):
        profit = M
    elif runway_AC[k] > runway_AP[IATA_index.index(dep)] or runway_AC[k] > runway_AP[IATA_index.index(arr)]:
        profit = M
    else:
        profit = revenue(dep, arr, k, t) - costs(dep, arr, k)
    return profit

#calculate flight times
flight_time = np.zeros((20,3))
for i in N:
    for k in K:
        #flight_time[i][k] = (distance(IATA[3], IATA[i])/speed[k] + 0.5 + TAT[k]/60)
        flight_time[i][k] = np.ceil((distance(IATA[3], IATA[i])/speed[k] + 0.5 + TAT[k]/60)*10) 
print(flight_time[1][0])

#%%

#calculate profit per aircraft type

#initialize state arrays
state = np.zeros((20,1200,3))
#initialize control city array
control_airport = np.zeros((1200,3), dtype=int)
control_airport[-1,:] = int(3) #begin with the hub
stop = False

while not stop:
    for k in K:
        stay_action = np.zeros((20,1200))
        to_hub_action = np.zeros((20,1200))
        from_hub_action = np.zeros((20,1200))
        for t in T:
            for airport in range(0,3):
                to_hub_action[airport][t] = profit(IATA[airport], IATA[3], k, t)
            for airport in range(4,20):
                to_hub_action[airport][t] = profit(IATA[airport], IATA[3], k, t)
        #Flight from hub to airport
            for airport in range(0,3):
                from_hub_action[airport][t] = profit(IATA[3], IATA[airport], k, t)
            for airport in range(4,20):
                from_hub_action[airport][t] = profit(IATA[3], IATA[airport], k, t)
        for t in range(1198,-1,-1):
            for a in N:
                if a ==3: #hub
                    profits = np.zeros(20)
                    for i in N:
                        if t + flight_time[i][k] < 1199:
                            profits[i] = from_hub_action[i][t] + state[i][int(t+flight_time[i][k])][k]
                        state[a][t][k] = max(profits)
                        control_airport[t][k] = np.argmax(profits).astype(int)
                else:
                   if t + flight_time[i][k] < 1199:
                       state[a][t][k] = max(to_hub_action[a][t] + state[3][int(t+flight_time[a][k])][k], stay_action[a][t+1] + state[a][t+1][k])
    #Determine final profit
    final_profit = np.zeros(3)
    for k in K:
        final_profit[k] = state[3][0][k] - 5 * Cl[k] #Minus leasing costs for 5 days
    k = np.argmax(final_profit) #k has index of highest profit 
    for k in K:
        if np.any(final_profit >= 0):
            flight_schedule = []
            current_time = int(0)
            time_steps = 1199
            aircraft_available = True
            at_hub = True
            start_airport = IATA[3]
            total_flight_time = 0
            
            while current_time < time_steps:
                if at_hub:
                    start_airport = IATA[3]
                    if aircraft_available:
                        if control_airport[int(current_time)][k] == 3:
                            current_time += 1
                        else:
                            if current_time + 2 * flight_time[control_airport[int(current_time)][k]][k] >= time_steps:
                                break
                            else: #fly to control airport
                                destination = IATA[control_airport[int(current_time)][k]]
                                destination_index = control_airport[int(current_time)][k]
                                travel_time = flight_time[control_airport[int(current_time)][k]][k]
                                
                                #store the flight in schedule
                                flight_schedule.append({'aircraft' : k,
                                                        'start': start_airport,
                                                       'destination': destination,
                                                       'departure_time' : current_time,
                                                       'arrival_time': current_time + travel_time,
                                                       'demand_served': demand_served})
                                if capacity[k] <= step_demand[0][destination_index][int(current_time)]:
                                    demand_served = capacity[k]
                                else:
                                    demand_served = step_demand[0][destination_index][int(current_time)]
                                window = int(current_time/40)
                                step_demand[0][destination_index][window*40:window*40+40] -= demand_served
                                total_flight_time += travel_time
                                current_time += travel_time
                                at_hub = False #aircraft no longer at the hub
                                aircraft_available = False #Aircraft no longer available for other flights
                    else:      #if aircraft is no longer available
                        next_available_time = flight_schedule[-1]['arrival_time']
                        if current_time >= next_available_time:
                            aircraft_available = True
                        else:
                            current_time += 1
                else:       #if aircraft is not located at hub
                    if aircraft_available:
                        if state[destination_index][int(current_time+1)][k] > state[3][int(current_time+1)][k]:
                            current_time += 1
                        else:
                            start_airport = destination
                            travel_time = flight_time[destination_index,k]
                            destination = IATA[3]
                            destination_index = 3
                            
                            flight_schedule.append({'aircraft': k,
                                            'start': start_airport,
                                           'destination': destination,
                                           'departure_time' : current_time,
                                           'arrival_time': current_time + travel_time,
                                            'demand_served': demand_served})
                            if capacity[k] <= step_demand[1][IATA_index.index(start_airport)][int(current_time)]:
                                demand_served = capacity[k]
                            else:
                                demand_served = step_demand[1][IATA_index.index(start_airport)][int(current_time)]
                            print(demand_served)
                            window = int(current_time / 40)
                            step_demand[1][IATA_index.index(start_airport)][window*40 : window *40+40] -= demand_served
                            
                            total_flight_time += travel_time
                            current_time += travel_time
                            at_hub = True
                            aircraft_available = False
                            
                    else:
                        next_available_time = flight_schedule[-1]['arrival_time']
                        if current_time >= next_available_time:
                            aircraft_available = True
                        else:
                            current_time += 1
        else:
            stop = True
            break
        if current_time >= time_steps:
            stop = True
            break
                
#%%
if flight_schedule:
    print("Flight Schedule per Aircraft:")
    # Groepeer vluchten per vliegtuig
    flights_by_aircraft = {}
    for flight in flight_schedule:
        aircraft = flight['aircraft']
        if aircraft not in flights_by_aircraft:
            flights_by_aircraft[aircraft] = []
        flights_by_aircraft[aircraft].append(flight)

    # Print vluchten voor elk vliegtuig
    for aircraft, flights in flights_by_aircraft.items():
        print(f"\nAircraft {aircraft}:")
        for i, flight in enumerate(flights):
            print(f"  Flight {i + 1}:")
            print(f"    Start Airport    : {flight['start']}")
            print(f"    Destination      : {flight['destination']}")
            print(f"    Departure Time   : {flight['departure_time']}")
            print(f"    Arrival Time     : {flight['arrival_time']}")
            print(f"    Fleet            : {flight['demand_served']}")
            print("-" * 30)
else:
    print("No flights scheduled.")
                    
                       
                
                    
                                                    
                                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        