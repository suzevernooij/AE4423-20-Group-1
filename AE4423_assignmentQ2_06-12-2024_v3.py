# Import necessary libraries
import gurobipy as gp  # Gurobi optimization library
from gurobipy import GRB  # Gurobi constants
import pandas as pd  # For data manipulation

# Define the model
m = gp.Model('AirlinePlanningAssignment1_Q2')

# Path to the Excel file
excel_file = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 1\Assignment Q2\Group_1.xlsx"
sheet_name_flights = "Flights"
sheet_name_itineraries = "Itineraries"
sheet_name_recapture = "Recapture"

# Load data from Excel sheets
df_flights = pd.read_excel(excel_file, sheet_name=sheet_name_flights)
df_itineraries = pd.read_excel(excel_file, sheet_name=sheet_name_itineraries)
df_recapture = pd.read_excel(excel_file, sheet_name=sheet_name_recapture)

# Debugging: Display first few rows to ensure data is read correctly
print("Flights Data:")
print(df_flights.head())
print("\nItineraries Data:")
print(df_itineraries.head())
print("\nRecapture Data:")
print(df_recapture.head())

# Sets
L = set(df_flights['Flight No.'])  # Set of flight numbers
P = set(df_itineraries['Itinerary'])  # Set of passenger itineraries

# Recapture pairs: Only include valid entries
P_p = set()
for _, row in df_recapture.iterrows():
    if pd.notna(row['From Itinerary']) and pd.notna(row['To Itinerary']):
        P_p.add((row['From Itinerary'], row['To Itinerary']))  # Recapture pairs

# Parameters
CAP = dict(zip(df_flights['Flight No.'], df_flights['Capacity']))  # Flight capacities
fare = dict(zip(df_itineraries['Itinerary'], df_itineraries['Price [EUR]']))  # Fares per itinerary
D = dict(zip(df_itineraries['Itinerary'], df_itineraries['Demand']))  # Demand per itinerary
# Initialize delta as an empty dictionary
delta = {} # 1 if flight (leg) i belongs to the path p; 0 otherwise
for _, row in df_itineraries.iterrows():
    itinerary = row['Itinerary']
    if pd.notna(row['Flight 1']):  # If Flight 1 is not NaN, add to delta
        delta[(row['Flight 1'], itinerary)] = 1
    if pd.notna(row['Flight 2']):  # If Flight 2 is not NaN, add to delta
        delta[(row['Flight 2'], itinerary)] = 1  # Flight-path inclusion
b = {(row['From Itinerary'], row['To Itinerary']): row['Recapture Rate'] for _, row in df_recapture.iterrows()}  # Recapture rates
Qi = {i: sum(delta.get((i, p), 0) * D[p] for p in P) for i in L} # Unconstrained demand on flight (leg) i

m = gp.Model('AirlinePlanningAssignment1_Q2')

# Decision Variables
t = m.addVars(P, P, lb=0, vtype=GRB.CONTINUOUS, name="t")  # Passengers reallocated from p to r

# Derived Variables
x_pp = {p: D[p] - gp.quicksum(t[p, r] for r in P if r != p) for p in P}  # Passengers on preferred itinerary
x_rp = {(p, r): b.get((p, r), 0) * t[p, r] for p in P for r in P}  # Passengers redirected to r
x = m.addVars(P, P, name="x", lb=0, vtype=GRB.INTEGER)  # Passengers from p to r

# Objective Function: Maximize revenue
m.setObjective(
    gp.quicksum(fare[r] * x[p, r] for p in P for r in P if (p, r) in b),
    GRB.MAXIMIZE
)

# Constraints
# Capacity constraints
for i in L:
    m.addConstr(
        gp.quicksum(delta.get((i, r), 0) * x[p, r] for p in P for r in P if (p, r) in b) <= CAP[i],
        name=f"Capacity_{i}"
    )

# Demand constraints
for p in P:
    m.addConstr(
        gp.quicksum(x[p, r] for r in P if (p, r) in b) <= D.get(p, 0),
        name=f"Demand_{p}"
    )

# # Recapture constraints: Sum of reallocated passengers / recapture rate <= demand for each itinerary p
# for p in P:
#     m.addConstr(
#         gp.quicksum(x[p, r] / b.get((p, r), 1) for r in P if (p, r) in b) <= D.get(p, 0),
#         name=f"Recapture_{p}"
#     )

# Non-negativity constraint
for p in P:
    for r in P:
        if (p, r) in b:  # Ensure the pair exists in recapture data
            m.addConstr(x[p, r] >= 0, name=f"NonNegativity_{p}_{r}")

# Optimize the model
m.optimize()

# Output results
if m.status == GRB.OPTIMAL:
    print(f"Optimal Objective Value: {m.objVal}")
    for p in P:
        for r in P:
            if (p, r) in b and x[p, r].x > 0:  # Show non-zero allocations
                print(f"Passengers from {p} to {r}: {x[p, r].x}")
else:
    print("Optimization was not successful.")
