import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
# If any of this libraries is missing from your computer,
# please install them using pip.

filename = 'Flight_Delays_2018.csv'

# ARR_DELAY is the column name that should be used as dependent variable (Y).

df = pd.read_csv(filename)

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nArrival Delay Statistics:")
print(df["ARR_DELAY"].describe())

airline_counts = df["OP_CARRIER_NAME"].value_counts()  

print("\nTop 10 Airlines by Number of Delayed Flights:")
print(airline_counts.head(10))

airline_delays = df.groupby("OP_CARRIER_NAME")["ARR_DELAY"].agg(
    ["count", "mean", "median"]
)

airline_delays = airline_delays.sort_values(
    by="count", ascending=False
)

print("\nArrival Delay Statistics by Airline:")
print(airline_delays.head(10))

american_df = df[
    df["OP_CARRIER_NAME"] == "American Airlines Inc."
].copy()

print("\nAmerican Airlines Flight Count:")
print(len(american_df))

predictors = [
    "DEP_DELAY",
    "TAXI_OUT",
    "DISTANCE",
    "CRS_ELAPSED_TIME"
]

print("\nAmerican Airlines Descriptive Statistics:")
print(american_df[predictors + ["ARR_DELAY"]].describe())

print("\nCorrelations with Arrival Delay:")
correlations = american_df[
    predictors + ["ARR_DELAY"]
].corr()["ARR_DELAY"]

print(correlations.sort_values(ascending=False))

plt.figure(figsize=(9, 5))
plt.hist(american_df["ARR_DELAY"], bins=50,
         edgecolor="black")
plt.title("Distribution of American Airlines Arrival Delays")
plt.xlabel("Arrival Delay (Minutes)")
plt.ylabel("Number of Flights")
plt.xlim(0, 300)
plt.tight_layout()
plt.savefig("arrival_delay_distribution.png")
plt.show()


plt.figure(figsize=(9, 5))
plt.scatter(american_df["DEP_DELAY"],
            american_df["ARR_DELAY"],
            alpha=0.2, s=8)
plt.title("Departure Delay vs. Arrival Delay")
plt.xlabel("Departure Delay (Minutes)")
plt.ylabel("Arrival Delay (Minutes)")
plt.tight_layout()
plt.savefig("departure_vs_arrival.png")
plt.show()

X = american_df[
    ["DEP_DELAY", "TAXI_OUT", "DISTANCE", "CRS_ELAPSED_TIME"]
]

Y = american_df["ARR_DELAY"]

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

print("\nOLS Regression Results:")
print(model.summary())

print("\nRegression Coefficients:")
print(model.params)

american_df["PREDICTED_DELAY"] = model.predict(X)

plt.figure(figsize=(9, 5))

plt.scatter(
    american_df["ARR_DELAY"],
    american_df["PREDICTED_DELAY"],
    alpha=0.2,
    s=8
)

plt.plot([0, 1600], [0, 1600], "r--",
         label="Perfect Prediction")

plt.title("Actual vs. Predicted Arrival Delays")
plt.xlabel("Actual Arrival Delay (Minutes)")
plt.ylabel("Predicted Arrival Delay (Minutes)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()
