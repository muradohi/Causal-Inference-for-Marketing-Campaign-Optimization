# Databricks notebook source
# %pip install causalml dowhy scikit-learn pandas numpy

# COMMAND ----------

# %pip install openpyxl

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

dataset = pd.read_excel("/Volumes/pandas/file_sch/file_vol/Dataset.xlsx")

# COMMAND ----------

dataset.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Does exposure to this campaign increase the chance that the user converts and buys something?**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Q1. Is treatment randomly assigned? Or is there bias in who gets exposed?**

# COMMAND ----------

dataset.groupby("treatment_exposed")[["prior_visits_30d","prior_spend_180d"]].mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Q2: Which variables are confounders?**

# COMMAND ----------

# MAGIC %md
# MAGIC **Numerical Data**

# COMMAND ----------

dataset.groupby("treatment_exposed")["prior_spend_180d"].mean()

# COMMAND ----------

dataset.groupby("conversion")["prior_spend_180d"].mean()

# COMMAND ----------

# MAGIC %md
# MAGIC **Categorical Data**

# COMMAND ----------

dataset.groupby("treatment_exposed")[["prior_visits_30d","prior_spend_180d"]].mean()


# COMMAND ----------

dataset.groupby(["country", "treatment_exposed"])["conversion"].mean()


# COMMAND ----------

# MAGIC %md
# MAGIC ## **Q3: Identify mediators**

# COMMAND ----------

# MAGIC %md
# MAGIC - If these strongly change after treatment -- they are mediators.
# MAGIC - Do not include them in X for causal ML.

# COMMAND ----------

dataset.groupby("treatment_exposed")[["clicks", "impressions", "spend_usd"]].mean()


# COMMAND ----------

# MAGIC %md
# MAGIC ## **Q4: Does treatment actually change conversion?**

# COMMAND ----------

dataset.groupby("treatment_exposed")["conversion"].mean()

# COMMAND ----------

dataset.groupby(["segment", "treatment_exposed"])["conversion"].mean()

# COMMAND ----------

dataset.groupby(["channel","treatment_exposed"])["conversion"].mean()

# COMMAND ----------

dataset.groupby(["country","treatment_exposed"])["conversion"].mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Q5: Is the effect different across groups? (Heterogeneous effects)**

# COMMAND ----------

dataset["spend_group"] = pd.qcut(dataset["prior_spend_180d"], q=3, precision=2)
dataset["spend_group_name"] = np.where(dataset["spend_group"].astype(str) == "(1.5, 68.86]", "Low", np.where(dataset["spend_group"].astype(str) == "(68.86, 134.01]", "Medium", "High"))
dataset.head()

# COMMAND ----------

dataset.groupby(["spend_group", "treatment_exposed"])["conversion"].mean()

# COMMAND ----------

dataset.groupby(["campaign_id", "treatment_exposed"])["conversion"].mean().unstack()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **1️⃣ Is the campaign effective overall?**

# COMMAND ----------

dataset.groupby("treatment_exposed")["conversion"].mean()


# COMMAND ----------

float((dataset.groupby("treatment_exposed")["conversion"].mean()[1] - dataset.groupby("treatment_exposed")["conversion"].mean()[0]).round(3))

# COMMAND ----------

ate_naive = (
    dataset.loc[dataset["treatment_exposed"]==1, "conversion"].mean()
    - dataset.loc[dataset["treatment_exposed"]==0, "conversion"].mean()
)
print("Naive ATE (unadjusted):", ate_naive.round(3))


# COMMAND ----------

# MAGIC %md
# MAGIC **--Exposed users convert more – but this is correlation, not yet clean causation.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## **2️⃣ In which countries is it most effective?**

# COMMAND ----------

country_conv = (
    dataset.groupby(["country", "treatment_exposed"])["conversion"]
      .mean()
      .unstack()
)
country_conv["difference"] = country_conv[1] - country_conv[0]
country_conv.sort_values("difference", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **3️⃣ Is it worth targeting low-spend users?**

# COMMAND ----------

dataset.groupby("spend_group_name")["treatment_exposed"].mean()


# COMMAND ----------

spend_eff = dataset.groupby(["spend_group_name", "treatment_exposed"])["conversion"].mean().unstack()
spend_eff["diff"] = spend_eff[1] - spend_eff[0]
spend_eff.sort_values("diff", ascending=False)


# COMMAND ----------

# MAGIC %md
# MAGIC **Campaign doesn’t help low-spend users much → maybe don’t target them.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### **4️⃣ Which segment benefits the most?**

# COMMAND ----------

dataset.groupby(["segment", "treatment_exposed"])["conversion"].mean().unstack()


# COMMAND ----------

# MAGIC %md
# MAGIC # **Propensity Score**

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
# Outcome
y = dataset["treatment_exposed"]

# Variables: treatment + confounders
X = dataset[[
    "prior_spend_180d",
    "prior_visits_30d",
    "country",
    "device",
    "channel",
    "segment"
]]

X = pd.get_dummies(X, drop_first=True)
X = X.astype(float)

# COMMAND ----------

model = LogisticRegression(max_iter=1000).fit(X, y)
dataset["propensity_score"] = model.predict_proba(X)[:, 1]
dataset.display()

# COMMAND ----------

dataset["propensity_class"]= pd.qcut(dataset["propensity_score"], q=3, labels=["low", "medium", "high"])

# COMMAND ----------

dataset.groupby(["propensity_class","treatment_exposed"])["conversion"].mean().unstack()

# COMMAND ----------

# MAGIC %md
# MAGIC # **Regression controlling for confounders**

# COMMAND ----------

import pandas as pd
import statsmodels.api as sm


# Outcome
y = dataset["conversion"]

# Variables: treatment + confounders
X = dataset[[
    "treatment_exposed",
    "prior_spend_180d",
    "prior_visits_30d",
    "country",
    "device",
    "channel",
    "segment"
]]

# Convert categorical variables into numbers
X = pd.get_dummies(X, drop_first=True)

# Add intercept column
X = sm.add_constant(X)
X = X.astype(float)


# COMMAND ----------

logit_model = sm.Logit(y, X)
result = logit_model.fit()

print(result.summary())

# COMMAND ----------

# Make two copies of your dataset
df_control = dataset.copy()
df_treat = dataset.copy()

# Set treatment explicitly
df_control["treatment_exposed"] = 0
df_treat["treatment_exposed"] = 1

# Rebuild X matrix exactly like you did for the regression
X_control = pd.get_dummies(df_control[[
    "treatment_exposed",
    "prior_spend_180d",
    "prior_visits_30d",
    "country",
    "device",
    "channel",
    "segment",
    "spend_group_name"
]], drop_first=True)

X_treat = pd.get_dummies(df_treat[[
    "treatment_exposed",
    "prior_spend_180d",
    "prior_visits_30d",
    "country",
    "device",
    "channel",
    "segment",
    "spend_group_name"
]], drop_first=True)

# Align columns (important!)
X_control = X_control.reindex(columns=result.model.exog_names, fill_value=0)
X_treat = X_treat.reindex(columns=result.model.exog_names, fill_value=0)

# Convert to float
X_control = X_control.astype(float)
X_treat = X_treat.astype(float)

# Predict probabilities
df_control["pred_conv_control"] = result.predict(X_control)
df_treat["pred_conv_treat"] = result.predict(X_treat)

# Combine predictions
dataset["pred_control"] = df_control["pred_conv_control"]
dataset["pred_treat"] = df_treat["pred_conv_treat"]
dataset["pred_effect"] = dataset["pred_treat"] - dataset["pred_control"]


# COMMAND ----------

model_effect_by_group = (
    dataset
    .groupby("spend_group_name")[["pred_effect"]]
    .mean()
)

manual_effect_by_group = (
    dataset
    .groupby(["spend_group_name", "treatment_exposed"])["conversion"]
    .mean()
    .unstack()
)

manual_effect_by_group["manual_diff"] = (
    manual_effect_by_group[1] - manual_effect_by_group[0]
)

print("===== Manual Effects =====")
print(manual_effect_by_group)

print("\n===== Model Adjusted Effects =====")
print(model_effect_by_group)


# COMMAND ----------

dataset.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **S-Leraner**

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# COMMAND ----------

X = pd.get_dummies(dataset[[
    "prior_spend_180d",
    "prior_visits_30d",
    "country",
    "device",
    "channel",
    "segment"
]], drop_first=True)

X = X.astype(float)
y = dataset["conversion"]
T = dataset["treatment_exposed"]

# COMMAND ----------

X_s = X.copy()
X_s["treatment_exposed"] = T
y_s = y.copy()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_s, y_s)

# COMMAND ----------

X_s_control = X_s.copy()
X_s_treat = X_s.copy()

X_s_control["treatment_exposed"] = 0
X_s_treat["treatment_exposed"] = 1

pred_s_control = rf.predict_proba(X_s_control)[:, 1]
pred_s_treat = rf.predict_proba(X_s_treat)[:,1]
pred_s_effect = pred_s_treat - pred_s_control

dataset["s_learner_effect"] = pred_s_effect
dataset.display()

# COMMAND ----------

dataset[["s_learner_effect"]].describe()

# COMMAND ----------

target = dataset[dataset["s_learner_effect"] > 0.10].sort_values("s_learner_effect", ascending=False)
target.display()

# COMMAND ----------

# User A: no treatment, no conversion, high predicted effect
userA = dataset[
    (dataset["treatment_exposed"] == 0) &
    (dataset["conversion"] == 0) &
    (dataset["s_learner_effect"] > 0.5)
].iloc[0]

# User B: treated, converted, high predicted effect
userB = dataset[
    (dataset["treatment_exposed"] == 1) &
    (dataset["conversion"] == 1) &
    (dataset["s_learner_effect"] > 0.5)
].iloc[0]

print(userA)
print("\n")
print(userB)



# COMMAND ----------

compare_cols = [
    "prior_spend_180d",
    "prior_visits_30d",
    "segment",
    "channel",
    "device",
    "country"
]

print(userA[compare_cols])
print("\n")
print(userB[compare_cols])


# COMMAND ----------

# MAGIC %md
# MAGIC ## **T-Leraner**

# COMMAND ----------

# Separate treated and control
X_treated = X[T == 1]
y_treated = y[T == 1]

X_control = X[T == 0]
y_control = y[T == 0]
print(X_treated.shape)
print(X_control.shape)



# COMMAND ----------

# Train two models
model_treated = RandomForestClassifier(n_estimators=200, random_state=42)
model_control = RandomForestClassifier(n_estimators=200, random_state=42)

model_treated.fit(X_treated, y_treated)
model_control.fit(X_control, y_control)

# Predict both worlds
pred_treated_outcome = model_treated.predict_proba(X)[:, 1]
pred_control_outcome = model_control.predict_proba(X)[:, 1]

# Individual treatment effect
dataset["T_learner_effect"] = pred_treated_outcome - pred_control_outcome

# COMMAND ----------

dataset[["s_learner_effect", "T_learner_effect"]].corr()

# COMMAND ----------

# MAGIC %md
# MAGIC ### **1. Who benefited most from the treatment?**

# COMMAND ----------

# Top 10% by S-learner effect
top_s = dataset.nlargest(int(len(dataset)*0.1), "s_learner_effect")

# Top 10% by T-learner effect
top_t = dataset.nlargest(int(len(dataset)*0.1), "T_learner_effect")


# COMMAND ----------

# MAGIC %md
# MAGIC ### **2. Who should we target next?**

# COMMAND ----------

future_targets = dataset[
    (dataset["treatment_exposed"] == 0) &
    (dataset["conversion"] == 0) &
    (dataset["s_learner_effect"] > 0.1) &
    (dataset["T_learner_effect"] > 0.1)
]

future_targets.sort_values("s_learner_effect", ascending=False).head(20)

# COMMAND ----------

top_s_idx = set(top_s.index)
top_t_idx = set(top_t.index)

common_high = top_s_idx.intersection(top_t_idx)

common_users = dataset.loc[list(common_high)]


# COMMAND ----------

# Check patterns
common_users["segment"].value_counts(normalize=True)

# COMMAND ----------

common_users["channel"].value_counts(normalize=True)

# COMMAND ----------

common_users["country"].value_counts(normalize=True)

# COMMAND ----------

common_users["device"].value_counts(normalize=True)

# COMMAND ----------

dataset["uplift_score"] = dataset["T_learner_effect"]  # or s_learner_effect

# COMMAND ----------

threshold = -0.02  # more than -2% drop in conversion probability

bad_to_treat = dataset[
    dataset["uplift_score"] < threshold
]

# COMMAND ----------

print(bad_to_treat["segment"].value_counts(normalize=True))
print("\n")
print(bad_to_treat["channel"].value_counts(normalize=True))
print("\n")
print(bad_to_treat["country"].value_counts(normalize=True))
print("\n")
print(bad_to_treat["device"].value_counts(normalize=True))

# COMMAND ----------

bad_segments = dataset.groupby("segment")["uplift_score"].mean().sort_values()

print(bad_segments)

# COMMAND ----------

print(dataset.groupby("segment")["uplift_score"].mean().sort_values())
print(dataset.groupby("channel")["uplift_score"].mean().sort_values())
print(dataset.groupby("device")["uplift_score"].mean().sort_values())
print(dataset.groupby("country")["uplift_score"].mean().sort_values())

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Validate targeting quality**

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Uplift Curve – Are we targeting the right people?**

# COMMAND ----------

import numpy as np
import pandas as pd

df = dataset.copy()

# 1) Sort by predicted uplift, descending
df = df.sort_values("uplift_score", ascending=False).reset_index(drop=True)

# 2) Create a percentile rank (what fraction of users we have seen)
df["percentile"] = (np.arange(len(df)) + 1) / len(df)  # from 0 to 1

# 3) For cumulative slices, compute treated vs control conversions
df["treated"] = df["treatment_exposed"] == 1
df["control"] = df["treatment_exposed"] == 0

# cumulative sums
df["cum_treated"] = df["treated"].cumsum()
df["cum_control"] = df["control"].cumsum()

df["cum_conv_treated"] = (df["treated"] & (df["conversion"] == 1)).cumsum()
df["cum_conv_control"] = (df["control"] & (df["conversion"] == 1)).cumsum()

# Avoid division by zero
df["rate_treated"] = df["cum_conv_treated"] / df["cum_treated"].replace(0, np.nan)
df["rate_control"] = df["cum_conv_control"] / df["cum_control"].replace(0, np.nan)

# 4) Uplift at each percentile = extra conversions vs if they had control rate
df["uplift_curve"] = (df["rate_treated"] - df["rate_control"])


# COMMAND ----------

import matplotlib.pyplot as plt

plt.plot(df["percentile"], df["uplift_curve"])
plt.xlabel("Fraction of population targeted")
plt.ylabel("Estimated uplift (conversion rate diff)")
plt.title("Uplift Curve (model-based)")
plt.grid(True)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### **Qini Curve – How much absolute gain vs doing nothing?**

# COMMAND ----------

df = dataset.copy()
df = df.sort_values("uplift_score", ascending=False).reset_index(drop=True)

# treated & control masks
treated_mask = df["treatment_exposed"] == 1
control_mask = df["treatment_exposed"] == 0

# We need average conversion rate in control group
p_control = df.loc[control_mask, "conversion"].mean()

# cumulative treated users and conversions
df["cum_treated"] = treated_mask.cumsum()
df["cum_conv_treated"] = (treated_mask & (df["conversion"] == 1)).cumsum()

# Expected conversions if those treated users had not been treated = cum_treated * baseline control rate
df["expected_conv_if_control"] = df["cum_treated"] * p_control

# Qini curve: extra conversions vs baseline
df["qini"] = df["cum_conv_treated"] - df["expected_conv_if_control"]

df["percentile"] = (np.arange(len(df)) + 1) / len(df)


# COMMAND ----------

plt.plot(df["percentile"], df["qini"])
plt.xlabel("Fraction of population targeted")
plt.ylabel("Extra conversions vs baseline")
plt.title("Qini Curve")
plt.grid(True)
plt.show()


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### **1. Core Idea: Decision Metric**
# MAGIC ### Expected Profit=Uplift×Avg Revenue−Cost

# COMMAND ----------

avg_revenue = dataset[dataset["conversion"] == 1]["revenue_usd"].mean()

dataset["expected_revenue"] = dataset["uplift_score"] * avg_revenue
dataset["expected_profit"] = dataset["expected_revenue"] - dataset["spend_usd"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### **2. Segment-wise Decision**

# COMMAND ----------

tot_sample_size = dataset["user_id"].nunique()
segment_decision = dataset.groupby("segment").agg(
    avg_uplift = ("uplift_score", "mean"),
    avg_profit = ("expected_profit", "mean"),
    treatment_cost = ("spend_usd", "mean"),
    sample_size = ("user_id", "count")
).sort_values(by="avg_profit", ascending=False)

print(segment_decision)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **3. Device-wise Decision**

# COMMAND ----------

device_decision = dataset.groupby("device").agg(
    avg_uplift = ("uplift_score", "mean"),
    avg_profit = ("expected_profit", "mean"),
    cost = ("spend_usd", "mean"),
    count = ("user_id", "count")
).sort_values(by="avg_profit", ascending=False)

print(device_decision)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **4. Channel-wise Decision**

# COMMAND ----------

channel_decision = dataset.groupby("channel").agg(
    avg_uplift=("uplift_score", "mean"),
    avg_profit=("expected_profit", "mean"),
    avg_cost=("spend_usd", "mean"),
    impressions=("impressions", "mean"),
    clicks=("clicks", "mean"),
    count=("user_id", "count")
).sort_values(by="avg_profit", ascending=False)

print(channel_decision)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **5. Finding People to Exclude (Do NOT Treat)**

# COMMAND ----------

bad_users = dataset[dataset["expected_profit"] < 0]

bad_users.groupby(["segment", "channel"]).size().sort_values(ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Who to Target Next Campaign**

# COMMAND ----------

target_candidates = dataset[
    (dataset["expected_profit"] > 0) &
    (dataset["uplift_score"] > dataset["uplift_score"].quantile(0.75))
]

target_candidates = target_candidates.sort_values("expected_profit", ascending=False)

target_candidates.head(100)

# COMMAND ----------

