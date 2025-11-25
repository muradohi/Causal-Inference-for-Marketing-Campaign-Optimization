# Causal Inference & Uplift Modeling

This project analyzes a real marketing campaign dataset using **causal inference and causal machine learning** to estimate true campaign impact.

---

## What I Have Done

- Performed **Exploratory Data Analysis (EDA)**  
  - Compared treated vs control groups  
  - Analyzed conversion, ROI and treatment patterns by:
    - Segment
    - Channel
    - Country
    - Device
    - Spend group

- Built **Quasi-Experimental Models**
  - Logistic regression with confounder control  
  - Propensity score modeling: P(Treatment | X)  
  - Calculated counterfactual outcomes:
    - Predicted control outcome  
    - Predicted treated outcome  

- Implemented **Causal Machine Learning Models**
  - S-Learner
  - T-Learner
  - Uplift Modeling (Random Forest)
  - Estimated **Individual Treatment Effects (ITE)** for each user

---

## Output Generated

- Individual uplift scores  
- Propensity scores  
- Counterfactual predictions  
- Segment-level treatment effects  
- Channel, device, and country-level causal insights  

---

## Tools Used

Python, Pandas, NumPy, Scikit-learn, Statsmodels

# Decision-Making Using Causal Models

This document explains how I converted causal model outputs into **business decisions**.

---

## What I Have Done

- Converted uplift scores into **expected business value**:
  
  expected_profit = uplift × avg_revenue − treatment_cost

- Assigned each user a decision:
  - Target  
  - Do not target  
  - Exclude (negative effect)

- Analyzed decisions at group level:
  - Segment-wise
  - Channel-wise
  - Country-wise
  - Device-wise

- Identified:
  - High value target segments  
  - Users who should not be treated  
  - Channels and regions to prioritize  
  - Segments and channels to avoid  

- Built a **profit-based targeting strategy** instead of relying only on conversion probability.

---

## Outcome

Created a causal decision framework to help optimize:
- Marketing targeting  
- Budget allocation  
- Campaign efficiency  
- ROI optimization  
