import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ==========================================

# CASCADEGUARD AI

# AI-POWERED SOFTWARE FAILURE PREDICTION

# ==========================================

st.set_page_config(
page_title="CascadeGuard AI",
page_icon="🛡️",
layout="wide"
)

st.title("🛡️ CascadeGuard AI")
st.subheader("Predict Failures. Trace Causes. Recommend Recovery.")

st.markdown(
"""
CascadeGuard AI demonstrates how machine learning and service
dependency analysis can help identify potential cascading failures.

```
**Demo mode:** All service metrics are simulated. The model is
trained on synthetic data and is not validated for production use.
"""
```

)

# ==========================================

# 1. GENERATE SYNTHETIC TRAINING DATA

# ==========================================

@st.cache_resource
def train_failure_model():
rng = np.random.default_rng(42)
n = 10000

```
cpu = rng.beta(2, 3, n) * 100
memory = rng.beta(2, 3, n) * 100
latency = rng.lognormal(4.2, 0.7, n)
errors = rng.exponential(1.5, n)
failures = rng.poisson(0.4, n)

degraded = rng.random(n) < 0.18

cpu[degraded] = rng.uniform(75, 100, degraded.sum())
memory[degraded] = rng.uniform(75, 100, degraded.sum())
latency[degraded] *= rng.uniform(3, 8, degraded.sum())
errors[degraded] += rng.uniform(5, 20, degraded.sum())
failures[degraded] += rng.integers(1, 6, degraded.sum())

cpu = np.clip(cpu, 0, 100)
memory = np.clip(memory, 0, 100)
latency = np.clip(latency, 1, 10000)
errors = np.clip(errors, 0, 100)

score = (
    0.25 * cpu / 100
    \+ 0.25 * memory / 100
    \+ 0.20 * np.clip(latency / 1000, 0, 1)
    \+ 0.20 * np.clip(errors / 20, 0, 1)
    \+ 0.10 * np.clip(failures / 5, 0, 1)
)

probability = 1 / (1 + np.exp(-12 * (score - 0.55)))
labels = rng.binomial(1, np.clip(probability, 0.01, 0.99))

X = pd.DataFrame({
    "cpu_usage": cpu,
    "memory_usage": memory,
    "latency_ms": latency,
    "error_rate": errors,
    "recent_failures": failures
})

y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

return model, accuracy, auc
```

with st.spinner("Initializing the synthetic-data ML model..."):
model, model_accuracy, model_auc = train_failure_model()

features = [
"cpu_usage",
"memory_usage",
"latency_ms",
"error_rate",
"recent_failures"
]

# ==========================================

# 2. SERVICE DEPENDENCY GRAPH

# An edge A -> B means A depends on B.

# ==========================================

G = nx.DiGraph()

services = [
"API_Gateway",
"Authentication",
"Payment",
"Order_Service",
"Database"
]

G.add_nodes_from(services)

G.add_edges_from([
("API_Gateway", "Authentication"),
("API_Gateway", "Payment"),
("Payment", "Order_Service"),
("Order_Service", "Database")
])

# ==========================================

# 3. INCIDENT SCENARIO SELECTOR

# ==========================================

st.sidebar.header("Incident Simulator")

scenario = st.sidebar.selectbox(
"Select a scenario",
[
"Normal Operation",
"Database Failure",
"Payment Failure",
"API Gateway Overload"
]
)

apply_recovery = st.sidebar.checkbox(
"Simulate recovery",
value=False
)

normal_metrics = pd.DataFrame(
[
[25, 40, 50, 0.2, 0],
[30, 45, 60, 0.3, 0],
[35, 50, 80, 0.5, 0],
[30, 45, 70, 0.4, 0],
[40, 50, 90, 0.5, 0]
],
index=services,
columns=features
)

scenario_metrics = normal_metrics.copy()

root_service = None

if scenario == "Database Failure":
root_service = "Database"

```
scenario_metrics.loc["Database"] = [96, 93, 950, 18, 4]
scenario_metrics.loc["Order_Service"] = [75, 72, 350, 8, 2]
scenario_metrics.loc["Payment"] = [82, 78, 450, 12, 3]
```

elif scenario == "Payment Failure":
root_service = "Payment"

```
scenario_metrics.loc["Payment"] = [98, 85, 1200, 25, 6]
scenario_metrics.loc["API_Gateway"] = [80, 70, 400, 8, 2]
scenario_metrics.loc["Order_Service"] = [70, 65, 300, 6, 2]
```

elif scenario == "API Gateway Overload":
root_service = "API_Gateway"

```
scenario_metrics.loc["API_Gateway"] = [98, 90, 1000, 22, 5]
scenario_metrics.loc["Payment"] = [75, 70, 350, 6, 2]
scenario_metrics.loc["Authentication"] = [70, 65, 250, 5, 1]
```

# ==========================================

# 4. SIMULATED RECOVERY

# ==========================================

metrics = scenario_metrics.copy()

if apply_recovery and root_service is not None:
metrics.loc[root_service] = [40, 50, 100, 0.5, 0]

```
if root_service == "Database":
    metrics.loc["Order_Service"] = [35, 45, 80, 0.5, 0]
    metrics.loc["Payment"] = [40, 50, 100, 0.5, 0]

elif root_service == "Payment":
    metrics.loc["Payment"] = [40, 50, 100, 0.5, 0]
    metrics.loc["Order_Service"] = [35, 45, 80, 0.5, 0]

elif root_service == "API_Gateway":
    metrics.loc["API_Gateway"] = [40, 50, 100, 0.5, 0]
```

# ==========================================

# 5. PREDICT SERVICE FAILURE RISK

# ==========================================

risk_probabilities = model.predict_proba(metrics[features])[:, 1] * 100

risk_table = metrics.copy()
risk_table["Service"] = risk_table.index
risk_table["Failure Risk (%)"] = risk_probabilities.round(2)

risk_table["Risk Level"] = np.select(
[
risk_probabilities >= 70,
risk_probabilities >= 40
],
[
"HIGH",
"MEDIUM"
],
default="LOW"
)

# ==========================================

# 6. DASHBOARD SUMMARY

# ==========================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric(
"Services Monitored",
len(services)
)

col2.metric(
"High-Risk Services",
int((risk_probabilities >= 70).sum())
)

col3.metric(
"Scenario",
scenario
)

if apply_recovery and root_service is not None:
st.success("Recovery simulation enabled. Displayed metrics are hypothetical.")

st.header("📊 Service Health and Failure Risk")

display_columns = [
"Service",
"cpu_usage",
"memory_usage",
"latency_ms",
"error_rate",
"recent_failures",
"Failure Risk (%)",
"Risk Level"
]

st.dataframe(
risk_table[display_columns].sort_values(
"Failure Risk (%)",
ascending=False
),
use_container_width=True,
hide_index=True
)

# ==========================================

# 7. FAILURE RISK CHART

# ==========================================

st.header("📈 Predicted Failure Risk by Service")

chart_data = risk_table[["Failure Risk (%)"]].sort_values(
"Failure Risk (%)",
ascending=False
)

st.bar_chart(chart_data)

# ==========================================

# 8. CASCADING IMPACT ANALYSIS

# ==========================================

st.header("🔗 Service Dependency and Cascade Analysis")

affected_services = (
list(nx.ancestors(G, root_service))
if root_service is not None
else []
)

if root_service is None:
st.info("Normal Operation: no initiating failure has been selected.")

else:
st.write("**Simulated initiating service:**", root_service)

```
if apply_recovery:
    st.write("Recovery has been simulated for the initiating service.")

st.write("**Potentially affected dependent services:**")

if affected_services:
    st.write(", ".join(affected_services))
else:
    st.write("No dependent services are present in this graph.")
```

position = {
"API_Gateway": (0, 2),
"Authentication": (2, 3),
"Payment": (2, 1),
"Order_Service": (4, 1),
"Database": (6, 1)
}

node_colors = [
"tomato" if node == root_service
else "orange" if node in affected_services
else "lightblue"
for node in G.nodes()
]

fig, ax = plt.subplots(figsize=(11, 5))

nx.draw_networkx(
G,
pos=position,
node_color=node_colors,
node_size=3000,
font_size=8,
arrows=True,
arrowsize=20,
ax=ax
)

ax.set_title("Service Dependency Graph")
ax.axis("off")

st.pyplot(fig)
plt.close(fig)

st.caption(
"Red: initiating service | Orange: potentially affected dependent | "
"Blue: other services"
)

# ==========================================

# 9. ROOT CAUSE ANALYSIS

# ==========================================

st.header("🕵️ Root Cause Analysis")

if root_service is None:
st.write("No incident has been injected in Normal Operation mode.")

else:
root_metrics = scenario_metrics.loc[root_service]

```
st.write("**Suspected initiating service:**", root_service)

st.write(
    "The initiating service is known because this is a controlled "
    "simulation. In real incidents, the initiating cause would need "
    "to be inferred from telemetry and validated."
)

st.write("**Incident metrics:**")

st.dataframe(
    root_metrics.to_frame("Observed Value"),
    use_container_width=True
)

st.write("**Diagnostic observations:**")

st.write(
    "- CPU usage: "
    + str(root_metrics["cpu_usage"])
    + "%"
)

st.write(
    "- Memory usage: "
    + str(root_metrics["memory_usage"])
    + "%"
)

st.write(
    "- Latency: "
    + str(root_metrics["latency_ms"])
    + " ms"
)

st.write(
    "- Error rate: "
    + str(root_metrics["error_rate"])
    + "%"
)
```

# ==========================================

# 10. RECOVERY RECOMMENDATIONS

# ==========================================

st.header("🛠️ Recovery Recommendations")

recommendations = {
"Database": [
"Investigate slow database queries and resource utilization.",
"Check connection pool saturation.",
"Verify healthy replicas before rerouting traffic.",
"Monitor downstream payment and order services."
],
"Payment": [
"Inspect payment timeouts and service logs.",
"Check payment provider connectivity.",
"Review retry behavior and connection pools."
],
"API_Gateway": [
"Inspect request rates and gateway errors.",
"Review rate limits and connection saturation.",
"Check downstream service latency."
],
"Order_Service": [
"Inspect order-processing errors.",
"Check database response times.",
"Review retry and timeout configuration."
],
"Authentication": [
"Inspect authentication service logs.",
"Check identity provider connectivity."
]
}

if root_service is None:
st.write("No recovery actions are needed for the normal-operation scenario.")

else:
for index, action in enumerate(
recommendations[root_service],
start=1
):
st.write(str(index) + ". " + action)

if apply_recovery and root_service is not None:
st.success(
"The simulated recovery has changed the metrics. "
"This is not an actual production recovery."
)

# ==========================================

# 11. MODEL INFORMATION

# ==========================================

with st.expander("Model details and limitations"):
st.write("**Algorithm:** Random Forest Classifier")
st.write("**Training data:** 10,000 synthetic observations")
st.write("**Features:** CPU, memory, latency, error rate, recent failures")
st.write("**Synthetic test accuracy:**", round(model_accuracy * 100, 2), "%")
st.write("**Synthetic test ROC-AUC:**", round(model_auc, 4))

```
st.warning(
    "These evaluation results come from synthetic data with "
    "artificially generated labels. They do not establish real-world "
    "production accuracy or calibrated failure probabilities."
)
```

st.markdown("---")

st.caption(
"CascadeGuard AI | Hackathon Prototype | "
"Simulated data only — not connected to production systems."
)
