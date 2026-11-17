import altair as alt
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import streamlit as st

FEATURES = [f"f{i}" for i in range(12)]
MAX_ROWS = 1000  # the API accepts at most 1000 customers per request
SAMPLE_PATH = Path(__file__).parent / "sample_customers.csv"

st.set_page_config(page_title="Uplift Targeting", layout="wide")


@st.cache_data
def load_sample():
    return pd.read_csv(SAMPLE_PATH)


def call_recommend(api_url, customers, value, cost, max_contacts):
    payload = {
        "customers": customers[FEATURES].astype(float).to_dict("records"),
        "value": value,
        "cost": cost,
    }
    if max_contacts is not None:
        payload["max_contacts"] = int(max_contacts)
    r = httpx.post(f"{api_url.rstrip('/')}/recommend", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


st.title("Who should we contact?")
st.write(
    "An uplift model estimates how much a campaign changes each customer's chance of "
    "visiting. A customer is worth contacting only if the expected gain "
    "(value x predicted uplift) is larger than the cost of contacting them."
)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", "http://127.0.0.1:8000")
    value = st.number_input("Value of one incremental visit", min_value=0.01,
                            value=1.0, step=0.1)
    cost = st.number_input("Cost of contacting one customer", min_value=0.0,
                           value=0.01, step=0.005, format="%.3f")
    use_cap = st.checkbox("Limit the number of contacts (budget)")
    max_contacts = (st.number_input("Maximum contacts", min_value=0, value=50, step=10)
                    if use_cap else None)

source = st.radio("Customers", ["Demo sample (500 Criteo customers)", "Upload a CSV"],
                  horizontal=True)

if source.startswith("Demo"):
    customers = load_sample()
else:
    st.caption("The CSV needs numeric columns f0 to f11, one row per customer "
               f"(first {MAX_ROWS} rows are used).")
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is None:
        st.info("Upload a file to continue.")
        st.stop()
    customers = pd.read_csv(uploaded)
    missing = [c for c in FEATURES if c not in customers.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()
    numeric = customers[FEATURES].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        st.error("Features f0 to f11 must be numbers with no blanks.")
        st.stop()
    customers = numeric.head(MAX_ROWS)

customers = customers.reset_index(drop=True)

try:
    result = call_recommend(api_url, customers, value, cost, max_contacts)
except httpx.ConnectError:
    st.error("Could not reach the API. Start it with: uvicorn app.api:app")
    st.stop()
except httpx.HTTPStatusError as e:
    st.error(f"The API returned {e.response.status_code}: {e.response.text}")
    st.stop()

res = pd.DataFrame(result["results"])
res.insert(0, "customer", customers.index)

n = result["n_customers"]
n_contact = result["n_contact"]
profit_targeted = result["expected_total_profit"]
profit_everyone = float(res["expected_profit"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{n:,}")
c2.metric("Recommended to contact", f"{n_contact:,}", f"{100 * n_contact / n:.0f}% of customers")
c3.metric("Expected profit, targeted", f"{profit_targeted:.3f}")
c4.metric("Expected profit, contact everyone", f"{profit_everyone:.3f}")

st.caption("Profits are model estimates from predicted uplift, not measured results. "
           "Individual predictions are noisy; the ranking is more reliable than any single number.")

# Profit as we contact more customers, best first
ranked = res.sort_values("expected_profit", ascending=False).reset_index(drop=True)
curve = pd.DataFrame({
    "Share of customers contacted (%)": 100 * np.arange(1, n + 1) / n,
    "Expected profit": ranked["expected_profit"].cumsum().values,
})
st.subheader("Expected profit as we contact more customers (best first)")
chart = alt.Chart(curve).mark_line().encode(
    x=alt.X("Share of customers contacted (%):Q", scale=alt.Scale(domain=[0, 100])),
    y=alt.Y("Expected profit:Q"),
)
st.altair_chart(chart, use_container_width=True)

st.subheader("Customers")
table = res.sort_values(["contact", "expected_profit"], ascending=[False, False])
table = table.rename(columns={
    "customer": "Customer",
    "uplift": "Predicted uplift",
    "top_percent": "Top % by uplift",
    "expected_profit": "Expected profit",
    "contact": "Contact?",
})
st.dataframe(table, hide_index=True)

st.download_button(
    "Download contact list (CSV)",
    res[res["contact"]].to_csv(index=False),
    file_name="contact_list.csv",
    mime="text/csv",
)

with st.expander("How to read this"):
    st.markdown(
        "- **Predicted uplift** is the estimated change in visit probability if the customer is contacted.\n"
        "- **Top % by uplift** is the customer's rank among held-out customers; lower is better.\n"
        "- **Expected profit** = value x uplift - cost, for that customer if contacted.\n"
        "- The model was trained on Criteo randomized-experiment data. On that data a simple "
        "P(visit) baseline performed about as well in profit terms, so treat this as a "
        "targeting aid, not a guarantee."
    )
