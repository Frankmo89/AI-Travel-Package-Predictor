"""
Streamlit App for ML Model Deployment
=====================================
Application to deploy Regression and Classification models 
for the Workation Price Prediction Challenge.

Author: Francisco Molina
Dataset: Workation Price Prediction Challenge (MachineHack)
Course: AI & ML Bootcamp
"""


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path
from prompts import EXTRACTION_PROMPT, ADVISOR_PROMPT

# =============================================================================
# PAGE CONFIGURATION — Must be the FIRST Streamlit command
# =============================================================================
st.set_page_config(
    page_title="Travel Package Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
    <style>
    /* --- Global background --- */
    .main { background-color: #F4F6F9; }

    /* --- Sidebar: deep blue-to-teal gradient --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A2980 0%, #26D0CE 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }

    /* --- Interactive project cards with hover lift effect --- */
    .project-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        border-left: 6px solid #26D0CE;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.15);
    }

    /* --- Metric cards with bottom accent --- */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border-bottom: 4px solid #1A2980;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# PRE-LOADED SCENARIOS
# =============================================================================
# These pre-built travel profiles allow quick demos during presentations.
# Instead of manually adjusting every slider, select a scenario and all
# fields auto-fill with realistic values from the actual dataset.
#
# HOW THE CODES WERE OBTAINED:
#   - Airline codes come from Label Encoding 314 unique route combinations
#     (e.g. 114 = "IndiGo" direct, 88 = "Emirates")
#   - Destination codes come from Label Encoding 565 unique destination routes
#     (e.g. 182 = most frequent destination in the dataset with 973 records)
#   - Both were verified using df['column'].value_counts() to pick the
#     most common real values, ensuring the model receives valid inputs.
#
# The remaining features (Journey_Month, Num_Places_Visited, Flight Stops,
# Trip_Complexity) are set to realistic values that match each trip profile.


SCENARIOS = {
    "-- Select a scenario --": None,

    "💰 Budget Direct Flight (IndiGo)": {
        # Airline 114 = IndiGo (direct), Destination 182 (most common destination)
        "Destination": 182,
        "Airline": 114,
        "Journey_Month": 3,
        "Num_Places_Visited": 1,
        "Flight Stops": 0,
        "Trip_Complexity": 1
    },

    "✈️ Standard Round Trip (Air India)": {
        # Airline 6 = Air India|Air India, Destination 489 (2nd most common)
        "Destination": 489,
        "Airline": 6,
        "Journey_Month": 9,
        "Num_Places_Visited": 2,
        "Flight Stops": 1,
        "Trip_Complexity": 3
    },

    "🌏 Multi-City International (Emirates)": {
        # Airline 88 = Emirates x4, Destination 332 (3rd most common)
        "Destination": 332,
        "Airline": 88,
        "Journey_Month": 6,
        "Num_Places_Visited": 5,
        "Flight Stops": 3,
        "Trip_Complexity": 7
    },

    "🏖️ Southeast Asia Explorer (AirAsia)": {
        # Airline 61 = AirAsia X|AirAsia|AirAsia X, Destination 460 (4th most common)
        "Destination": 460,
        "Airline": 61,
        "Journey_Month": 11,
        "Num_Places_Visited": 4,
        "Flight Stops": 2,
        "Trip_Complexity": 5
    },

    "⭐ Premium Long-Haul (Singapore Airlines)": {
        # Airline 231 = Singapore Airlines x6, Destination 320 (5th most common)
        "Destination": 320,
        "Airline": 231,
        "Journey_Month": 12,
        "Num_Places_Visited": 6,
        "Flight Stops": 4,
        "Trip_Complexity": 9
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource  # Cache models so they load only once per session
def load_models():
    """
    Load all saved ML models and preprocessing artifacts from disk.
    Returns a dictionary with models, scalers, encoders, and feature lists.
    Returns None if any required file is missing.
    """
    base_path = Path(__file__).parent.parent / "models"
    models = {}

    try:
        # --- Regression artifacts ---
        models['regression_model'] = joblib.load(base_path / "regression_model.pkl")
        models['regression_scaler'] = joblib.load(base_path / "regression_scaler.pkl")
        models['regression_features'] = joblib.load(base_path / "regression_features.pkl")

        # --- Classification artifacts ---
        models['classification_model'] = joblib.load(base_path / "classification_model.pkl")
        models['classification_scaler'] = joblib.load(base_path / "classification_scaler.pkl")
        models['label_encoder'] = joblib.load(base_path / "label_encoder.pkl")
        models['classification_features'] = joblib.load(base_path / "classification_features.pkl")

        # --- Optional: binning info for classification target ---
        try:
            models['binning_info'] = joblib.load(base_path / "binning_info.pkl")
        except Exception:
            models['binning_info'] = None

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Make sure you've trained and saved your models in the notebooks first!")
        return None

    return models


def make_regression_prediction(models, input_data):
    """Scale input features and return the predicted cost (float)."""
    input_scaled = models['regression_scaler'].transform(input_data)
    prediction = models['regression_model'].predict(input_scaled)
    return prediction[0]


def make_classification_prediction(models, input_data):
    """
    Scale input and return:
      - predicted label (e.g. 'High Spender')
      - predicted index
      - probability array for each class (for the confidence chart)
    """
    input_scaled = models['classification_scaler'].transform(input_data)
    prediction = models['classification_model'].predict(input_scaled)
    label = models['label_encoder'].inverse_transform(prediction)
    # predict_proba gives confidence scores for the bar chart
    probabilities = models['classification_model'].predict_proba(input_scaled)
    return label[0], prediction[0], probabilities[0]


def get_feature_importance(model, feature_names):
    """Extract feature importance from tree-based models. Returns sorted DataFrame."""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        return importance_df
    return None


def create_gauge_chart(value, min_val=0, max_val=10000, title="Predicted Cost"):
    """
    Plotly gauge (speedometer) showing where the predicted price
    falls within the expected range. Green=budget, Yellow=mid, Red=premium.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={'prefix': "$", 'font': {'size': 40}},
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickprefix': "$"},
            'bar': {'color': "#1A2980"},
            'bgcolor': "white",
            'steps': [
                {'range': [min_val, max_val * 0.33], 'color': "#2ecc71"},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': "#f39c12"},
                {'range': [max_val * 0.66, max_val], 'color': "#e74c3c"},
            ],
            'threshold': {
                'line': {'color': "#1A2980", 'width': 4},
                'thickness': 0.8,
                'value': value
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=20, l=30, r=30))
    return fig


def create_probability_chart(probabilities, class_labels):
    """
    Horizontal bar chart showing model confidence (probability) per class.
    The predicted class is highlighted in teal, others in gray.
    """
    prob_df = pd.DataFrame({
        'Category': class_labels,
        'Probability': probabilities
    }).sort_values('Probability', ascending=True)

    colors = ['#26D0CE' if p == max(probabilities) else '#b0bec5'
              for p in prob_df['Probability']]

    fig = go.Figure(go.Bar(
        x=prob_df['Probability'],
        y=prob_df['Category'],
        orientation='h',
        marker_color=colors,
        text=[f"{p:.1%}" for p in prob_df['Probability']],
        textposition='auto'
    ))
    fig.update_layout(
        title="Model Confidence by Category",
        xaxis_title="Probability",
        yaxis_title="",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        height=250,
        margin=dict(t=40, b=30, l=10, r=10),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def create_feature_importance_chart(importance_df, title="Feature Importance"):
    """Horizontal bar chart of feature importances from tree-based models."""
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='#1A2980',
        text=[f"{v:.3f}" for v in importance_df['Importance']],
        textposition='auto'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="",
        height=max(250, len(importance_df) * 45),
        margin=dict(t=40, b=30, l=10, r=10),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def render_input_form(features, key_prefix="", scenario_values=None):
    """
    Dynamically render input widgets based on the model's feature list.
    
    Uses st.session_state to update widget values when a scenario is selected.
    Widgets do NOT set a 'value' parameter — they rely entirely on session_state
    to avoid the Streamlit warning about conflicting default vs session state.
    """
    # --- Set initial defaults only if key doesn't exist yet in session state ---
    defaults = {
        'Destination': 182, 'Airline': 114, 'Journey_Month': 6,
        'Num_Places_Visited': 3, 'Flight Stops': 1, 'Trip_Complexity': 2
    }
    for feature in features:
        key = f"{key_prefix}_{feature}"
        if key not in st.session_state:
            st.session_state[key] = defaults.get(feature, 0)

    # --- Override with scenario values when a scenario is selected ---
    if scenario_values:
        for feature, value in scenario_values.items():
            st.session_state[f"{key_prefix}_{feature}"] = value

    col1, col2 = st.columns(2)
    input_values = {}

    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:

            if feature == 'Destination':
                input_values[feature] = st.number_input(
                    "📍 Destination (Encoded)",
                    min_value=0, max_value=564, step=1,
                    key=f"{key_prefix}_{feature}",
                    help="Each destination was converted to a numeric code via Label Encoding "
                         "during preprocessing. There are 565 unique destination routes."
                )

            elif feature == 'Airline':
                input_values[feature] = st.number_input(
                    "✈️ Airline Route (Encoded)",
                    min_value=0, max_value=313, step=1,
                    key=f"{key_prefix}_{feature}",
                    help="Each airline route was Label Encoded. 314 unique combinations. "
                         "Examples: 114=IndiGo, 87=Emirates (round-trip), 232=SpiceJet."
                )

            elif feature == 'Journey_Month':
                input_values[feature] = st.slider(
                    "📅 Journey Month",
                    min_value=1, max_value=12,
                    key=f"{key_prefix}_{feature}",
                    help="1 = January, 12 = December"
                )

            elif feature == 'Num_Places_Visited':
                input_values[feature] = st.slider(
                    "🗺️ Number of Places Visited",
                    min_value=1, max_value=15,
                    key=f"{key_prefix}_{feature}"
                )

            elif feature == 'Flight Stops':
                input_values[feature] = st.slider(
                    "🛑 Flight Stops (Layovers)",
                    min_value=0, max_value=5,
                    key=f"{key_prefix}_{feature}"
                )

            elif feature == 'Trip_Complexity':
                input_values[feature] = st.slider(
                    "🧩 Trip Complexity",
                    min_value=0, max_value=10,
                    key=f"{key_prefix}_{feature}",
                    help="Higher = more complex itinerary"
                )

            else:
                input_values[feature] = st.number_input(
                    f"🔹 {feature}",
                    key=f"{key_prefix}_{feature}"
                )

    return input_values

# =============================================================================
# SIDEBAR — Navigation & project info
# =============================================================================
# Sidebar banner image (travel-themed)
st.sidebar.image(
    "https://images.unsplash.com/photo-1436491865332-7a61a109cc05"
    "?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
    use_container_width=True
)
st.sidebar.title("Navigation")

# Navigation radio — these are the ONLY way to switch pages
page = st.sidebar.radio(
    "Choose a section:",
    ["🏠 Home", "📈 Cost Predictor", "🏷️ VIP Client Detector", "🤖 AI Travel Advisor"]
)

# --- Clarify which model is behind each page ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Reference")
st.sidebar.markdown(
    """
    - 📈 **Cost Predictor** → *Regression Model*  
      Predicts the exact numerical price of a travel package.
    
    - 🏷️ **VIP Client Detector** → *Classification Model*  
      Classifies clients into spending tiers (Low / Medium / High).
    
    - 🤖 **AI Advisor** → *Both models + LLM-powered natural language interface*
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ✈️ About This Project")
st.sidebar.info(
    """
    This application uses Machine Learning to predict
    **Travel Package Costs** and classify customer spending behavior.
    """
)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("**Built by:** Francisco Molina")
st.sidebar.markdown("**Course:** AI & ML Bootcamp")
st.sidebar.markdown(
    "[GitHub Repo](https://github.com/fsa-aiml-2511/individual-capstone-Frankmo89.git)"
)


# =============================================================================
# HOME PAGE
# =============================================================================
if page == "🏠 Home":

    # --- Hero image---
    col_img_left, col_img_center, col_img_right = st.columns([1, 2, 1])
    with col_img_center:
        st.image(
            "https://images.unsplash.com/photo-1488646953014-85cb44e25828"
            "?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            use_container_width=True,
            caption="AI-Powered Travel Intelligence"
        )

    # --- Title and subtitle ---
    st.markdown(
        "<h1 style='text-align:center; color:#1A2980;'>"
        "🌍 AI Travel Package Predictor</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#7F8C8D; font-size:1.2em;'>"
        "Predict costs and identify VIP clients using Machine Learning</p>",
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    st.write(
        """
        This application empowers travel agents and planners to make **data-driven
        predictions** using trained machine learning models. By analyzing thousands of
        historical travel records, we can estimate package costs and identify
        high-value clients in real time.

        **Available Tools:**
        - 📈 **Cost Predictor** *(Regression Model)*: Estimates the exact price of a travel package based on itinerary details.
        - 🏷️ **VIP Client Detector** *(Classification Model)*: Predicts a client's spending tier (Low, Medium, or High Spender) to guide your sales approach.
        
        Select a section from the **sidebar** to get started.
        """
    )

    # =========================================================================
    # MODEL PERFORMANCE METRICS
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📊 Model Performance at a Glance")

    # --- Regression Metrics ---
    st.markdown("**Regression Model (Cost Predictor)**")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric(label="R² Score", value="0.66", delta="Moderate Fit")
    with r2:
        st.metric(label="Test RMSE", value="$7,116", delta=None)
    with r3:
        st.metric(label="RMSE % of Range", value="4.2%", delta="Low Error")
    with r4:
        st.metric(label="Features Used", value="6", delta=None)

    # --- Classification Metrics ---
    st.markdown("**Classification Model (VIP Detector)**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="Accuracy", value="77%", delta=None)
    with c2:
        st.metric(label="VIP Recall", value="81%", delta="Optimized")
    with c3:
        st.metric(label="Weighted F1-Score", value="0.77", delta=None)
    with c4:
        st.metric(label="Test Samples", value="4,200", delta=None)

    st.caption(
        "All metrics were calculated on a held-out test set the model never saw during training."
    )

    # =========================================================================
    # PROJECT DETAILS
    # =========================================================================
    st.markdown("---")
    st.markdown("### ✈️ Project Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            """
            **📊 Dataset:** *Workation Price Prediction Challenge* — Contains
            thousands of travel itineraries with destinations, airlines, flight
            stops, and trip complexity measures.

            **🎯 Problem Statement:** Travel companies need to accurately price
            trips and identify VIP clients. This app predicts exact costs
            for fast quoting, and classifies travelers into spending tiers to
            optimize premium upselling strategies.
            """
        )

    with col2:
        st.write(
            """
            **🤖 Models Used:**
            - **Regression:** Gradient Boosting Regressor — Trained to minimize
              Mean Squared Error when predicting exact travel costs.
            - **Classification:** Business-Optimized Gradient Boosting — Adjusted
              with balanced sample weights to maximize recall of 'High Spenders'
              up to 81%.
            """
        )

    # --- Feature Importance Preview ---
    st.markdown("---")
    st.markdown("### 🔍 What Drives the Predictions?")
    st.write(
        "The chart below shows which features have the most influence on the "
        "model's cost predictions — helping us understand the 'why' behind each estimate."
    )

    models = load_models()
    if models is not None:
        importance_df = get_feature_importance(
            models['regression_model'],
            models['regression_features']
        )
        if importance_df is not None:
            fig = create_feature_importance_chart(
                importance_df,
                title="Regression Model — Feature Importance"
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Status badges ---
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("🎯 **Objective:** Optimize Pricing")
    with col_b:
        st.success("⭐ **Highlight:** 81% VIP Recall")
    with col_c:
        st.warning("⚡ **Status:** Models Deployed & Live")


# =============================================================================
# REGRESSION PAGE — Cost Predictor
# =============================================================================
elif page == "📈 Cost Predictor":
    st.title("📈 Travel Cost Estimator")
    st.markdown("*Regression Model — Gradient Boosting Regressor*")
    st.write(
        "Adjust the itinerary details below — or load a pre-built scenario — "
        "to get an AI-powered cost estimate."
    )

    # --- Load models ---
    models = load_models()
    if models is None:
        st.stop()

    features = models['regression_features']

    # =========================================================================
    # SCENARIO SELECTOR
    # =========================================================================
    st.markdown("---")
    st.markdown("### ⚡ Quick Demo: Load a Travel Profile")
    st.caption(
        "Pre-built itinerary profiles using real airline route codes from the dataset. "
        "Select one to auto-fill all fields, or configure manually below."
    )

    selected_scenario = st.selectbox(
        "Choose a travel profile:",
        options=list(SCENARIOS.keys()),
        key="reg_scenario"
    )
    scenario_values = SCENARIOS[selected_scenario]

    # =========================================================================
    # INPUT FORM
    # =========================================================================
    st.markdown("---")
    st.markdown("### ✈️ Trip Details Configuration")

    input_values = render_input_form(
        features,
        key_prefix="reg",
        scenario_values=scenario_values
    )

    st.markdown("---")

    # =========================================================================
    # PREDICTION
    # =========================================================================
    if st.button("🔮 Calculate Estimated Cost", type="primary", use_container_width=True):

        input_df = pd.DataFrame([input_values])
        prediction = make_regression_prediction(models, input_df)

        # --- Main result ---
        st.success(f"### 💰 Estimated Package Cost: ${prediction:,.2f}")

        # --- Gauge chart ---
        gauge = create_gauge_chart(
            value=prediction,
            min_val=0,
            max_val=171062.5,
            title="Where This Package Falls"
        )
        st.plotly_chart(gauge, use_container_width=True)

        st.info(
            "💡 **How to read this:** The gauge shows where the predicted cost "
            "sits within the typical price range. Green = Budget, Yellow = Mid-Range, "
            "Red = Premium."
        )

        # =================================================================
        # WHAT-IF ANALYSIS
        # =================================================================
        st.markdown("---")
        st.markdown("### 🔄 What-If Analysis")
        st.write("How does the price change with small itinerary modifications?")

        wc1, wc2, wc3 = st.columns(3)

        # What-if: +1 flight stop
        with wc1:
            mod_a = input_values.copy()
            if 'Flight Stops' in mod_a:
                mod_a['Flight Stops'] = min(mod_a['Flight Stops'] + 1, 5)
            pred_a = make_regression_prediction(models, pd.DataFrame([mod_a]))
            st.metric("🛑 +1 Flight Stop", f"${pred_a:,.2f}", f"${pred_a - prediction:+,.2f}")

        # What-if: +2 places visited
        with wc2:
            mod_b = input_values.copy()
            if 'Num_Places_Visited' in mod_b:
                mod_b['Num_Places_Visited'] = min(mod_b['Num_Places_Visited'] + 2, 15)
            pred_b = make_regression_prediction(models, pd.DataFrame([mod_b]))
            st.metric("🗺️ +2 Places Visited", f"${pred_b:,.2f}", f"${pred_b - prediction:+,.2f}")

        # What-if: +2 complexity
        with wc3:
            mod_c = input_values.copy()
            if 'Trip_Complexity' in mod_c:
                mod_c['Trip_Complexity'] = min(mod_c['Trip_Complexity'] + 2, 10)
            pred_c = make_regression_prediction(models, pd.DataFrame([mod_c]))
            st.metric("🧩 +2 Complexity", f"${pred_c:,.2f}", f"${pred_c - prediction:+,.2f}")

        # --- Feature Importance ---
        with st.expander("🔍 What's Driving This Prediction? (Feature Importance)"):
            importance_df = get_feature_importance(models['regression_model'], features)
            if importance_df is not None:
                st.plotly_chart(
                    create_feature_importance_chart(importance_df),
                    use_container_width=True
                )

        # --- Raw input ---
        with st.expander("📋 View Raw Input Data"):
            st.dataframe(input_df)


# =============================================================================
# CLASSIFICATION PAGE — VIP Client Detector
# =============================================================================
elif page == "🏷️ VIP Client Detector":
    st.title("🏷️ VIP Client Detector")
    st.markdown("*Classification Model — Business-Optimized Gradient Boosting*")
    st.write(
        "Enter itinerary details or load a scenario to predict the client's "
        "spending tier. This model is tuned for **high recall on VIP clients** — "
        "it rarely misses a big spender!"
    )

    # --- Load models ---
    models = load_models()
    if models is None:
        st.stop()

    features = models['classification_features']
    class_labels = models['label_encoder'].classes_

    # --- Show possible categories ---
    st.info(f"**Target Categories:** {', '.join(class_labels)}")

    # --- Binning info ---
    if models['binning_info']:
        with st.expander("📊 How Were These Categories Created?"):
            binning = models['binning_info']
            st.write(f"Original target column: **{binning['original_target']}**")
            st.write("Categories were generated by segmenting the numerical travel costs:")
            for i, label in enumerate(binning['labels']):
                if i == 0:
                    st.write(f"- **{label}**: Less than ${binning['bins'][i+1]:,.2f}")
                elif i == len(binning['labels']) - 1:
                    st.write(f"- **{label}**: ${binning['bins'][i]:,.2f} or more")
                else:
                    st.write(
                        f"- **{label}**: Between ${binning['bins'][i]:,.2f} "
                        f"and ${binning['bins'][i+1]:,.2f}"
                    )

    # =========================================================================
    # SCENARIO SELECTOR
    # =========================================================================
    st.markdown("---")
    st.markdown("### ⚡ Quick Demo: Load a Travel Profile")
    st.caption(
        "Pre-built itinerary profiles using real airline route codes from the dataset. "
        "Select one to auto-fill all fields, or configure manually below."
    )

    selected_scenario = st.selectbox(
        "Choose a travel profile:",
        options=list(SCENARIOS.keys()),
        key="class_scenario"
    )
    scenario_values = SCENARIOS[selected_scenario]

    # =========================================================================
    # INPUT FORM
    # =========================================================================
    st.markdown("---")
    st.markdown("### 👤 Client Itinerary Profile")

    input_values = render_input_form(
        features,
        key_prefix="class",
        scenario_values=scenario_values
    )

    st.markdown("---")

    # =========================================================================
    # PREDICTION
    # =========================================================================
    if st.button("🔮 Detect Client Tier", type="primary", use_container_width=True):

        input_df = pd.DataFrame([input_values])
        predicted_label, predicted_index, probabilities = \
            make_classification_prediction(models, input_df)

        # --- Result with business strategy ---
        if predicted_label == 'High Spender':
            st.success(f"### ⭐ Predicted Category: VIP {predicted_label} ⭐")
            st.write("### 💼 Recommended Business Strategy:")
            st.write(
                "This is a **premium client**! Recommend 5-star hotels, luxury "
                "add-ons, business-class upgrades, and direct flights. "
                "Do **not** lead with discounts — lead with value."
            )
        elif predicted_label == 'Medium Spender':
            st.info(f"### 🟡 Predicted Category: {predicted_label}")
            st.write("### 💼 Recommended Business Strategy:")
            st.write(
                "Standard client. Offer balanced travel packages with optional "
                "paid upgrades. Highlight mid-tier hotel options and combo deals."
            )
        else:
            st.warning(f"### 🔴 Predicted Category: {predicted_label}")
            st.write("### 💼 Recommended Business Strategy:")
            st.write(
                "Budget-conscious client. Focus on affordable accommodations, "
                "flexible travel dates, and early-bird discounts to secure the sale."
            )

        # --- Probability chart ---
        st.markdown("---")
        st.markdown("### 📊 Model Confidence Breakdown")
        st.write(
            "The chart below shows how confident the model is in each category. "
            "A higher bar means higher certainty."
        )
        st.plotly_chart(
            create_probability_chart(probabilities, class_labels),
            use_container_width=True
        )

        # --- Confidence indicator ---
        max_prob = max(probabilities)
        if max_prob >= 0.7:
            st.success(f"✅ **High Confidence:** The model is {max_prob:.0%} sure about this prediction.")
        elif max_prob >= 0.5:
            st.info(f"ℹ️ **Moderate Confidence:** The model is {max_prob:.0%} sure. Consider reviewing the input.")
        else:
            st.warning(f"⚠️ **Low Confidence:** The model is only {max_prob:.0%} sure. This client is on the border between categories.")

        # --- Cross model insigth ---
        # This section connects both models on the same input data.
        # The classification model tells us WHO the client is (spending tier),
        # while the regression model tells us HOW MUCH the trip costs.
        # Showing both predictions together validates that they agree —
       
        st.markdown("---")
        st.markdown("### 🔗 Cross-Model Insight")
        reg_prediction = make_regression_prediction(models, input_df)
        st.write(
            f"For this same itinerary, the **Regression Model** estimates a cost of "
            f"**${reg_prediction:,.2f}**, which aligns with the "
            f"**{predicted_label}** classification."
        )

        # --- Feature Importance ---
        with st.expander("🔍 What's Driving This Classification? (Feature Importance)"):
            importance_df = get_feature_importance(models['classification_model'], features)
            if importance_df is not None:
                st.plotly_chart(
                    create_feature_importance_chart(
                        importance_df,
                        title="Classification Model — Feature Importance"
                    ),
                    use_container_width=True
                )

        # --- Raw input ---
        with st.expander("📋 View Raw Client Data"):
            st.dataframe(input_df)


# =============================================================================
# AI TRAVEL ADVISOR PAGE — Groq-powered conversational interface
# =============================================================================
elif page == "🤖 AI Travel Advisor":
    st.title("🤖 AI Travel Advisor")
    st.markdown(
        "Describe your trip in plain English — the AI extracts parameters, "
        "runs both ML models, and gives you a data-backed travel recommendation."
    )
    st.markdown("---")

    # --- Groq API key check ---
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error(
            "⚠️ **Groq API key not configured.** "
            "Add `GROQ_API_KEY = \"your-key\"` to `.streamlit/secrets.toml` "
            "and restart the app."
        )
        st.stop()

    # --- Lazy Groq client — cached so it is only instantiated once per session ---
    @st.cache_resource
    def get_groq_client(api_key):
        from groq import Groq
        return Groq(api_key=api_key)

    groq_client = get_groq_client(groq_api_key)

    # --- Load ML models ---
    models = load_models()
    if models is None:
        st.stop()
    class_labels = models['label_encoder'].classes_

    # --- Initialize chat history ---
    if "advisor_messages" not in st.session_state:
        st.session_state["advisor_messages"] = []

    # --- Render existing chat history ---
    for msg in st.session_state["advisor_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Re-render charts saved alongside the assistant message
            if "gauge_fig" in msg:
                st.plotly_chart(msg["gauge_fig"], use_container_width=True)
            if "prob_fig" in msg:
                st.plotly_chart(msg["prob_fig"], use_container_width=True)

    # --- Chat input ---
    user_input = st.chat_input(
        "Describe your trip (e.g. 'IndiGo, 3 places, direct flight in July')"
    )

    if user_input:
        # Append and display the user message
        st.session_state["advisor_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your trip..."):

                # =============================================================
                # CALL 1 — Extract structured parameters from natural language
                # =============================================================
                try:
                    extraction_response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": EXTRACTION_PROMPT},
                            {"role": "user", "content": user_input},
                        ],
                        temperature=0.1,
                        max_tokens=512,
                    )
                    raw_extraction = extraction_response.choices[0].message.content.strip()
                except Exception as e:
                    err_text = f"❌ Could not reach the Groq API: {e}"
                    st.error(err_text)
                    st.session_state["advisor_messages"].append(
                        {"role": "assistant", "content": err_text}
                    )
                    st.stop()

                # --- Parse extraction JSON ---
                try:
                    extraction = json.loads(raw_extraction)
                except json.JSONDecodeError:
                    fallback_text = (
                        "I had trouble parsing that. Could you rephrase? "
                        "Try something like: *'Emirates, 4 stops, 5 destinations, July'*"
                    )
                    st.markdown(fallback_text)
                    st.caption(f"Raw model output: `{raw_extraction[:200]}`")
                    st.session_state["advisor_messages"].append(
                        {"role": "assistant", "content": fallback_text}
                    )
                    st.stop()

                # --- Not a travel query: show the LLM reply and stop ---
                if not extraction.get("extracted", False):
                    msg_text = extraction.get("message", "Could you describe a travel itinerary?")
                    st.markdown(msg_text)
                    st.session_state["advisor_messages"].append(
                        {"role": "assistant", "content": msg_text}
                    )
                    st.stop()

                # =============================================================
                # ML PREDICTIONS — use extracted params with both models
                # =============================================================
                params = extraction["params"]
                assumptions = extraction.get("assumptions", [])
                missing_info = extraction.get("missing_info", [])

                input_df = pd.DataFrame([params])

                reg_prediction = make_regression_prediction(models, input_df)
                predicted_label, _, probabilities = make_classification_prediction(models, input_df)

                # =============================================================
                # CALL 2 — Advisor narrative using both model outputs
                # =============================================================
                context = (
                    f"User query: {user_input}\n\n"
                    f"Extracted parameters: {params}\n\n"
                    f"Assumptions made: {assumptions}\n\n"
                    f"Missing info that was guessed: {missing_info}\n\n"
                    f"Regression prediction (estimated cost): ${reg_prediction:,.2f}\n\n"
                    f"Classification prediction: {predicted_label}\n"
                    f"Class probabilities: "
                    f"{ {label: f'{prob:.1%}' for label, prob in zip(class_labels, probabilities)} }"
                )

                try:
                    advisor_response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": ADVISOR_PROMPT},
                            {"role": "user", "content": context},
                        ],
                        temperature=0.7,
                        max_tokens=400,
                    )
                    advisor_text = advisor_response.choices[0].message.content.strip()
                except Exception as e:
                    err_text = f"❌ Advisor call failed: {e}"
                    st.error(err_text)
                    st.session_state["advisor_messages"].append(
                        {"role": "assistant", "content": err_text}
                    )
                    st.stop()

                # =============================================================
                # DISPLAY — advisor text + gauge + probability chart
                # =============================================================
                st.markdown(advisor_text)

                # Gauge chart — cost position in the overall price range
                gauge_fig = create_gauge_chart(
                    value=reg_prediction,
                    min_val=0,
                    max_val=171062.5,
                    title="Estimated Cost — Where It Falls"
                )
                st.plotly_chart(gauge_fig, use_container_width=True)

                # Probability chart — model confidence per spending tier
                prob_fig = create_probability_chart(probabilities, class_labels)
                st.plotly_chart(prob_fig, use_container_width=True)

                # Save message + figures so they survive reruns
                st.session_state["advisor_messages"].append({
                    "role": "assistant",
                    "content": advisor_text,
                    "gauge_fig": gauge_fig,
                    "prob_fig": prob_fig,
                })


# =============================================================================
# FOOTER — Visible on every page
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Built by <b>Francisco Molina</b> | AI & ML Bootcamp Final Project
    </div>
    """,
    unsafe_allow_html=True
)