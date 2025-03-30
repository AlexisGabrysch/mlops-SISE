import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn import datasets
import numpy as np
import random
import time
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Iris Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main containers */
    .main {
        background-color: #f8f9fa;
        padding: 20px;
    }
    
    /* Headers */
    h1 {
        color: #4C566A;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        font-size: 3em;
        margin-bottom: 20px;
    }
    h2 {
        color: #5E81AC;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        font-size: 2em;
        margin-top: 30px;
    }
    h3 {
        color: #81A1C1;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        margin-top: 20px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-radius: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 5px;
        padding-left: 15px;
        padding-right: 15px;
        background-color: #E5E9F0;
        color: #4C566A;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #5E81AC !important;
        color: white !important;
    }
    
    /* Cards */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 5px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 5px;
    }
    
    /* Images */
    .image-container {
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 15px;
    }
    
    /* Dataframe */
    .dataframe-container {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #E5E9F0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    /* Success message */
    .success-box {
        background-color: #A3BE8C;
        color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    /* Error message */
    .error-box {
        background-color: #BF616A;
        color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    /* Metrics */
    .metrics-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        flex: 1;
        margin-right: 10px;
    }
    .metric-box:last-child {
        margin-right: 0;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #5E81AC;
    }
    .metric-label {
        font-size: 14px;
        color: #4C566A;
        margin-top: 5px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #ECEFF4;
        color: #4C566A;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Load Iris dataset
iris = datasets.load_iris()

# Header
st.markdown("<h1>🌸 Iris Species Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='info-box'>
    <p>A modern machine learning application for predicting Iris flower species based on sepal and petal measurements.</p>
</div>
""", unsafe_allow_html=True)

# Define iris images URLs
iris_images = {
    'setosa': "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/800px-Kosaciec_szczecinkowaty_Iris_setosa.jpg",
    'versicolor': "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/800px-Iris_versicolor_3.jpg",
    'virginica': "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/800px-Iris_virginica.jpg"
}

# Create tabs with icons
tabs = st.tabs(["📊 Data Entry", "📈 Data Visualization", "🔮 Prediction", "⚙️ Model Training"])

with tabs[0]:
    st.markdown("<h2>📊 Add Data to Database</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        user_input = st.text_input("Enter a fruit name to add to the database")
        button_clicked = st.button("Add to Database", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Recent Additions</h3>", unsafe_allow_html=True)
        
        # Display the last 5 added items if available
        try:
            latest_fruits = requests.get("http://server:8000/list").json()
            if latest_fruits["results"]:
                for item in latest_fruits["results"][-5:]:
                    st.markdown(f"• {item['fruit']}")
            else:
                st.info("No recent additions")
        except:
            st.info("Connect to the server to see recent additions")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if button_clicked:
        if user_input:
            try:
                with st.spinner("Adding to database..."):
                    response = requests.get(f"http://server:8000/add/{user_input}")
                    if response.status_code == 200:
                        st.markdown(f"""
                        <div class='success-box'>
                            <h3>Success! 🎉</h3>
                            <p>Successfully added '{user_input}' to the database!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='error-box'>
                            <h3>Error ❌</h3>
                            <p>Failed to add item to database. Please try again.</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class='error-box'>
                    <h3>Error ❌</h3>
                    <p>Could not connect to server: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text before submitting")

with tabs[1]:
    st.markdown("<h2>📈 Data Visualization Dashboard</h2>", unsafe_allow_html=True)
    
    try:
        # Get fruit list data
        with st.spinner("Loading data..."):
            response = requests.get("http://server:8000/list")
            if response.status_code == 200:
                liste_fruits = response.json()
                df = pd.DataFrame(liste_fruits["results"])
                
                if not df.empty:
                    # Count occurrences of each fruit
                    fruit_counts = df['fruit'].value_counts().reset_index()
                    fruit_counts.columns = ['Fruit', 'Count']
                    
                    # Create a custom color scale
                    colors = px.colors.qualitative.Bold
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("<h3>📋 Fruit Database</h3>", unsafe_allow_html=True)
                        st.markdown("<div class='dataframe-container'>", unsafe_allow_html=True)
                        st.dataframe(df, height=300, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("<h3>📊 Distribution of Fruits</h3>", unsafe_allow_html=True)
                        fig = px.bar(fruit_counts, x='Fruit', y='Count', 
                                    color='Fruit', 
                                    color_discrete_sequence=colors,
                                    title=None)
                        fig.update_layout(
                            xaxis_title="Fruit Type", 
                            yaxis_title="Quantity",
                            height=400,
                            template="plotly_white",
                            plot_bgcolor='rgba(0, 0, 0, 0)',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif"),
                            margin=dict(l=40, r=40, t=40, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Additional visualization - pie chart
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<h3>🥧 Proportion of Each Fruit</h3>", unsafe_allow_html=True)
                    fig2 = px.pie(fruit_counts, values='Count', names='Fruit', 
                                color_discrete_sequence=colors,
                                title=None,
                                hole=0.4)
                    fig2.update_layout(
                        height=450,
                        template="plotly_white",
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif"),
                        margin=dict(l=40, r=40, t=40, b=40),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    fig2.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig2, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    
                else:
                    st.markdown("""
                    <div class='info-box'>
                        <h3>No data available</h3>
                        <p>Add some fruits in the Data Entry tab to see visualizations.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='error-box'>
                    <h3>Error ❌</h3>
                    <p>Failed to retrieve data from the server.</p>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class='error-box'>
            <h3>Error connecting to the server ❌</h3>
            <p>{str(e)}</p>
            <p>Make sure the server is running and accessible.</p>
        </div>
        """, unsafe_allow_html=True)

with tabs[2]:
    st.markdown("<h2>🔮 Iris Species Prediction</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>📏 Enter Iris Measurements</h3>", unsafe_allow_html=True)
        
        # Get the range of values from the Iris dataset
        X = iris.data
        sepal_length = st.slider("Sepal length (cm)", 
                                min_value=float(min(X[:, 0])), 
                                max_value=float(max(X[:, 0])), 
                                value=5.4, 
                                step=0.1)
        
        sepal_width = st.slider("Sepal width (cm)", 
                                min_value=float(min(X[:, 1])), 
                                max_value=float(max(X[:, 1])), 
                                value=3.4, 
                                step=0.1)
        
        petal_length = st.slider("Petal length (cm)", 
                                min_value=float(min(X[:, 2])), 
                                max_value=float(max(X[:, 2])), 
                                value=1.3, 
                                step=0.1)
        
        petal_width = st.slider("Petal width (cm)", 
                            min_value=float(min(X[:, 3])), 
                            max_value=float(max(X[:, 3])), 
                            value=0.2, 
                            step=0.1)
        
        predict_button = st.button("Predict Species", type="primary")
        
        # Create radar chart for input values
        categories = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        
        # Get average values for each species for comparison
        setosa_avg = np.mean(X[iris.target == 0], axis=0)
        versicolor_avg = np.mean(X[iris.target == 1], axis=0)
        virginica_avg = np.mean(X[iris.target == 2], axis=0)
        
        # Normalize values for better radar chart visualization
        max_vals = np.max(X, axis=0)
        min_vals = np.min(X, axis=0)
        
        norm_input = (np.array([sepal_length, sepal_width, petal_length, petal_width]) - min_vals) / (max_vals - min_vals)
        norm_setosa = (setosa_avg - min_vals) / (max_vals - min_vals)
        norm_versicolor = (versicolor_avg - min_vals) / (max_vals - min_vals)
        norm_virginica = (virginica_avg - min_vals) / (max_vals - min_vals)
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=norm_input.tolist(),
            theta=categories,
            fill='toself',
            name='Your Input',
            line=dict(color='#5E81AC', width=3),
            fillcolor='rgba(94, 129, 172, 0.3)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=norm_setosa.tolist(),
            theta=categories,
            fill='toself',
            name='Setosa Avg',
            line=dict(color='rgba(163, 190, 140, 0.8)', width=1, dash='dot'),
            fillcolor='rgba(163, 190, 140, 0.1)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=norm_versicolor.tolist(),
            theta=categories,
            fill='toself',
            name='Versicolor Avg',
            line=dict(color='rgba(235, 203, 139, 0.8)', width=1, dash='dot'),
            fillcolor='rgba(235, 203, 139, 0.1)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=norm_virginica.tolist(),
            theta=categories,
            fill='toself',
            name='Virginica Avg',
            line=dict(color='rgba(180, 142, 173, 0.8)', width=1, dash='dot'),
            fillcolor='rgba(180, 142, 173, 0.1)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            height=400,
            template="plotly_white",
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif"),
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>🌱 Prediction Result</h3>", unsafe_allow_html=True)
        
        if predict_button:
            try:
                # Make prediction request
                with st.spinner("Making prediction..."):
                    response = requests.post("http://server:8000/predict", json={
                        "sepal_length": sepal_length,
                        "sepal_width": sepal_width,
                        "petal_length": petal_length,
                        "petal_width": petal_width
                    })
                    
                    if response.status_code == 200:
                        pred_class = response.json()["prediction"]
                        
                        # Display prediction with styling
                        st.markdown(f"""
                        <div style="background-color: #A3BE8C; color: white; padding: 16px; border-radius: 10px; margin-bottom: 20px;">
                            <h3 style="margin: 0;">Predicted Species: {pred_class.capitalize()}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display image of the predicted species
                        if pred_class in iris_images:
                            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                            st.image(iris_images[pred_class], 
                                    width=300, 
                                    caption=f"Iris {pred_class.capitalize()}")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Information about the species
                            species_info = {
                                'setosa': "Iris setosa is characterized by small flowers with blue to purple petals. Native to Alaska, Maine, and northern Asia.",
                                'versicolor': "Iris versicolor (also known as Blue Flag) has purple to blue flowers and is native to eastern North America.",
                                'virginica': "Iris virginica has larger flowers with purple to blue petals and is native to eastern United States."
                            }
                            
                            st.markdown(f"""
                            <div class='info-box'>
                                <h3>About Iris {pred_class.capitalize()}</h3>
                                <p>{species_info[pred_class]}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Add a confidence indicator (purely illustrative)
                            st.markdown("<h4>Prediction Confidence</h4>", unsafe_allow_html=True)
                            # This is a mock confidence level - in a real app, you would get this from your model
                            mock_confidence = random.uniform(0.8, 0.99)
                            
                            st.progress(mock_confidence)
                            st.markdown(f"<p style='text-align: center;'>{mock_confidence:.2%}</p>", unsafe_allow_html=True)
                        else:
                            st.warning("No image available for this species")
                    else:
                        st.markdown("""
                        <div class='error-box'>
                            <h3>Error ❌</h3>
                            <p>Error making prediction. Please try again.</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class='error-box'>
                    <h3>Error ❌</h3>
                    <p>{str(e)}</p>
                    <p>Make sure the server is running and accessible.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 50px 0;">
                <img src="https://cdn-icons-png.flaticon.com/512/6295/6295417.png" width="100">
                <p style="margin-top: 20px; color: #4C566A;">Adjust the measurements and click "Predict Species" to see results</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[3]:
    st.markdown("<h2>⚙️ Model Training</h2>", unsafe_allow_html=True)

    # Current model info
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📋 Current Model Information</h3>", unsafe_allow_html=True)
    
    try:
        # Get current model info
        with st.spinner("Loading model information..."):
            response = requests.get("http://server:8000/model-info")
            if response.status_code == 200:
                model_info = response.json()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div style="background-color: #5E81AC; color: white; padding: 16px; border-radius: 10px; text-align: center;">
                        <h4 style="margin: 0; font-size: 14px;">Model Type</h4>
                        <p style="font-size: 24px; font-weight: bold; margin: 10px 0 0 0;">{}</p>
                    </div>
                    """.format(model_info['model_type']), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style="background-color: #81A1C1; color: white; padding: 16px; border-radius: 10px; text-align: center;">
                        <h4 style="margin: 0; font-size: 14px;">Accuracy</h4>
                        <p style="font-size: 24px; font-weight: bold; margin: 10px 0 0 0;">{:.2%}</p>
                    </div>
                    """.format(model_info['accuracy']), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div style="background-color: #88C0D0; color: white; padding: 16px; border-radius: 10px; text-align: center;">
                        <h4 style="margin: 0; font-size: 14px;">Last Trained</h4>
                        <p style="font-size: 18px; font-weight: bold; margin: 10px 0 0 0;">{}</p>
                    </div>
                    """.format(model_info['last_trained']), unsafe_allow_html=True)
                
                # Display model parameters
                st.markdown("<h4>Model Parameters:</h4>", unsafe_allow_html=True)
                params_df = pd.DataFrame(model_info['parameters'].items(), columns=['Parameter', 'Value'])
                st.markdown("<div class='dataframe-container'>", unsafe_allow_html=True)
                st.table(params_df)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Could not retrieve current model information")
    except Exception as e:
        st.error(f"Error connecting to server: {e}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Train new model
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>🔄 Train a New Model</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Model Parameters</h4>", unsafe_allow_html=True)
        
        model_type = st.selectbox(
            "Select Model Type",
            ["DecisionTree", "RandomForest", "SVC", "LogisticRegression", "KNeighbors"]
        )
        
        test_size = st.slider("Test Size (%)", 10, 40, 20)
        
        random_state = st.number_input("Random State", 0, 100, 42)
        
        # Model specific parameters
        if model_type == "DecisionTree":
            max_depth = st.slider("Max Depth", 1, 20, 3)
            criterion = st.selectbox("Criterion", ["gini", "entropy"])
            params = {"max_depth": max_depth, "criterion": criterion}
            
        elif model_type == "RandomForest":
            n_estimators = st.slider("Number of Estimators", 10, 200, 100)
            max_depth = st.slider("Max Depth", 1, 20, 3)
            params = {"n_estimators": n_estimators, "max_depth": max_depth}
            
        elif model_type == "SVC":
            C = st.slider("C (Regularization)", 0.1, 10.0, 1.0, 0.1)
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            params = {"C": C, "kernel": kernel}
            
        elif model_type == "LogisticRegression":
            C = st.slider("C (Regularization)", 0.1, 10.0, 1.0, 0.1)
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg"])
            params = {"C": C, "solver": solver}
            
        elif model_type == "KNeighbors":
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
            weights = st.selectbox("Weights", ["uniform", "distance"])
            params = {"n_neighbors": n_neighbors, "weights": weights}
    
    with col2:
        st.markdown("<h4>Training Information</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <p>The model will be trained on the Iris dataset which contains:</p>
            <ul>
                <li>150 samples</li>
                <li>4 features (sepal length, sepal width, petal length, petal width)</li>
                <li>3 classes (setosa, versicolor, virginica)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.image("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png", 
                caption="Iris Dataset Visual Representation")
        
        train_button = st.button("Train Model", type="primary")
    
    # Display training results
    if train_button:
        # Prepare training parameters
        training_data = {
            "model_type": model_type,
            "test_size": test_size / 100,  # Convert to fraction
            "random_state": random_state,
            "params": params
        }
        
        try:
            with st.spinner("Training model... This may take a moment."):
                # Add a progress bar for visual effect
                progress_bar = st.progress(0)
                for i in range(100):
                    # Simulate training progress
                    time.sleep(0.03)
                    progress_bar.progress(i + 1)
                
                # Send training request
                response = requests.post("http://server:8000/train", json=training_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.markdown("""
                    <div class='success-box'>
                        <h3>Model successfully trained! 🎉</h3>
                        <p>Your new model is now active and ready for predictions.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics
                    st.markdown("<h4>Training Results</h4>", unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">{result['accuracy']:.2%}</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">{result['f1_score']:.2%}</div>
                            <div class="metric-label">F1 Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">{result['training_time']:.2f}s</div>
                            <div class="metric-label">Training Time</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display confusion matrix as a heatmap
                    st.markdown("<h4>Confusion Matrix</h4>", unsafe_allow_html=True)
                    cm = result['confusion_matrix']
                    cm_df = pd.DataFrame(cm, 
                                        index=['Actual Setosa', 'Actual Versicolor', 'Actual Virginica'], 
                                        columns=['Predicted Setosa', 'Predicted Versicolor', 'Predicted Virginica'])
                    
                    fig = px.imshow(cm_df, 
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=cm_df.columns,
                                   y=cm_df.index,
                                   text_auto=True,
                                   color_continuous_scale='Blues')
                    fig.update_layout(
                        height=400,
                        template="plotly_white",
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif"),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display classification report
                    st.markdown("<h4>Classification Report</h4>", unsafe_allow_html=True)
                    st.code(result['classification_report'])
                    
                else:
                    st.markdown(f"""
                    <div class='error-box'>
                        <h3>Error training model ❌</h3>
                        <p>{response.text}</p>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div class='error-box'>
                <h3>Error ❌</h3>
                <p>{str(e)}</p>
                <p>Make sure the server is running and accessible.</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    <p>🌸 Iris Predictor ML App • Created by Alexis GABRYSCH • {}</p>
</div>
""".format(datetime.now().year), unsafe_allow_html=True)