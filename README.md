
1. **Clone the Repository**
    ```bash
    git clone https://github.com/alexisgabrysch/mlops-td.git
    cd mlops-td
    ```
2. **Run the Application**
    ```bash
    docker-compose --build up
    ```

## Usage

1. **Data Input Tab:**
    - Enter text data in the input field.
    - Click the "Click me" button to send the data to the backend.

2. **Data Visualization Tab:**
    - View the data in a table format.
    - Explore interactive bar charts representing the distribution of fruits.

3. **Prediction Model Tab:**
    - Adjust the sliders for sepal length, sepal width, petal length, and petal width.
    - Click "Predict" to see the predicted Iris species based on the input measurements.


# API Endpoints

- `GET http://http://127.0.0.1/:8000/add/{text}`: Adds the input text to the database.
- `GET http://http://127.0.0.1/:8000/list`: Retrieves the list of fruits.
- `POST http://http://127.0.0.1/:8000/predict`: Sends feature data for prediction.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
