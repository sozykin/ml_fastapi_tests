# FastAPI Sentiment Analysis API
This project is a simple FastAPI-based API that utilizes the Hugging Face Transformers library to perform sentiment analysis on text input. It uses a pre-trained sentiment analysis model to classify text as either positive or negative.

## Setup
1. Clone the repository:

```
git clone https://github.com/Kaktys36/ml_fastapi_tests
```

2. Install the required dependencies:

```
pip install fastapi transformers uvicorn
```

3. Start the FastAPI server:

```
uvicorn main:app --reload
```

4. Navigate to http://127.0.0.1:8000 to access the API.

## API Endpoints
- GET /: Returns a simple "Hello World" message.
- POST /predict/: Takes a JSON payload with a text field and returns the sentiment analysis prediction for that text.
## Testing
You can run the tests included in the test_main.py file by executing:

```
pytest test_main.py
```

## Dependencies
- FastAPI: A modern web framework for building APIs with Python.
- Transformers: An NLP library by Hugging Face for pre-trained models.
## License
This project is licensed under the MIT License. Feel free to use and modify it according to your needs.