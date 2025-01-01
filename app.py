# app.py
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# API URL الخاص بـ FastAPI
API_URL = "http://localhost:8000"  # تأكد من أن FastAPI يعمل على هذا المنفذ

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        user_id = request.form.get('user_id', type=int) 
        n_recommendations = request.form.get('n_recommendations', 5, type=int)  
        
        
        response = requests.post(
            f"{API_URL}/recommend/",  
            json={"user_id": user_id, "n_recommendations": n_recommendations}
        )
        response.raise_for_status() 

        recommendations = response.json()
        return jsonify(recommendations)  
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request to FastAPI failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
