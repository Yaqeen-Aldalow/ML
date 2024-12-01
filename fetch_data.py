import requests
import pandas as pd

# مفتاح الـ API الخاص بك
API_KEY = "baa5d22412ad3ae24bd003e2c28afccc"

# قاعدة URL للـ API
BASE_URL = "https://api.themoviedb.org/3"

def fetch_movies(genre_id, page=1):
    """
    Fetch movies by genre.
    Args:
        genre_id (int): ID of the genre.
        page (int): Page number for pagination (default=1).
    Returns:
        List of movies (JSON).
    """
    url = f"{BASE_URL}/discover/movie"
    params = {
        "api_key": API_KEY,
        "language": "en-US",
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "include_video": "false",
        "page": page,
        "with_genres": genre_id,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def save_to_csv(movies, filename="movies.csv"):
    """
    Save movie data to a CSV file.
    Args:
        movies (list): List of movie dictionaries.
        filename (str): File name for the CSV file.
    """
    df = pd.DataFrame(movies)
    df.to_csv(filename, index=False)
    print(f"Movies saved to {filename}")

if __name__ == "__main__":
    genre_id = 35 
    all_movies = []
    
 
    for page in range(1, 6): 
        data = fetch_movies(genre_id, page)
        if data:
            all_movies.extend(data["results"])
    
    
    save_to_csv(all_movies)
