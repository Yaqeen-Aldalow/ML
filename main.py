from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key="super-secret-key")

templates = Jinja2Templates(directory="templates")

movies_df = pd.read_csv("clean_movies.csv")
ratings_df = pd.read_csv("clean_ratings.csv")
users_df = pd.read_csv("clean_users.csv")

def complete_poster_path(path):
    if pd.isna(path) or not path.startswith('/'):
        return 'https://via.placeholder.com/500x750.png?text=No+Image+Available'
    return f'https://image.tmdb.org/t/p/w500{path}'

movies_df['poster_path'] = movies_df['poster_path'].apply(complete_poster_path)

tfidf = TfidfVectorizer(stop_words='english')
movies_df['genres'] = movies_df['genre_ids'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = movies_df.index[movies_df['title'] == title].tolist()[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return movies_df.iloc[movie_indices]
    except IndexError:
        raise HTTPException(status_code=404, detail="Movie not found")

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    global users_df, ratings_df  
    try:
        request.session["username"] = username
        if username not in users_df['user_id'].values:
            new_user = pd.DataFrame({'user_id': [username], 'age': [0], 'join_date': ['2024-12-18']})
            users_df = pd.concat([users_df, new_user], ignore_index=True)
            users_df.to_csv('clean_users.csv', index=False)
        search_history = ratings_df[ratings_df['user_id'] == username]['movie_id'].tolist()
        request.session["search_history"] = [int(x) for x in search_history]
        return RedirectResponse(url="/home", status_code=302)
    except Exception as e:
        return HTMLResponse(content=f"Error during login: {e}", status_code=500)

@app.post("/logout", response_class=HTMLResponse)
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=302)

@app.get("/home", response_class=HTMLResponse)
async def home_page(request: Request):
    global movies_df 
    try:
        username = request.session.get("username")
        if not username:
            return RedirectResponse(url="/")
        
        search_history = request.session.get("search_history", [])
        recommended_movies = pd.DataFrame()

        if search_history:
            last_searched_movie_id = search_history[-1]
            last_searched_movie = movies_df[movies_df['id'] == last_searched_movie_id]['title'].values[0]
            recommended_movies = get_recommendations(last_searched_movie, cosine_sim)
        else:
            recommended_movies = movies_df.sample(10)

        return templates.TemplateResponse("home.html", {"request": request, "movies": recommended_movies.to_dict(orient='records')})
    except Exception as e:
        return HTMLResponse(content=f"Error during home page load: {e}", status_code=500)

@app.post("/recommend", response_class=HTMLResponse)
async def recommend_movies(request: Request, title: str = Form(...)):
    global movies_df, ratings_df  
    try:
        if title not in movies_df['title'].values:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        recommendations = get_recommendations(title, cosine_sim)
        search_history = request.session.get("search_history", [])
        movie_id = int(movies_df[movies_df['title'] == title]['id'].values[0])  # Convert int64 to int
        search_history.append(movie_id)
        request.session["search_history"] = search_history
        new_entry = pd.DataFrame({'user_id': [request.session.get("username")], 'movie_id': [movie_id], 'rating': [5], 'timestamp': ['2024-12-18']})
        ratings_df = pd.concat([ratings_df, new_entry], ignore_index=True)
        ratings_df.to_csv('clean_ratings.csv', index=False)
        
        selected_movie = movies_df[movies_df['title'] == title].to_dict(orient='records')[0]
        return templates.TemplateResponse("movie_detail.html", {"request": request, "movie": selected_movie, "recommendations": recommendations.to_dict(orient='records')})
    except Exception as e:
        return HTMLResponse(content=f"Error during recommendation: {e}", status_code=500)
