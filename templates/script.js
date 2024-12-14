// بيانات الأفلام
const movies = [
    {
        title: "Deadpool & Wolverine",
        overview: "A listless Wade Wilson toils away in civilian life...",
        release_date: "2024-07-24",
        poster_path: "/8cdWjvZQUExUUTzyp4t6EDMubfO.jpg",
        vote_average: 7.689
    },
    {
        title: "Moana 2",
        overview: "After receiving an unexpected call from her wayfinding ancestors...",
        release_date: "2024-11-27",
        poster_path: "/m0SbwFNCa9epW1X60deLqTHiP7x.jpg",
        vote_average: 7.9
    },
    {
        title: "Red One",
        overview: "After Santa Claus (codename: Red One) is kidnapped...",
        release_date: "2024-10-31",
        poster_path: "/cdqLnri3NEGcmfnqwk2TSIYtddg.jpg",
        vote_average: 6.595
    },
    {
        title: "Despicable Me 4",
        overview: "Gru and Lucy and their girls—Margo, Edith, and Agnes...",
        release_date: "2024-06-20",
        poster_path: "/wWba3TaojhK7NdycRhoQpsG0FaH.jpg",
        vote_average: 7.083
    },
    {
        title: "Inside Out 2",
        overview: "Teenager Riley's mind headquarters is undergoing a sudden demolition...",
        release_date: "2024-06-11",
        poster_path: "/vpnVM9B6NMmQpWeZvzLvDESb2QY.jpg",
        vote_average: 7.6
    },
    {
        title: "Sing: Thriller",
        overview: "Buster Moon dreams up a star-studded spectacle set to Michael Jackson's 'Thriller'...",
        release_date: "2024-10-16",
        poster_path: "/i77OInTKcrnRlAozFOaB6D5mk15.jpg",
        vote_average: 7.364
    },
    {
        title: "Bad Boys: Ride or Die",
        overview: "After their late former Captain is framed, Lowrey and Burnett try to clear his name...",
        release_date: "2024-06-05",
        poster_path: "/oGythE98MYleE6mZlGs5oBGkux1.jpg",
        vote_average: 7.5
    }
];

// الدالة لعرض الأفلام
function displayMovies(movieList) {
    const movieContainer = document.getElementById("movieContainer");
    movieContainer.innerHTML = ""; // تنظيف المحتوى القديم

    movieList.forEach(movie => {
        const movieCard = `
            <div class="col-md-4 mb-4">
                <div class="card">
                    <img src="https://image.tmdb.org/t/p/w500${movie.poster_path}" class="card-img-top" alt="${movie.title}">
                    <div class="card-body">
                        <h5 class="card-title">${movie.title}</h5>
                        <p class="card-text">${movie.overview}</p>
                        <p class="card-text"><small class="text-muted">Release Date: ${movie.release_date}</small></p>
                        <p class="card-text"><strong>Rating: ${movie.vote_average}</strong></p>
                    </div>
                </div>
            </div>
        `;
        movieContainer.innerHTML += movieCard;
    });
}

// عرض جميع الأفلام عند تحميل الصفحة
displayMovies(movies);

// البحث عن الأفلام
const searchInput = document.getElementById("searchInput");

searchInput.addEventListener("input", function () {
    const query = searchInput.value.toLowerCase(); // النص المدخل في شريط البحث
    const filteredMovies = movies.filter(movie => movie.title.toLowerCase().includes(query)); // تصفية الأفلام
    displayMovies(filteredMovies); // إعادة عرض الأفلام
});
