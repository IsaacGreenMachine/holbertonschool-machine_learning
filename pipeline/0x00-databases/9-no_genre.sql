-- lists all shows contained in hbtn_0d_tvshows without a genre linked.
-- display format: tv_shows.title - tv_show_genres.genre_id

SELECT title, tv_show_genres.genre_id from tv_shows
LEFT JOIN tv_show_genres
ON tv_shows.id = tv_show_genres.show_id
WHERE tv_show_genres.genre_id IS NULL
ORDER BY tv_shows.title, tv_show_genres.genre_id;