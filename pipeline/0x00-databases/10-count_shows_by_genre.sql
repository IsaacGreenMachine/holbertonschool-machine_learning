-- lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.
-- display format: <TV Show genre> - <Number of shows linked to this genre>

SELECT tv_genres.name AS genre, COUNT(tv_shows.title) AS number_of_shows
FROM tv_show_genres
LEFT JOIN tv_shows
ON tv_show_genres.show_id = tv_shows.id
LEFT JOIN tv_genres
ON tv_genres.id = tv_show_genres.genre_id
GROUP BY genre_id
ORDER BY COUNT(tv_shows.title) DESC;
