--  lists all shows contained in hbtn_0d_tvshows that have at least one genre linked
-- joining tv_shows with columns (id, title), tv_show_genres with columns (show_id, genre_id), and tv_genres with columns (id, name)

SELECT title, tv_show_genres.genre_id FROM tv_shows
RIGHT JOIN tv_show_genres
ON tv_shows.id = tv_show_genres.show_id
ORDER BY tv_shows.title, tv_show_genres.genre_id;
