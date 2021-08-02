--  Best genre
-- script that lists all genres in the database hbtn_0d_tvshows_rate by their rating.

SELECT tv_genres.name AS name, SUM(rate) AS rating 
FROM tv_show_genres, tv_genres
INNER JOIN tv_show_ratings
WHERE tv_show_genres.show_id = tv_show_ratings.show_id AND tv_show_genres.genre_id = tv_genres.id
GROUP BY tv_genres.name
ORDER BY  rating  DESC;
