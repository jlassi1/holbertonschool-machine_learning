--  Best genre
-- script that lists all genres in the database hbtn_0d_tvshows_rate by their rating.

SELECT name AS name, SUM(rate) AS rating 
FROM tv_genres
INNER JOIN tv_show_ratings
WHERE tv_genres.id = tv_show_ratings.show_id
GROUP BY name
ORDER BY  rating  DESC;
