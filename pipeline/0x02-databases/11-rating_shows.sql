-- Rotten tomatoes 
-- cript that lists all shows from hbtn_0d_tvshows_rate by their rating.

SELECT title AS title, SUM(rate) AS rating 
FROM tv_shows
INNER JOIN tv_show_ratings
WHERE tv_shows.id = tv_show_ratings.show_id
GROUP BY title
ORDER BY  rating  DESC;
