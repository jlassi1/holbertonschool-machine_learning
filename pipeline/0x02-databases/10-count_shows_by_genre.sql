-- Number of shows by genre 
-- script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.
SELECT tv_genres.name AS genre, COUNT(*) AS number_of_shows
FROM tv_show_genres
INNER JOIN tv_genres
WHERE tv_genres.id = tv_show_genres.show_id
GROUP BY tv_genres.name 
ORDER BY  number_of_shows DESC;