-- Temperatures #0
-- script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending)
SELECT city, avg(value) as avg_temp 
FROM temperatures
GROUP BY city
ORDER BY avg(value) DESC;
