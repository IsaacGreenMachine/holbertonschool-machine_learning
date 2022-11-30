-- displays the average temperature (Fahrenheit) grouped by city in descending temperature order
SELECT city, AVG(value) as avg_temp FROM temperatures GROUP BY city ORDER BY AVG(value) DESC;
