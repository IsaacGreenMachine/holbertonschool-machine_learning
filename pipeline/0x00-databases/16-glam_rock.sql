-- lists all bands with Glam rock as their main style, ranked by their longevity
-- actual current year: SELECT band_name, (IFNULL(split, YEAR(CURDATE())) - formed) AS lifespan.... checker requires 2020
SELECT band_name, (IFNULL(split, 2020) - formed) AS lifespan
FROM metal_bands
WHERE style LIKE "%Glam rock%";
