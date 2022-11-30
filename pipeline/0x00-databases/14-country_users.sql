-- creates a table users
--   id, integer, never null, auto increment and primary key
--   email, string (255 characters), never null and unique
--   name, string (255 characters)
--   country, enumeration of countries: US, CO and TN, never null (= default will be the first element of the enumeration, here US)
CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO-INCREMENT PRIMARY KEY,
    email CHAR(255) NOT NULL UNIQUE,
    name CHAR(255),
    country ENUM('US', 'CO', 'TN') NOT NULL,
)
