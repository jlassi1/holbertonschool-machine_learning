--  In and not out
-- script that creates a table users use same requirements

CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO_INCREMENT, 
    email varchar(256) NOT NULL UNIQUE,
    name varchar(256),
    country ENUM('US', 'CO', 'TN') DEFAULT 'US' NOT NULL,
    PRIMARY KEY (id));
