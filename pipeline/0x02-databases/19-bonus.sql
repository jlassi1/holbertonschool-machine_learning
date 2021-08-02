-- Add bonus 
-- script that creates a stored procedure AddBonus that adds a new correction for a student.

DROP PROCEDURE IF EXISTS AddBonus;
DELIMITER //
CREATE PROCEDURE AddBonus (IN user_id INT, IN project_name CHAR(255), IN score INT)
    BEGIN
        DECLARE pr_id INT DEFAULT 0;
        IF NOT EXISTS(SELECT name FROM projects WHERE name = project_name) THEN
            INSERT INTO projects (name) VALUES (project_name);
        END IF;
        SET pr_id = (SELECT id FROM projects WHERE projects.name = project_name);
       INSERT INTO corrections (user_id, project_id, score) VALUES (user_id, pr_id, score);

    END//
DELIMITER ;
