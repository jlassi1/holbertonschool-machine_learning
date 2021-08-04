--  Average weighted score 
-- script that creates a stored procedure ComputeAverageWeightedScoreForUser that computes and store the average weighted score for a student.
DELIMITER //

DROP PROCEDURE IF EXISTS ComputeAverageWeightedScoreForUser;
CREATE Procedure ComputeAverageWeightedScoreForUser(IN user_id INT)
    BEGIN
        SET @avg = (SELECT SUM(corrections.score * projects.weight) / SUM(projects.weight) FROM corrections
            INNER JOIN projects ON projects.id = corrections.project_id
            WHERE corrections.user_id = user_id);
        UPDATE users SET average_score = @avg WHERE id = user_id;

    END//
DELIMITER ;
