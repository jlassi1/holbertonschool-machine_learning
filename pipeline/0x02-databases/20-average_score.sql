-- Average score 
-- script that creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
-- DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;

DELIMITER //
CREATE Procedure ComputeAverageScoreForUser(IN user_id INT)
    BEGIN
        DECLARE avg INT DEFAULT 0;
    
        SET avg = (SELECT AVG(score) FROM corrections WHERE corrections.user_id = user_id);
        UPDATE users SET users.average_score = avg WHERE users.id = user_id;

    END//
DELIMITER ;
