-- Email validation to sent 
-- script that creates a trigger that resets the attribute valid_email only when the email has been changed.

DELIMITER //
CREATE TRIGGER Resets_email BEFORE UPDATE ON users
FOR EACH ROW 
BEGIN
    IF NEW.email NOT LIKE OLD.email  THEN
        SET NEW.valid_email = 0;

    END IF;
END //
DELIMITER ;
