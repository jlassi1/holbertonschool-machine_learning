-- Safe divide 
-- script that creates a function SafeDiv that divides (and returns) the first by the second number or returns 0 if the second number is equal to 0.

DELIMITER $$
DROP FUNCTION IF EXISTS SafeDiv;
CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS FLOAT
DETERMINISTIC
BEGIN
    DECLARE divis FLOAT;
    IF b = 0 THEN  SET divis = 0;
    ELSEIF (b != 0) THEN SET divis = (a/b);
    END IF;
    -- SET div = a / b;
    RETURN(divis);
END $$

DELIMITER ;
