-- creates a trigger that resets the attribute valid_email only when the email has been changed.
DROP TRIGGER IF EXISTS reset_email;
CREATE TRIGGER reset_email
BEFORE UPDATE ON users
FOR EACH ROW
-- IF OLD.email != NEW.email
IF STRCMP(old.email, new.email) != 0
THEN SET NEW.valid_email = 0;
END IF;
