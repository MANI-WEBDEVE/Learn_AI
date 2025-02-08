-- create a database in sql
CREATE DATABASE sqldatabase;

-- use the database
USE sqldatabase;

-- create a table in database
CREATE TABLE usertable(
_id INT primary key,
name VARCHAR(50),
email VARCHAR(50),
age INT,
salary FLOAT
);
-- drop the table
DROP TABLE userstable;

-- insert the data in table
INSERT INTO usertable VALUE(001, 'Muhammad Inam', 'inam@GMail.com', 18,9099.9);
INSERT INTO usertable VALUE(002, 'Muhammad Tahir', 'tahir@gmail.com', 16,100000.9);
-- see the data in the table
SELECT * FROM usertable;

-- see the particular field
SELECT name FROM usertable;









