
CREATE TABLE employess(
id INT ,
name VARCHAR(30),
city VARCHAR(30),
salary INT DEFAULT 50000
);

INSERT INTO employess 
(id, name, city,salary)
VALUE 
(001, "Inam1", "karachi", salary),
(002, "Inam2", "punjab", 20000),
(003, "Inam3", "karachi", 10000),
(004, "Inam4", "Queta", 10000),
(005, "Inam5", "punjab", 60000),
(006, "Inam6", "karachi", 38091),
(007, "Inam7", "korangi", 90000);

DROP TABLE employess;

SELECT DISTINCT city FROM employess; 
SELECT COUNT(DISTINCT city) FROM employess; 


-- Where Clause its conditional operators and define condition to filter the recored

SELECT * 
FROM employess
WHERE city = "korangi";

SELECT * 
FROM employess
WHERE salary > 90000;

SELECT * 
FROM employess
WHERE salary >= 90000;


SELECT * 
FROM employess
WHERE salary > 5000 AND city = "korangi";

SELECT * 
FROM employess
WHERE salary > 5000 AND city = "new York";

SELECT * 
FROM employess
WHERE salary >= 100000 OR city = "punjab";
	     -- this salary not exist

SELECT *
FROM employess
WHERE city IN( "new york");
	-- this city not exist in our table to return null 

SELECT *
FROM employess
WHERE city NOT IN("punjab");

SELECT * 
FROM employess
WHERE salary BETWEEN 10000 AND 38091;

-- LIMIT CLAUSE
SELECT * 
FROM employess
LIMIT 3;

SELECT * 
FROM employess
WHERE salary > 15000
LIMIT 4;

-- ORDER CLAUES
-- order is two types first ASC second one DESC

SELECT *
FROM employess
ORDER BY name DESC;

SELECT *
FROM employess
ORDER BY salary DESC;

-- Aggregate Funations

SELECT MAX(salary)
FROM employess;

SELECT MIN(salary)
FROM employess;

SELECT AVG(salary)
FROM employess;

SELECT COUNT(salary)
FROM employess;

-- GROPU CLAUSE

SELECT  city , COUNT(*) AS total_emp
FROM employess
GROUP BY city;

SELECT city , AVG(salary) as emp_sal
FROM employess
GROUP BY city;

SELECT salary , COUNT(*) as emp_
FROM employess
GROUP BY salary;

-- HAVING Clause
SELECT city, COUNT(*) 
FROM employess
GROUP BY city
HAVING  count(*)> 2;


-- UPDATES CLAUSE
UPDATE employess
SET name = 'tahir'
WHERE name = "Inam4";

SET SQL_SAFE_UPDATES = 0;

SELECT * FROM employess;

SELECT DISTINCT salary 
FROM employess
ORDER BY salary DESC 
LIMIT 1 OFFSET 2;

SELECT MAX(salary)
FROM employess
where salary < (
SELECT MAX(salary)
FROM employess
WHERE salary < (
SELECT MAX(salary)
FROM employess
)
) 





















