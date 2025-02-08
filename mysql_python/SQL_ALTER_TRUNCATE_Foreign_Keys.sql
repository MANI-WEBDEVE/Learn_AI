CREATE DATABASE students;
USE  students;

CREATE TABLE student_cs_1(
id INT PRIMARY KEY,
name VARCHAR(50),
_id VARCHAR(90) UNIQUE
);

DROP TABLE student_cs_1;

INSERT INTO student_cs_1 
VALUE
(01, "Muhammad inam", "i"),
(02, "Muhammad Tahir", "n");

UPDATE student_cs_1
SET _id = "inam"
WHERE _id = "i";

SELECT * FROM student_cs_1;

CREATE TABLE teacher_cs_1(
id INT PRIMARY KEY,
name VARCHAR(50),
stds_id INT,
FOREIGN KEY (stds_id) REFERENCES student_cs_1(id)
ON UPDATE CASCADE
ON DELETE CASCADE,
FOREIGN KEY (name) REFERENCES student_cs_1(_id)
ON UPDATE CASCADE
ON DELETE CASCADE
);

DROP TABLE teacher_cs_1 ;

INSERT INTO teacher_cs_1 
VALUE
(01, "i", 01),
(02, "n",02);

SELECT * FROM teacher_cs_1;

-- Alter method in sql

CREATE TABLE student(
id INT PRIMARY KEY,
name VARCHAR(40),
fee INT
);

INSERT INTO student
VALUE
(001, "MUHAMMAD INAM", 9000),
(002, "Muhammad Tahir", 7000),
(003, "MUHAMMAD INAM", 9000.90);

SELECT * FROM student;

ALTER TABLE student
CHANGE COLUMN name std_name VARCHAR(30);

-----------------------
ALTER TABLE student
ADD COLUMN age INT;

---------------------
ALTER TABLE student
DROP COLUMN age;

------------------------
ALTER TABLE student
MODIFY fee FLOAT;

ALTER TABLE student 
RENAME students_1;

TRUNCATE TABLE student;

/* joins in  sql 
1. inner join
2. left join
3. right join
4. full join
*/

CREATE TABLE emp_1(
id INT NOT NULL,
name VARCHAR(50),
age INT
);

CREATE TABLE emp_2(
id INT NOT NULL,
name VARCHAR(50),
age INT,
salary FLOAT
);

INSERT INTO emp_1 
VALUE
(1, "INAM", 67),
(2, "TAHIR", 67),
(3, "KAREEM", 67),
(4, "MUZZAMMIL", 67);

INSERT INTO emp_2 
VALUE
(1, "INAM", 67,100.00),
(2, "TAHIR", 90,100.00),
(3, "KAREEM", 34, 200.90),
(4, "MUZZAMMIL", 20,8080.90);

DROP TABLE emp_2;

SELECT * FROM emp_2;

SELECT * 
FROM emp_1 AS e1
INNER JOIN emp_2 AS e2
ON e1.age = e2.age;

SELECT * 
FROM emp_1 AS e1
LEFT JOIN emp_2 AS e2
ON e1.age = e2.age;

SELECT * 
FROM emp_1 AS e1
RIGHT JOIN emp_2 AS e2
ON e1.age = e2.age;

------ full join ---------

SELECT * 
FROM emp_1 AS e1
LEFT JOIN emp_2 AS e2
ON e1.id = e2.id

UNION


SELECT * 
FROM emp_1 AS e1
RIGHT JOIN emp_2 AS e2
ON e1.id = e2.id;


-- self join
SELECT 
	e1.salary AS emplyee1,
    e2.salary AS employee2
FROM
	emp_2 AS e1
LEFT JOIN 
	emp_2 AS e2
ON
e1.salary <> e2.salary



