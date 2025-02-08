CREATE DATABASE datascience;
create database data2;

drop database data2;

USE  datascience;

CREATE TABLE course (
    topics VARCHAR(50),
    duration INT,
    _id INT PRIMARY KEY,
    fees INT NOT NULL
);


  INSERT INTO course VALUES("Machine learning", 5, 01, 9000);
INSERT INTO course VALUES("Deep learning", 10, 02, 12000);

SELECT * FROM course