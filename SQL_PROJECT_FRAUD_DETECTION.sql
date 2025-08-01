     --creating database
--CREATE database fraud_db
--GO 
--USE fraud_db
--GO
--creating table
DROP TABLE IF EXISTS Transactions;
CREATE TABLE Transactions (
    Transaction_id INT IDENTITY(1,1) PRIMARY KEY,
    Transation_time INT,
    Num_transaction_today FLOAT,
    Time_since_last_transaction FLOAT,
    Balance_ratio FLOAT,
    amount DECIMAL(10,2),
    ISFRAUD TINYINT
);

-- entering values by random func
DECLARE @i INT=1

WHILE @i<=1000
BEGIN
     INSERT INTO Transactions(
    Transation_time ,
    Num_transaction_today ,
    Time_since_last_transaction ,
    Balance_ratio ,
    amount ,
    ISFRAUD 
) VALUES(
    --FLOOR (rand()*20)+1,-- +1 for getting values b/w 1-20
    FLOOR (rand()*1440),
    FLOOR(rand()*100),
    FLOOR(rand()*300),
    FLOOR(rand()*100),
    ROUND(rand()*2000,2),
    FLOOR(rand()*2)
);
SET @i=@i+1;
END;
SELECT * FROM transactions;
--BASIC OPERATIONS
--1)fraud & unfraud count
SELECT isfraud,COUNT(*) AS fraud_count FROM transactions
GROUP BY isfraud;

--2)average frauded amount
SELECT isfraud ,AVG(amount) AS avg_frauded_amount
FROM transactions
GROUP BY isfraud;

--3)average ratio and time delay
SELECT isfraud,ROUND(AVG(Balance_ratio),2) AS average_ratio,ROUND(AVG(Time_since_last_transaction),2) AS average_delay
FROM transactions
GROUP BY isfraud;

--4)
SELECT isfraud,CASE WHEN amount<500 THEN 'Low'
                    WHEN amount BETWEEN 500 AND 1000 THEN 'Medium'
                    ELSE 'High'
                    END AS amount_range,COUNT(*) AS frequency
FROM transactions
GROUP BY CASE WHEN amount<500 THEN 'Low'
                    WHEN amount BETWEEN 500 AND 1000 THEN 'Medium'
                    ELSE 'High'
                    END ,isfraud;

--5)suspected fraud
SELECT COUNT(*) AS suspected_fraud
FROM transactions
WHERE amount>1500 AND Balance_ratio<20 AND isfraud=1;
