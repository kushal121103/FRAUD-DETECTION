--Transactions Above Suspicious Amount Threshold
SELECT TransactionID,CustomerID,Amount,TransactionDate
FROM Transactions
WHERE Amount > 5000
ORDER BY Amount DESC;

--Multiple Transactions in a Short Time
SELECT CustomerID,COUNT(*) AS TransactionCount,MIN(TransactionDate) AS FirstTransaction,MAX(TransactionDate) AS LastTransaction
FROM Transactions
WHERE TransactionDate >= DATEADD(MINUTE, -30, GETDATE())
GROUP BY CustomerID
HAVING COUNT(*) > 5;

--Same Device Used by Multiple Customers
SELECT DeviceID,COUNT(DISTINCT CustomerID) AS UniqueCustomers
FROM Transactions
GROUP BY DeviceID
HAVING COUNT(DISTINCT CustomerID) > 3;

--Transactions From Different Countries in Short Period
SELECT CustomerID,COUNT(DISTINCT Location) AS LocationCount,MIN(TransactionDate) AS FirstTransaction,MAX(TransactionDate) AS LastTransaction
FROM Transactions
WHERE TransactionDate >= DATEADD(HOUR, -24, GETDATE())
GROUP BY CustomerID
HAVING COUNT(DISTINCT Location) > 1;

-- Same Amount Repeated Many Times
SELECT CustomerID,Amount,COUNT(*) AS RepeatCount
FROM Transactions
GROUP BY CustomerID, Amount
HAVING COUNT(*) > 3
ORDER BY RepeatCount DESC;

--Transactions at Unusual Hours
SELECT TransactionID,CustomerID,Amount,TransactionDate
FROM Transactions
WHERE DATEPART(HOUR, TransactionDate) BETWEEN 0 AND 5;


--Sudden Spending Spike Compared to Average
SELECT T.CustomerID,T.TransactionID,T.Amount,AVG(T2.Amount) AS AvgSpending
FROM Transactions T
JOIN Transactions T2 
    ON T.CustomerID = T2.CustomerID
GROUP BY T.CustomerID, T.TransactionID, T.Amount
HAVING T.Amount > 3 * AVG(T2.Amount);

--Transactions Marked as Fraud in History
SELECT TransactionID,CustomerID,Amount,Location,TransactionDate
FROM Transactions
WHERE IsFraud = 1
ORDER BY TransactionDate DESC;

--Customers With High Fraud Ratio
SELECT CustomerID,COUNT(*) AS TotalTransactions,SUM(CASE WHEN IsFraud = 1 THEN 1 ELSE 0 END) AS FraudCount,
    ROUND(
        100.0 * SUM(CASE WHEN IsFraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 2
    ) AS FraudPercentage
FROM Transactions
GROUP BY CustomerID
HAVING COUNT(*) > 5
ORDER BY FraudPercentage DESC;

--Suspicious Merchants With Many Fraud Cases
SELECT Merchant,COUNT(*) AS TotalTransactions,SUM(CASE WHEN IsFraud = 1 THEN 1 ELSE 0 END) AS FraudCount
FROM Transactions
GROUP BY Merchant
HAVING SUM(CASE WHEN IsFraud = 1 THEN 1 ELSE 0 END) > 5
ORDER BY FraudCount DESC;
