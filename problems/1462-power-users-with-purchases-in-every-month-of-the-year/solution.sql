WITH qualifying_months AS (
    SELECT 
        user_id, 
        DATE_TRUNC('month', purchase_date) AS purchase_month
    FROM purchases
    WHERE purchase_date >= '2024-01-01' AND purchase_date < '2025-01-01'
    GROUP BY 
        user_id, 
        DATE_TRUNC('month', purchase_date)
    HAVING COUNT(purchase_id) >= 2
)
SELECT user_id
FROM qualifying_months
GROUP BY user_id
HAVING COUNT(*) = 12
ORDER BY user_id;