SELECT 
    ad_id,
    SUM(clicks) AS clicks,
    SUM(impressions) AS impressions,
    ROUND(100.0 * SUM(clicks) / NULLIF(SUM(impressions), 0), 2) AS ctr
FROM 
    ad_events
WHERE 
    EXTRACT(YEAR FROM event_date) = 2023
GROUP BY 
    ad_id
ORDER BY 
    ad_id ASC;