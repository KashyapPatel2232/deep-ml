-- your query
SELECT sum(CASE when department = 'Engineering' then salary else 0 end) as engineering_total,
sum(CASE when department = 'Sales' then salary else 0 end) as sales_total,
(engineering_total - sales_total) as salary_difference
from employees
