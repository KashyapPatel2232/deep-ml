-- Return name and salary
SELECT
name,
sum(salary)
FROM employees
group by
name
