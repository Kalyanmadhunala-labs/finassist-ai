from sql_tool import run_sql


result = run_sql(
    """
    SELECT SUM(revenue)
    FROM sales_transactions
    WHERE region = %s
    """,
    ("South",)
)


print("South region revenue:", result[0][0])