SELECT 
  Product_Category,
  SUM(Revenue) AS Total_Revenue
FROM 
  dbo.raw_sales_data
GROUP BY 
  Product_Category
ORDER BY 
  Total_Revenue DESC;

  SELECT 
  Product_Category,
  AVG(Discount) AS Average_Discount
FROM 
  dbo.raw_sales_data
GROUP BY 
  Product_Category
ORDER BY 
  Average_Discount DESC;

SELECT 
  FORMAT(CAST(Order_Date AS DATE), 'yyyy-MM') AS Sales_Month,
  SUM(Revenue) AS Monthly_Revenue
FROM 
  dbo.raw_sales_data
GROUP BY 
  FORMAT(CAST(Order_Date AS DATE), 'yyyy-MM')
ORDER BY 
  Sales_Month;


  SELECT 
    Customer_Name,
    COUNT(*) AS Order_Count,
    SUM(Revenue) AS Total_Revenue
FROM 
    dbo.raw_sales_data
GROUP BY 
    Customer_Name
ORDER BY 
    Total_Revenue DESC;

	SELECT 
    Product_Category,
    COUNT(*) AS Order_Count
FROM 
    dbo.raw_sales_data
GROUP BY 
    Product_Category
ORDER BY 
    Order_Count DESC;

	SELECT 
    FORMAT(CAST(Order_Date AS DATE), 'yyyy-MM') AS Sales_Month,
    COUNT(*) AS Order_Count
FROM 
    dbo.raw_sales_data
GROUP BY 
    FORMAT(CAST(Order_Date AS DATE), 'yyyy-MM')
ORDER BY 
    Sales_Month;

	SELECT 
    FORMAT(CAST(Order_Date AS DATE), 'yyyy-MM') AS Sales_Month,
    SUM(Revenue) AS Monthly_Revenue,
    AVG(Discount) AS Avg_Discount
FROM 
    dbo.raw_sales_data
GROUP BY 
    FORMAT(CAST(Order_Date AS DATE), 'yyyy-MM')
ORDER BY 
    Sales_Month;





