
-- Adin Solomon, abs358

-- Q1

select CategoryNames
from categories;

-- Q2

select CustomerName 
from customers
where   Country='Canada'
order by CustomerName desc;

-- Q3

select CustomerName, Address, City, Country
from customers
where   Country<>'Brazil'
    and Country<>'France'
    and Country<>'Germany'
    and Country<>'USA'
order by Country asc, City asc;

-- Q4

select OrderDate, ShippedDate, CustomerID, Freight
from orders
where OrderDate='1997-07-25';

-- Q5

select ProductName, UnitPrice
from products
where ProductName like 'C%'
order by UnitPrice desc;

-- Q6

select ProductName, UnitPrice, QuantityPerUnit
from products
where QuantityPerUnit like '%bottle%'
order by ProductName;

--Q7

select p.ProductName, c.CategoryName
from products p, categories c
where   p.CategoryID=c.CategoryID
    and c.CategoryName='Produce'
order by p.ProductName;

-- Q8

select o.OrderID, c.CompanyName, c.CustomerName, o.OrderDate, o.ShippedDate, o.RequiredDate
from orders o, customers c
where   o.OrderDate>='1998-01-01'
    and o.ShippedDate>o.RequiredDate
order by o.OrderID;

-- Q9

select OrderID, Freight, cast(Freight * 1.1 as decimal(10,2)) as freight_with_tax
from orders
where Freight>=500;

-- Q10

select od.OrderID, p.ProductName, od.Quantity, p.UnitPrice, od.Discount, 
    cast((od.Quantity * p.UnitPrice) * (1 - od.Discount) as decimal(10,2)) as product_total_price
from order_details od, products p
where   (od.OrderID=10248 or od.OrderID=10866)
    and od.ProductID=p.ProductID
order by od.OrderID;
