
-- Adin Solomon, abs358

-- Q1

select CategoryID, count(ProductID)
from products
group by CategoryID
order by CategoryID asc;

-- Q2

select c.CategoryName, count(p.ProductID)
from categories c, products p
where c.CategoryID = p.CategoryID
group by c.CategoryName
order by c.CategoryName asc;

-- Q3

select count(distinct SupplierID)
from products;

-- Q4

select count(distinct CustomerID)
from customers;

-- Q5

select c.CompanyName, count(o.OrderID) as ordercount
from customers c, orders o
where c.CustomerID = o.CustomerID and extract(year from OrderDate)=1997
group by c.CompanyName
having count(o.OrderID) >= 10
order by count(o.OrderID) desc;

-- Q6

select count(distinct ProductID)
from order_details
where Discount=0.25;

-- Q7

select count(distinct CustomerID)
from customers
where CustomerID not in (
    select CustomerID
    from orders
    where extract(year from OrderDate) = 1996
);

-- Q8

select City, count(distinct CompanyName) CompanyCount
from customers
group by City
having count(distinct CompanyName) >= 2
order by count(distinct CompanyName) desc, City asc;

-- Q9

select cast(sum(UnitPrice * Quantity * (1 - Discount)) as decimal(10,2))
from order_details
where OrderID = 10332;

-- Q10

with CompanyCounts (City, CompanyCount) as (
    select City, count(distinct CompanyName) CompanyCount
    from customers
    group by City
)
select City, CompanyName
from customers
where City = (
    select City from CompanyCounts where CompanyCount = (
        select max(CompanyCount) from CompanyCounts
    )
);
