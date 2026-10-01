import pandas as pd

# Load datasets
orders = pd.read_csv("datasets/olist_orders_dataset.csv")
products = pd.read_csv("datasets/olist_products_dataset.csv")

# Convert order date columns
date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    orders[column] = pd.to_datetime(orders[column])

# Calculate delivery time
orders["delivery_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_purchase_timestamp"]
).dt.days

# Average delivery time
average_delivery = orders["delivery_days"].mean()
print("Average Delivery Days:", average_delivery)

# Product dataset exploration
print("\nProducts Shape:", products.shape)
print("\nProduct Columns:")
print(products.columns)

print("\nProduct Missing Values:")
print(products.isnull().sum())

# Top 10 product categories by number of products
category_counts = (
    products["product_category_name"]
    .value_counts()
    .head(10)
)

print("\nTop 10 Product Categories by Number of Products:")
print(category_counts)

# Revenue by product category

order_items = pd.read_csv("datasets/olist_order_items_dataset.csv")
products = pd.read_csv("datasets/olist_products_dataset.csv")

category_revenue = (
    order_items
    .merge(products[["product_id", "product_category_name"]], on="product_id")
    .groupby("product_category_name")["price"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Product Categories by Revenue:")
print(category_revenue)
# Customer analysis

customers = pd.read_csv("datasets/olist_customers_dataset.csv")

# Unique customers
total_customers = customers["customer_unique_id"].nunique()

print("\nTotal Unique Customers:", total_customers)

# Customers by state
customers_by_state = (
    customers["customer_state"]
    .value_counts()
    .head(10)
)

print("\nTop 10 States by Customer Count:")
print(customers_by_state)
# Customer spending / Average Order Value

payments = pd.read_csv("datasets/olist_order_payments_dataset.csv")
reviews = pd.read_csv("datasets/olist_order_reviews_dataset.csv")

total_revenue = payments["payment_value"].sum()
total_orders = payments["order_id"].nunique()

average_order_value = total_revenue / total_orders

print("\nTotal Revenue:", total_revenue)
print("Total Orders:", total_orders)
print("Average Order Value:", average_order_value)

# Payment method analysis

payment_analysis = (
    payments
    .groupby("payment_type")
    .agg(
        Total_Payments=("payment_type", "count"),
        Total_Revenue=("payment_value", "sum")
    )
    .sort_values("Total_Revenue", ascending=False)
)

print("\nPayment Method Analysis:")
print(payment_analysis)

# Customer Review Analysis

reviews["review_score"] = pd.to_numeric(
    reviews["review_score"],
    errors="coerce"
)

average_review_score = reviews["review_score"].mean()

review_distribution = (
    reviews["review_score"]
    .value_counts()
    .sort_index()
)

low_rating_percentage = (
    reviews["review_score"].isin([1, 2]).mean() * 100
)

print("\nAverage Review Score:", round(average_review_score, 2))

print("\nReview Score Distribution:")
print(review_distribution)

print(
    "\nLow Rating Percentage (1-2):",
    round(low_rating_percentage, 2),
    "%"
)


# Seller Analysis

sellers = pd.read_csv("datasets/olist_sellers_dataset.csv")

seller_analysis = (
    order_items
    .groupby("seller_id")
    .agg(
        Total_Items=("order_item_id", "count"),
        Total_Revenue=("price", "sum")
    )
    .sort_values("Total_Revenue", ascending=False)
    .head(10)
)

print("\nTop 10 Sellers by Revenue:")
print(seller_analysis)

# Monthly Revenue Trend

orders["year_month"] = (
    orders["order_purchase_timestamp"]
    .dt.to_period("M")
    .astype(str)
)

monthly_revenue = (
    orders[["order_id", "year_month"]]
    .merge(
        payments[["order_id", "payment_value"]],
        on="order_id"
    )
    .groupby("year_month")["payment_value"]
    .sum()
)

print("\nMonthly Revenue:")
print(monthly_revenue)


# Delivery Time vs Review Score

delivery_reviews = (
    orders[
        [
            "order_id",
            "order_status",
            "delivery_days"
        ]
    ]
    .merge(
        reviews[["order_id", "review_score"]],
        on="order_id"
    )
)

delivery_reviews = delivery_reviews[
    (delivery_reviews["order_status"] == "delivered")
    & (delivery_reviews["delivery_days"].notna())
]

delivery_vs_review = (
    delivery_reviews
    .groupby("review_score")["delivery_days"]
    .mean()
)

print("\nAverage Delivery Days by Review Score:")
print(delivery_vs_review)


import matplotlib.pyplot as plt

# Monthly Revenue Chart

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_revenue.index,
    monthly_revenue.values,
    marker="o"
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Top 10 Product Categories by Revenue Chart

plt.figure(figsize=(12, 6))

category_revenue.sort_values().plot(kind="barh")

plt.title("Top 10 Product Categories by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product Category")

plt.tight_layout()
plt.show()

# Top 10 Product Categories by Product Count Chart

plt.figure(figsize=(12, 6))

category_counts.sort_values().plot(kind="barh")

plt.title("Top 10 Product Categories by Number of Products")
plt.xlabel("Number of Products")
plt.ylabel("Product Category")

plt.tight_layout()
plt.show()

# Review Score Distribution Chart

plt.figure(figsize=(8, 5))

review_distribution.plot(kind="bar")

plt.title("Review Score Distribution")
plt.xlabel("Review Score")
plt.ylabel("Number of Reviews")

plt.tight_layout()
plt.show()

# Payment Method Revenue Chart

plt.figure(figsize=(8, 5))

payment_analysis["Total_Revenue"].plot(kind="bar")

plt.title("Revenue by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Revenue")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Delivery Time vs Review Score Chart

plt.figure(figsize=(8, 5))

delivery_vs_review.plot(
    kind="line",
    marker="o"
)

plt.title("Average Delivery Days vs Review Score")
plt.xlabel("Review Score")
plt.ylabel("Average Delivery Days")

plt.tight_layout()
plt.show()


# Delivery Time vs Review Score Chart

plt.figure(figsize=(8, 5))

delivery_vs_review.plot(
    kind="line",
    marker="o"
)

plt.title("Average Delivery Days vs Review Score")
plt.xlabel("Review Score")
plt.ylabel("Average Delivery Days")

plt.tight_layout()
plt.show()