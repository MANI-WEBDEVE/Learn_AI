import csv
import random

# Define column names
columns = ["id", "name", "age", "gender", "city", "country", "salary"]

# Sample data for meaningful columns
names = ["John Doe", "Jane Smith", "Emily Davis", "Michael Brown", "David Wilson", "Sarah Johnson", "Zara Khan"]
cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "London", "Paris", "Karachi", "Dubai", "Tokyo"]
countries = ["USA", "UK", "Canada", "Australia", "Germany", "France", "Pakistan", "UAE", "Japan"]

# Generate 100 rows of data
data = []
for i in range(1, 101):
    row = {
        "id": i,
        "name": random.choice(names),
        "age": random.randint(20, 60),
        "gender": random.choice(["Male", "Female"]),
        "city": random.choice(cities),
        "country": random.choice(countries),
        "salary": random.randint(3000, 10000)
    }
    data.append(row)

# Write data to CSV file
with open("sample_data.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(data)

print("CSV file generated successfully!")