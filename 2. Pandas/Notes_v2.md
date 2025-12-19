# Pandas Complete Guide for Data Engineers & Analysts

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Installation & Setup](#installation--setup)
4. [Data Structures](#data-structures)
5. [Reading & Writing Data](#reading--writing-data)
6. [Data Selection & Indexing](#data-selection--indexing)
7. [Data Cleaning](#data-cleaning)
8. [Data Manipulation](#data-manipulation)
9. [Aggregation & Grouping](#aggregation--grouping)
10. [Merging & Joining](#merging--joining)
11. [Time Series](#time-series)
12. [String Operations](#string-operations)
13. [Visualization](#visualization)
14. [Advanced Operations](#advanced-operations)
15. [Performance Tips](#performance-tips)
16. [Quick Reference](#quick-reference)

---

## Introduction

**Pandas** is a powerful and popular Python library designed for:
- **Data Manipulation**: Cleaning, transforming, and structuring data
- **Data Analysis**: Finding patterns, trends, and insights

### Why Pandas?

- Simplifies working with structured data
- Handles missing values efficiently
- Built on top of NumPy for fast computations
- Used in Finance, Retail, Healthcare (Real-time applications)

### History

- Created by **Wes McKinney** in **2008**
- Originally developed to handle large financial datasets
- Excel was inefficient for large-scale data operations

### Key Differentiators

**Data Manipulation**: Changing data, organizing/preparing to make it useful

**Data Analysis**: Extracting patterns, trends, and insights from the data

### Applications

1. **Data Scientist**: Building ML models, statistical analysis
2. **Data Analyst**: Business intelligence, reporting
3. **ML/AI/Business Analyst**: Feature engineering, predictive modeling
4. **Research Scientist**: Academic research, scientific computing

---

## Core Concepts

### Series
A Series is a **1-dimensional labeled array** that can hold any data type.

### DataFrame
A DataFrame is like a **table with multiple columns**, a **2-dimensional labeled dataset** with rows and columns.
- **Axis 0**: Rows
- **Axis 1**: Columns

---

## Installation & Setup

```bash
# Install pandas
pip install pandas

# Install with additional dependencies
pip install pandas openpyxl xlrd  # For Excel support
pip install pandas sqlalchemy     # For database support
```

### **Import Pandas**

```python
import pandas as pd
import numpy as np
```

- **pd.\_\_version\_\_**: Check Pandas version

**Example:**
```python
import pandas as pd
print(pd.__version__)
```

**Output:**
```
2.1.0
```

---

## Data Structures

### **1. Creating Series**

- **pd.Series(data, index)**: Create a 1D labeled array with optional custom index
- **pd.Series(dict)**: Create Series from dictionary

**Example:**
```python
import pandas as pd

# Create Series from list
s = pd.Series([1, 2, 3, 4, 5])
print(s)
```

**Output:**
```
0    1
1    2
2    3
3    4
4    5
dtype: int64
```

```python
# Series with custom index
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
```

**Output:**
```
a    10
b    20
c    30
dtype: int64
```

```python
# Series from dictionary
data = {'a': 10, 'b': 20, 'c': 30}
s = pd.Series(data)
print(s)
```

**Output:**
```
a    10
b    20
c    30
dtype: int64
```

### **2. Creating DataFrames**

- **pd.DataFrame(data, index, columns)**: Create 2D table from lists, dicts, Series, or arrays
- **pd.DataFrame(dict)**: Create DataFrame from dictionary of lists
- **pd.DataFrame(list_of_dicts)**: Create DataFrame from list of dictionaries
- **pd.DataFrame(np.array)**: Create DataFrame from NumPy array

**Example:**
```python
# Create DataFrame from dictionary
data = {
    "name": ["Ank", "Harsh", "Ankit"], 
    "age": [18, 22, 28],
    "city": ["Delhi", "Mumbai", "Bangalore"]
}
df = pd.DataFrame(data)
print(df)
```

**Output:**
```
    name  age       city
0    Ank   18      Delhi
1  Harsh   22     Mumbai
2  Ankit   28  Bangalore
```

```python
# DataFrame from list of dictionaries
data = [
    {"name": "Ank", "age": 18},
    {"name": "Harsh", "age": 22},
    {"name": "Ankit", "age": 28}
]
df = pd.DataFrame(data)
print(df)
```

**Output:**
```
    name  age
0    Ank   18
1  Harsh   22
2  Ankit   28
```

```python
# DataFrame from NumPy array
import numpy as np
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print(df)
```

**Output:**
```
   A  B  C
0  1  2  3
1  4  5  6
2  7  8  9
```

---

## Reading & Writing Data

### **Reading Data**

- **pd.read_csv('file.csv')**: Read CSV file into DataFrame
- **pd.read_csv('file.csv', encoding='utf-8')**: Read CSV with specific encoding
- **pd.read_csv('file.csv', usecols=['col1', 'col2'])**: Read specific columns only
- **pd.read_csv('file.csv', skiprows=[0, 2])**: Skip specific rows while reading
- **pd.read_csv('file.txt', delimiter='\t')**: Read file with custom delimiter
- **pd.read_csv('file.csv', chunksize=10000)**: Read large file in chunks
- **pd.read_excel('file.xlsx')**: Read Excel file
- **pd.read_excel('file.xlsx', sheet_name='Sheet1')**: Read specific Excel sheet
- **pd.read_json('file.json')**: Read JSON file
- **pd.read_html('url')**: Read HTML tables from URL (returns list of DataFrames)
- **pd.read_sql(query, connection)**: Read data from SQL database
- **pd.read_parquet('file.parquet')**: Read Parquet file (columnar format for big data)
- **pd.read_clipboard()**: Read data from clipboard

**Example:**
```python
import pandas as pd

# Read CSV
df = pd.read_csv("data.csv")

# Read CSV with specific encoding (for special characters)
df = pd.read_csv("data.csv", encoding="utf-8")
df = pd.read_csv("data.csv", encoding="latin1")

# Read specific columns
df = pd.read_csv("data.csv", usecols=['name', 'age'])

# Skip rows
df = pd.read_csv("data.csv", skiprows=[0, 2])

# Read with custom delimiter
df = pd.read_csv("data.txt", delimiter='\t')  # Tab-separated

# Read Excel
df = pd.read_excel("data.xlsx")
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# Read JSON
df = pd.read_json("data.json")

# Read HTML tables
df = pd.read_html("https://example.com/table.html")[0]  # First table

# Read from SQL database
import sqlalchemy
engine = sqlalchemy.create_engine('sqlite:///database.db')
df = pd.read_sql("SELECT * FROM table_name", engine)

# Read Parquet (Big Data format)
df = pd.read_parquet("data.parquet")
```

### **Writing Data**

- **df.to_csv('file.csv', index=False)**: Export to CSV without index
- **df.to_csv('file.csv', encoding='utf-8')**: Export with specific encoding
- **df.to_excel('file.xlsx', index=False)**: Export to Excel
- **df.to_excel('file.xlsx', sheet_name='MySheet')**: Export to Excel with sheet name
- **df.to_json('file.json', orient='records')**: Export to JSON
- **df.to_sql('table_name', connection, if_exists='replace')**: Export to SQL database
- **df.to_parquet('file.parquet', compression='gzip')**: Export to compressed Parquet
- **df.to_html('file.html')**: Export to HTML table
- **df.to_clipboard()**: Copy DataFrame to clipboard

**Example:**
```python
# Write to CSV without index
df.to_csv("output.csv", index=False)

# Write to CSV with specific encoding
df.to_csv("output.csv", encoding="utf-8", index=False)

# Write to Excel
df.to_excel("output.xlsx", index=False)
df.to_excel("output.xlsx", sheet_name="MySheet", index=False)

# Write to JSON
df.to_json("output.json", orient='records', indent=4)

# Write to SQL database
df.to_sql("table_name", engine, if_exists='replace', index=False)

# Write to Parquet (compressed, efficient for big data)
df.to_parquet("output.parquet", compression='gzip')

# Write to HTML
df.to_html("output.html", index=False)
```

**Pro Tip for Big Data Engineers:**
- Use **Parquet** format for large datasets (compressed, columnar storage)
- Use **chunking** for large CSV files:

```python
# Read large CSV in chunks
chunk_size = 10000
chunks = []
for chunk in pd.read_csv("large_file.csv", chunksize=chunk_size):
    # Process each chunk
    processed = chunk[chunk['age'] > 18]
    chunks.append(processed)

df = pd.concat(chunks, ignore_index=True)
```

---

## Data Selection & Indexing

### **Basic Inspection**

- **df.head(n)**: View first n rows (default 5)
- **df.tail(n)**: View last n rows (default 5)
- **df.shape**: Get dimensions (rows, columns)
- **df.columns**: Get column names as Index object
- **df.index**: Get row index
- **df.dtypes**: Get data types of each column
- **df.info()**: Get summary including memory usage and data types
- **df.describe()**: Get statistical summary for numeric columns
- **df.memory_usage(deep=True)**: Get detailed memory usage

**Example:**
```python
data = {
    "name": ["Ank", "Harsh", "Ankit", "Priya"], 
    "age": [18, 22, 28, 24],
    "salary": [30000, 50000, 60000, 45000]
}
df = pd.DataFrame(data)

# Top n rows (default 5)
print(df.head(2))

# Last n rows (default 5)
print(df.tail(2))

# Shape (rows, columns)
print(df.shape)  # (4, 3)

# Column names
print(df.columns)  # Index(['name', 'age', 'salary'], dtype='object')

# Index
print(df.index)  # RangeIndex(start=0, stop=4, step=1)

# Data types
print(df.dtypes)

# Info (memory usage, data types)
print(df.info())

# Statistical summary
print(df.describe())
```

**Output for `df.head(2)`:**
```
    name  age  salary
0    Ank   18   30000
1  Harsh   22   50000
```

**Output for `df.describe()`:**
```
             age        salary
count   4.000000      4.000000
mean   23.000000  46250.000000
std     4.320494  13149.778199
min    18.000000  30000.000000
25%    20.500000  41250.000000
50%    23.000000  47500.000000
75%    25.500000  52500.000000
max    28.000000  60000.000000
```

### **Selecting Columns**

- **df['column']**: Select single column (returns Series)
- **df[['col1', 'col2']]**: Select multiple columns (returns DataFrame)
- **df.column**: Select column using dot notation (only if no spaces in name)

**Example:**
```python
# Select single column (returns Series)
print(df["name"])

# Select single column (returns DataFrame)
print(df[["name"]])

# Select multiple columns
print(df[["name", "age"]])
```

**Output:**
```
0      Ank
1    Harsh
2    Ankit
3    Priya
Name: name, dtype: object
```

### **Selecting Rows**

- **df.iloc[row_index]**: Select row by integer position
- **df.iloc[start:end]**: Select rows by position range
- **df.iloc[row, col]**: Select specific cell by position
- **df.loc[row_label]**: Select row by label/index
- **df.loc[row_label, 'column']**: Select specific cell by label
- **df.at[row, 'column']**: Fast access to single scalar value by label
- **df.iat[row, col]**: Fast access to single scalar value by position

**Example:**
```python
# Select row by index position (iloc)
print(df.iloc[0])  # First row

# Select multiple rows by position
print(df.iloc[0:2])  # First two rows

# Select specific row and column by position
print(df.iloc[0, 1])  # First row, second column (age of Ank = 18)

# Select rows by label (loc)
df_indexed = df.set_index('name')
print(df_indexed.loc['Ank'])  # Row where name is 'Ank'

# Select specific cell
print(df.at[0, 'age'])  # 18 (faster than loc for single value)
print(df.iat[0, 1])     # 18 (faster than iloc for single value)
```

### **Boolean Indexing (Filtering)**

- **df[df['column'] > value]**: Filter rows based on condition
- **df[(condition1) & (condition2)]**: Filter with AND condition
- **df[(condition1) | (condition2)]**: Filter with OR condition
- **df[~condition]**: Filter with NOT condition
- **df[df['column'].isin([values])]**: Filter rows where column value is in list
- **df[df['column'].between(low, high)]**: Filter rows where column is between values
- **df.query('expression')**: Filter using SQL-like string expression

**Example:**
```python
# Return boolean mask
print(df["age"] < 25)
```

**Output:**
```
0     True
1     True
2    False
3     True
Name: age, dtype: bool
```

```python
# Filter rows where condition is True
print(df[df["age"] < 25])
```

**Output:**
```
    name  age  salary
0    Ank   18   30000
1  Harsh   22   50000
3  Priya   24   45000
```

```python
# Multiple conditions (AND)
print(df[(df["age"] < 25) & (df["salary"] > 40000)])

# Multiple conditions (OR)
print(df[(df["age"] < 20) | (df["salary"] > 55000)])

# NOT condition
print(df[~(df["age"] < 25)])  # Age NOT less than 25

# Using .isin() for multiple values
print(df[df["name"].isin(["Ank", "Priya"])])

# Using .between()
print(df[df["age"].between(20, 25)])

# Query method (SQL-like syntax)
print(df.query("age > 22"))
print(df.query("age > 20 and salary < 55000"))

# Using variables in query
min_age = 20
print(df.query("age > @min_age"))
```

### **Slicing**

- **df[start:end]**: Slice rows by position
- **df[::step]**: Slice with step
- **df.loc[:, 'col1':'col2']**: Slice columns by label

**Example:**
```python
# Slice rows
print(df[0:2])  # First two rows

# Slice with step
print(df[::2])  # Every other row

# Slice columns
print(df.loc[:, 'name':'age'])  # From 'name' to 'age' columns
```

---

## Data Cleaning

### **Handling Missing Values**

- **df.isnull()**: Return boolean DataFrame showing null values
- **df.notnull()**: Return boolean DataFrame showing non-null values
- **df.isna()**: Alias for isnull()
- **df.notna()**: Alias for notnull()
- **df.isnull().sum()**: Count null values per column
- **df.isnull().any()**: Check if any null exists per column
- **df.isnull().all()**: Check if all values are null per column

**Example:**
```python
# Create DataFrame with missing values
data = {
    "name": ["Ank", "Harsh", None, "Priya"],
    "age": [18, None, 28, 24],
    "salary": [30000, 50000, None, 45000]
}
df = pd.DataFrame(data)

# Check for null values
print(df.isnull())  # Returns boolean DataFrame
print(df.notnull())  # Opposite of isnull()

# Count null values per column
print(df.isnull().sum())

# Check if any null exists
print(df.isnull().any())

# Check if all values are null
print(df.isnull().all())
```

**Output for `df.isnull()`:**
```
    name    age  salary
0  False  False   False
1  False   True   False
2   True  False    True
3  False  False   False
```

**Output for `df.isnull().sum()`:**
```
name      1
age       1
salary    1
dtype: int64
```

### **Dropping Missing Values**

- **df.dropna()**: Drop rows with any null value
- **df.dropna(how='all')**: Drop rows where all values are null
- **df.dropna(subset=['col'])**: Drop rows with null in specific columns
- **df.dropna(axis=1)**: Drop columns with any null value
- **df.dropna(thresh=n)**: Keep rows with at least n non-null values
- **df.dropna(inplace=True)**: Modify DataFrame in place

**Example:**
```python
# Drop rows with any null value
df_clean = df.dropna()
print(df_clean)

# Drop rows where all values are null
df_clean = df.dropna(how='all')

# Drop rows with null in specific columns
df_clean = df.dropna(subset=['age'])

# Drop columns with any null value
df_clean = df.dropna(axis=1)

# Drop if at least N non-null values aren't present
df_clean = df.dropna(thresh=2)  # Keep rows with at least 2 non-null values

# Modify in place
df.dropna(inplace=True)
```

### **Filling Missing Values**

- **df.fillna(value)**: Fill all null values with a constant
- **df.fillna({'col1': val1, 'col2': val2})**: Fill different columns with different values
- **df.fillna(method='ffill')**: Forward fill (use previous value)
- **df.fillna(method='bfill')**: Backward fill (use next value)
- **df['column'].fillna(df['column'].mean())**: Fill with column mean
- **df['column'].fillna(df['column'].median())**: Fill with column median
- **df['column'].fillna(df['column'].mode()[0])**: Fill with column mode
- **df.fillna(inplace=True)**: Modify DataFrame in place

**Example:**
```python
# Fill with constant value
df_filled = df.fillna(0)

# Fill with column mean
df['age'].fillna(df['age'].mean(), inplace=True)

# Fill with column median
df['salary'].fillna(df['salary'].median(), inplace=True)

# Fill with mode (most frequent value)
df['name'].fillna(df['name'].mode()[0], inplace=True)

# Forward fill (use previous value)
df_filled = df.fillna(method='ffill')

# Backward fill (use next value)
df_filled = df.fillna(method='bfill')

# Fill different columns with different values
df.fillna({'age': df['age'].mean(), 'salary': 0}, inplace=True)
```

### **Interpolation**

- **df.interpolate()**: Interpolate missing values using default method (linear)
- **df.interpolate(method='linear')**: Linear interpolation
- **df.interpolate(method='polynomial', order=2)**: Polynomial interpolation
- **df.interpolate(method='time')**: Time-based interpolation for time series
- **df.interpolate(axis=0)**: Interpolate along rows
- **df.interpolate(inplace=True)**: Modify DataFrame in place

**Example:**
```python
# Create data with gaps
df = pd.DataFrame({
    'value': [1, None, None, 4, None, 6]
})

# Linear interpolation
df['value'].interpolate(method='linear', inplace=True)
print(df)
```

**Output:**
```
   value
0    1.0
1    2.0
2    3.0
3    4.0
4    5.0
5    6.0
```

```python
# Polynomial interpolation
df['value'].interpolate(method='polynomial', order=2, inplace=True)

# Time-based interpolation (for time series)
df['value'].interpolate(method='time', inplace=True)
```

### **Removing Duplicates**

- **df.duplicated()**: Return boolean Series indicating duplicate rows
- **df.drop_duplicates()**: Remove duplicate rows
- **df.drop_duplicates(subset=['col'])**: Drop duplicates based on specific columns
- **df.drop_duplicates(keep='first')**: Keep first occurrence (default)
- **df.drop_duplicates(keep='last')**: Keep last occurrence
- **df.drop_duplicates(keep=False)**: Remove all duplicates
- **df.drop_duplicates(inplace=True)**: Modify DataFrame in place

**Example:**
```python
# Create DataFrame with duplicates
data = {
    "name": ["Ank", "Harsh", "Ankit", "Ank"],
    "age": [18, 22, 28, 18]
}
df = pd.DataFrame(data)

# Check for duplicates
print(df.duplicated())  # Returns boolean Series

# Drop duplicates (all columns must match)
df_unique = df.drop_duplicates()
print(df_unique)

# Drop duplicates based on specific columns
df_unique = df.drop_duplicates(subset=['name'])

# Keep last occurrence instead of first
df_unique = df.drop_duplicates(keep='last')

# Keep all duplicates marked
df_unique = df.drop_duplicates(keep=False)

# Modify in place
df.drop_duplicates(inplace=True)
```

### **Renaming**

- **df.rename(columns={'old': 'new'})**: Rename specific columns
- **df.rename(index={0: 'first'})**: Rename specific rows
- **df.rename(columns=str.upper)**: Apply function to column names
- **df.rename(inplace=True)**: Modify DataFrame in place
- **df.columns = ['new1', 'new2']**: Rename all columns at once

**Example:**
```python
# Rename specific columns
df.rename(columns={"name": "Name", "age": "Age"}, inplace=True)
print(df)

# Rename all columns
df.columns = ['Name', 'Age', 'Salary']

# Rename using function
df.columns = df.columns.str.upper()  # Convert to uppercase

# Rename index
df.rename(index={0: 'first', 1: 'second'}, inplace=True)
```

### **Replacing Values**

- **df.replace(old_value, new_value)**: Replace specific value throughout DataFrame
- **df.replace([val1, val2], [new1, new2])**: Replace multiple values
- **df.replace({'col': {'old': 'new'}})**: Replace in specific column
- **df.replace(regex_pattern, value, regex=True)**: Replace using regex
- **df['column'].replace(old, new, inplace=True)**: Replace in single column

**Example:**
```python
# Replace specific value
df.replace({"Ank": "Ankur"}, inplace=True)

# Replace in specific column
df["name"].replace("Harsh", "Harshit", inplace=True)

# Replace multiple values
df.replace({"Ank": "Ankur", "Harsh": "Harshit"}, inplace=True)

# Replace with dictionary for different columns
df.replace({"name": {"Ank": "Ankur"}, "age": {18: 19}}, inplace=True)

# Replace with regex
df["name"].replace(r'^A.*', 'Anonymous', regex=True, inplace=True)
```

### **Changing Data Types**

- **df['column'].astype(dtype)**: Convert column to specified data type
- **df.astype({'col1': int, 'col2': float})**: Convert multiple columns
- **df['column'].astype('category')**: Convert to categorical type
- **pd.to_numeric(df['column'], errors='coerce')**: Convert to numeric, invalid values become NaN
- **pd.to_datetime(df['column'])**: Convert to datetime
- **pd.to_timedelta(df['column'])**: Convert to timedelta

**Example:**
```python
# Convert column to different type
df["age"] = df["age"].astype(float)
df["salary"] = df["salary"].astype(int)
df["name"] = df["name"].astype(str)

# Convert to categorical (saves memory for repeated values)
df["city"] = df["city"].astype('category')

# Convert to datetime
df["date"] = pd.to_datetime(df["date"])

# Handle errors during conversion
df["age"] = pd.to_numeric(df["age"], errors='coerce')  # Invalid values become NaN
```

### **Dropping Columns/Rows**

- **df.drop(columns=['col1', 'col2'])**: Drop specific columns
- **df.drop([0, 2])**: Drop rows by index
- **df.drop(index=[0, 2])**: Drop rows by index (explicit)
- **df.drop('column', axis=1)**: Drop column (axis=1 for columns)
- **df.drop(inplace=True)**: Modify DataFrame in place

**Example:**
```python
# Drop columns
df_dropped = df.drop(columns=["age"])
df.drop(columns=["age", "salary"], inplace=True)

# Drop rows by index
df_dropped = df.drop([0, 2])  # Drop rows at index 0 and 2

# Drop rows by condition
df = df[df["age"] >= 18]  # Keep only rows where age >= 18
```

---

## Data Manipulation

### **Adding Columns**

- **df['new_col'] = value**: Add new column with constant or calculated value
- **df['new_col'] = df['col1'] + df['col2']**: Add column with calculation
- **df.insert(loc, 'column', value)**: Insert column at specific position
- **df['new_col'] = df.apply(func, axis=1)**: Add column based on function applied to rows
- **df.assign(new_col=value)**: Add column and return new DataFrame (doesn't modify original)

**Example:**
```python
data = {
    "name": ["Ank", "Harsh", "Ankit"],
    "age": [18, 22, 28],
    "salary": [30000, 50000, 60000]
}
df = pd.DataFrame(data)

# Add new column with calculation
df["Bonus"] = df["salary"] * 0.1
print(df)

# Add column based on condition
df["Category"] = df["age"].apply(lambda x: "Junior" if x < 25 else "Senior")

# Add column at specific position
df.insert(0, "Employee_ID", [100, 101, 102])
print(df)
```

**Output:**
```
   Employee_ID   name  age  salary   Bonus Category
0          100    Ank   18   30000  3000.0   Junior
1          101  Harsh   22   50000  5000.0   Junior
2          102  Ankit   28   60000  6000.0   Senior
```

### **Modifying Values**

- **df.at[row, 'col'] = value**: Set value at specific location (fast for scalars)
- **df.iat[row, col] = value**: Set value by position (fast for scalars)
- **df.loc[condition, 'col'] = value**: Update column based on condition
- **df['col'] = df['col'].apply(func)**: Apply function to column
- **df[['col1', 'col2']] = df[['col1', 'col2']].apply(func)**: Apply to multiple columns
- **df.apply(func, axis=1)**: Apply function row-wise

**Example:**
```python
# Set value at specific location
df.at[0, 'age'] = 19

# Set value using iloc
df.iat[0, 2] = 20

# Update column based on condition
df.loc[df['name'] == "Ank", "age"] = 25

# Apply function to column
df['salary'] = df['salary'].apply(lambda x: x * 1.1)  # 10% raise

# Apply function to multiple columns
df[['age', 'salary']] = df[['age', 'salary']].apply(lambda x: x * 2)

# Apply function to entire DataFrame
def categorize(row):
    if row['age'] < 25:
        return 'Junior'
    else:
        return 'Senior'

df['Category'] = df.apply(categorize, axis=1)
```

### **Sorting**

- **df.sort_values(by='col')**: Sort by single column
- **df.sort_values(by='col', ascending=False)**: Sort in descending order
- **df.sort_values(by=['col1', 'col2'])**: Sort by multiple columns
- **df.sort_values(by='col', inplace=True)**: Sort in place
- **df.sort_index()**: Sort by index
- **df.nlargest(n, 'col')**: Get n largest values
- **df.nsmallest(n, 'col')**: Get n smallest values

**Example:**
```python
# Sort by single column
df_sorted = df.sort_values(by='age')
df_sorted = df.sort_values(by='age', ascending=False)

# Sort by multiple columns
df_sorted = df.sort_values(by=['age', 'salary'], ascending=[True, False])

# Sort index
df_sorted = df.sort_index()

# Sort in place
df.sort_values(by='age', inplace=True)

# Get largest/smallest
top_3 = df.nlargest(3, 'salary')
bottom_3 = df.nsmallest(3, 'age')
```

### **Creating Calculated Columns**

- **df['col3'] = df['col1'] + df['col2']**: Mathematical operations
- **df['col'] = np.where(condition, true_val, false_val)**: Conditional column with single condition
- **df['col'] = np.select(conditions, choices)**: Conditional column with multiple conditions
- **df['col'] = df['str_col1'] + ' ' + df['str_col2']**: String concatenation

**Example:**
```python
# Mathematical operations
df['tax'] = df['salary'] * 0.2
df['net_salary'] = df['salary'] - df['tax']

# String concatenation
df['full_info'] = df['name'] + ' - ' + df['age'].astype(str) + ' years'

# Conditional column with np.where
import numpy as np
df['status'] = np.where(df['salary'] > 40000, 'High', 'Low')

# Multiple conditions with np.select
conditions = [
    df['salary'] < 35000,
    (df['salary'] >= 35000) & (df['salary'] < 55000),
    df['salary'] >= 55000
]
choices = ['Low', 'Medium', 'High']
df['salary_band'] = np.select(conditions, choices, default='Unknown')
```

### **Binning (Creating Categories)**

- **pd.cut(series, bins, labels)**: Cut continuous values into discrete bins
- **pd.qcut(series, q, labels)**: Cut into equal-sized bins (quantiles)

**Example:**
```python
# Create age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 20, 30, 100], 
                          labels=['0-20', '21-30', '31+'])

# Create equal-sized bins (quartiles)
df['salary_quartile'] = pd.qcut(df['salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

---

## Aggregation & Grouping

### **Basic Aggregations**

- **df['col'].sum()**: Sum of column values
- **df['col'].mean()**: Mean (average) of column
- **df['col'].median()**: Median of column
- **df['col'].mode()**: Mode (most frequent value)
- **df['col'].min()**: Minimum value
- **df['col'].max()**: Maximum value
- **df['col'].std()**: Standard deviation
- **df['col'].var()**: Variance
- **df['col'].count()**: Count of non-null values
- **df['col'].nunique()**: Count of unique values
- **df['col'].quantile(q)**: Quantile (e.g., 0.25 for 25th percentile)
- **df['col'].cumsum()**: Cumulative sum
- **df['col'].cumprod()**: Cumulative product
- **df['col'].diff()**: Difference between consecutive values
- **df['col'].rank()**: Rank values

**Example:**
```python
data = {
    "department": ["IT", "HR", "IT", "HR", "IT"],
    "name": ["Ank", "Harsh", "Ankit", "Priya", "Rahul"],
    "salary": [30000, 50000, 60000, 45000, 55000]
}
df = pd.DataFrame(data)

# Basic statistics
print(df['salary'].sum())      # 240000
print(df['salary'].mean())     # 48000.0
print(df['salary'].median())   # 50000.0
print(df['salary'].min())      # 30000
print(df['salary'].max())      # 60000
print(df['salary'].std())      # Standard deviation
print(df['salary'].var())      # Variance
print(df['salary'].count())    # 5
```

### **GroupBy Operations**

- **df.groupby('col')**: Group DataFrame by column
- **df.groupby('col')['target'].agg(func)**: Aggregate grouped data
- **df.groupby('col').agg(['func1', 'func2'])**: Multiple aggregations
- **df.groupby('col').agg({'col1': 'sum', 'col2': 'mean'})**: Different aggregations for different columns
- **df.groupby(['col1', 'col2'])**: Group by multiple columns
- **df.groupby('col').transform(func)**: Transform groups and return same shape as input
- **df.groupby('col').filter(func)**: Filter groups based on function
- **df.groupby('col').apply(func)**: Apply custom function to each group

**Example:**
```python
# Group by single column
grouped = df.groupby('department')['salary'].sum()
print(grouped)
```

**Output:**
```
department
HR     95000
IT    145000
Name: salary, dtype: int64
```

```python
# Multiple aggregations
grouped = df.groupby('department')['salary'].agg(['sum', 'mean', 'count'])
print(grouped)
```

**Output:**
```
                sum     mean  count
department                        
HR           95000  47500.0      2
IT          145000  48333.3      3
```

```python
# Group by multiple columns
grouped = df.groupby(['department', 'name'])['salary'].sum()

# Different aggregations for different columns
grouped = df.groupby('department').agg({
    'salary': ['sum', 'mean'],
    'name': 'count'
})
print(grouped)

# Custom aggregation function
def salary_range(x):
    return x.max() - x.min()

grouped = df.groupby('department')['salary'].agg(['mean', salary_range])

# Apply transformation (keeps original shape)
df['dept_avg_salary'] = df.groupby('department')['salary'].transform('mean')
print(df)
```

**Output:**
```
  department   name  salary  dept_avg_salary
0         IT    Ank   30000       48333.33
1         HR  Harsh   50000       47500.00
2         IT  Ankit   60000       48333.33
3         HR  Priya   45000       47500.00
4         IT  Rahul   55000       48333.33
```

### **Value Counts**

- **df['col'].value_counts()**: Count unique values in column
- **df['col'].value_counts(normalize=True)**: Get proportions instead of counts
- **df['col'].value_counts(dropna=False)**: Include NaN in counts
- **df['col'].value_counts(ascending=True)**: Sort in ascending order

**Example:**
```python
# Count occurrences
print(df['department'].value_counts())
```

**Output:**
```
department
IT    3
HR    2
Name: count, dtype: int64
```

```python
# With percentages
print(df['department'].value_counts(normalize=True))

# Include NaN in count
print(df['department'].value_counts(dropna=False))
```

### **Pivot Tables**

- **df.pivot_table(values, index, columns, aggfunc)**: Create pivot table with aggregation
- **df.pivot(index, columns, values)**: Reshape without aggregation
- **pd.crosstab(index, columns)**: Create cross-tabulation (frequency table)

**Example:**
```python
# Create pivot table
pivot = df.pivot_table(
    values='salary',
    index='department',
    aggfunc='mean'
)
print(pivot)

# Multiple aggregations
pivot = df.pivot_table(
    values='salary',
    index='department',
    aggfunc=['sum', 'mean', 'count']
)
print(pivot)

# Cross-tabulation
df['salary_category'] = pd.cut(df['salary'], bins=[0, 40000, 60000, 100000], 
                                labels=['Low', 'Medium', 'High'])
crosstab = pd.crosstab(df['department'], df['salary_category'])
print(crosstab)
```

---

## Merging & Joining

### **Concatenation**

- **pd.concat([df1, df2])**: Concatenate DataFrames vertically (stack rows)
- **pd.concat([df1, df2], axis=1)**: Concatenate horizontally (side by side)
- **pd.concat([df1, df2], ignore_index=True)**: Reset index after concatenation
- **pd.concat([df1, df2], keys=['A', 'B'])**: Add hierarchical index

**Example:**
```python
# Create sample DataFrames
df1 = pd.DataFrame({
    'name': ['Ank', 'Harsh'],
    'age': [18, 22]
})

df2 = pd.DataFrame({
    'name': ['Ankit', 'Priya'],
    'age': [28, 24]
})

# Vertical concatenation (stack rows)
result = pd.concat([df1, df2], ignore_index=True)
print(result)
```

**Output:**
```
    name  age
0    Ank   18
1  Harsh   22
2  Ankit   28
3  Priya   24
```

```python
# Horizontal concatenation (side by side)
df3 = pd.DataFrame({
    'salary': [30000, 50000]
})

result = pd.concat([df1, df3], axis=1)
print(result)
```

**Output:**
```
    name  age  salary
0    Ank   18   30000
1  Harsh   22   50000
```

### **Merging (SQL-like Joins)**

- **pd.merge(df1, df2, on='col')**: Merge DataFrames on common column (inner join by default)
- **pd.merge(df1, df2, on='col', how='inner')**: Inner join (only matching records)
- **pd.merge(df1, df2, on='col', how='left')**: Left join (all from left, matching from right)
- **pd.merge(df1, df2, on='col', how='right')**: Right join (all from right, matching from left)
- **pd.merge(df1, df2, on='col', how='outer')**: Outer join (all records from both)
- **pd.merge(df1, df2, left_on='col1', right_on='col2')**: Merge on different column names
- **pd.merge(df1, df2, on=['col1', 'col2'])**: Merge on multiple columns
- **pd.merge(df1, df2, suffixes=('_x', '_y'))**: Suffixes for duplicate columns

**Example:**
```python
# Create sample DataFrames
employees = pd.DataFrame({
    'emp_id': [1, 2, 3, 4],
    'name': ['Ank', 'Harsh', 'Ankit', 'Priya'],
    'dept_id': [101, 102, 101, 103]
})

departments = pd.DataFrame({
    'dept_id': [101, 102, 103],
    'dept_name': ['IT', 'HR', 'Finance']
})

# Inner join (only matching records)
result = pd.merge(employees, departments, on='dept_id', how='inner')
print(result)
```

**Output:**
```
   emp_id   name  dept_id dept_name
0       1    Ank      101        IT
1       3  Ankit      101        IT
2       2  Harsh      102        HR
3       4  Priya      103   Finance
```

```python
# Left join (all from left, matching from right)
result = pd.merge(employees, departments, on='dept_id', how='left')

# Right join (all from right, matching from left)
result = pd.merge(employees, departments, on='dept_id', how='right')

# Outer join (all records from both)
result = pd.merge(employees, departments, on='dept_id', how='outer')

# Merge on different column names
result = pd.merge(employees, departments, 
                  left_on='dept_id', right_on='dept_id')

# Merge on multiple columns
result = pd.merge(df1, df2, on=['col1', 'col2'])

# Merge with suffixes for duplicate columns
result = pd.merge(employees, departments, on='dept_id', 
                  suffixes=('_emp', '_dept'))
```

### **Join (Index-based)**

- **df1.join(df2)**: Join on index (left join by default)
- **df1.join(df2, how='inner')**: Inner join on index
- **df1.join(df2, on='col')**: Join on specific column from left DataFrame
- **df1.join(df2, lsuffix='_left', rsuffix='_right')**: Suffixes for duplicate columns

**Example:**
```python
# Set index
employees_indexed = employees.set_index('emp_id')
departments_indexed = departments.set_index('dept_id')

# Join on index
result = employees_indexed.join(departments_indexed, on='dept_id')
```

---

## Time Series

### **Creating Datetime Objects**

- **pd.to_datetime(series)**: Convert string/int to datetime
- **pd.to_datetime(series, format='%Y-%m-%d')**: Convert with specific format
- **pd.to_datetime(series, errors='coerce')**: Convert invalid dates to NaT
- **pd.Timestamp('2024-01-01')**: Create single timestamp
- **pd.DatetimeIndex(dates)**: Create datetime index

**Example:**
```python
# Convert string to datetime
df = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'value': [100, 150, 120]
})

df['date'] = pd.to_datetime(df['date'])
print(df.dtypes)

# Different datetime formats
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

# Handle errors
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Invalid dates become NaT
```

### **Date Range**

- **pd.date_range(start, end, freq)**: Create range of dates
- **pd.date_range(start, periods, freq)**: Create range with specific number of periods
- **pd.bdate_range(start, end)**: Business day range (excludes weekends)

**Frequency options**: 'D' (day), 'W' (week), 'M' (month), 'Q' (quarter), 'Y' (year), 'H' (hour), 'T' (minute), 'S' (second)

**Example:**
```python
# Create date range
dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
print(dates)

# Specific number of periods
dates = pd.date_range(start='2024-01-01', periods=10, freq='D')

# Different frequencies
dates = pd.date_range(start='2024-01-01', periods=10, freq='H')  # Hourly
dates = pd.date_range(start='2024-01-01', periods=10, freq='W')  # Weekly
dates = pd.date_range(start='2024-01-01', periods=10, freq='M')  # Monthly
dates = pd.date_range(start='2024-01-01', periods=10, freq='Q')  # Quarterly
dates = pd.date_range(start='2024-01-01', periods=10, freq='Y')  # Yearly
```

### **Extracting Date Components**

- **df['date'].dt.year**: Extract year
- **df['date'].dt.month**: Extract month
- **df['date'].dt.day**: Extract day
- **df['date'].dt.hour**: Extract hour
- **df['date'].dt.minute**: Extract minute
- **df['date'].dt.dayofweek**: Day of week (0=Monday, 6=Sunday)
- **df['date'].dt.dayofyear**: Day of year (1-365/366)
- **df['date'].dt.week**: Week number of year
- **df['date'].dt.quarter**: Quarter (1-4)
- **df['date'].dt.month_name()**: Month name ('January', 'February', etc.)
- **df['date'].dt.day_name()**: Day name ('Monday', 'Tuesday', etc.)
- **df['date'].dt.date**: Date component only (without time)
- **df['date'].dt.time**: Time component only

**Example:**
```python
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'value': np.random.randint(100, 200, 100)
})

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['dayofyear'] = df['date'].dt.dayofyear
df['weekofyear'] = df['date'].dt.isocalendar().week
df['quarter'] = df['date'].dt.quarter
df['month_name'] = df['date'].dt.month_name()
df['day_name'] = df['date'].dt.day_name()

print(df.head())
```

### **Resampling (Aggregating Time Series)**

- **df.resample('M').mean()**: Resample to monthly and calculate mean
- **df.resample('W').sum()**: Resample to weekly and calculate sum
- **df.resample('D').agg({'col': 'mean'})**: Resample with custom aggregation
- **df.resample('M').ffill()**: Forward fill after resampling
- **df.resample('M').bfill()**: Backward fill after resampling

**Example:**
```python
# Set date as index
df = df.set_index('date')

# Resample to monthly average
monthly = df.resample('M').mean()
print(monthly)

# Resample to weekly sum
weekly = df.resample('W').sum()

# Different aggregations
resampled = df.resample('M').agg({
    'value': ['sum', 'mean', 'min', 'max']
})

# Forward fill missing values after resampling
daily = df.resample('D').ffill()

# Backward fill
daily = df.resample('D').bfill()
```

### **Time Shifting**

- **df.shift(n)**: Shift data forward by n periods
- **df.shift(-n)**: Shift data backward by n periods
- **df['col'].pct_change()**: Calculate percentage change
- **df['col'].diff()**: Calculate difference from previous value

**Example:**
```python
# Shift data forward
df['previous_day'] = df['value'].shift(1)

# Shift data backward
df['next_day'] = df['value'].shift(-1)

# Calculate day-over-day change
df['change'] = df['value'] - df['value'].shift(1)
df['pct_change'] = df['value'].pct_change() * 100  # Percentage change
```

### **Rolling Windows (Moving Averages)**

- **df['col'].rolling(window).mean()**: Rolling mean
- **df['col'].rolling(window).sum()**: Rolling sum
- **df['col'].rolling(window).std()**: Rolling standard deviation
- **df['col'].rolling(window).min()**: Rolling minimum
- **df['col'].rolling(window).max()**: Rolling maximum
- **df['col'].expanding().mean()**: Expanding window (cumulative)
- **df['col'].ewm(span).mean()**: Exponential moving average

**Example:**
```python
# 7-day moving average
df['MA_7'] = df['value'].rolling(window=7).mean()

# 30-day moving average
df['MA_30'] = df['value'].rolling(window=30).mean()

# Rolling sum
df['rolling_sum'] = df['value'].rolling(window=7).sum()

# Rolling standard deviation
df['volatility'] = df['value'].rolling(window=7).std()

# Exponential moving average
df['EMA'] = df['value'].ewm(span=7, adjust=False).mean()
```

---

## String Operations

### **Basic String Methods**

- **df['col'].str.lower()**: Convert to lowercase
- **df['col'].str.upper()**: Convert to uppercase
- **df['col'].str.title()**: Convert to title case
- **df['col'].str.capitalize()**: Capitalize first letter
- **df['col'].str.strip()**: Remove leading/trailing whitespace
- **df['col'].str.lstrip()**: Remove leading whitespace
- **df['col'].str.rstrip()**: Remove trailing whitespace
- **df['col'].str.len()**: Length of string
- **df['col'].str.slice(start, end)**: Slice string

**Example:**
```python
data = {
    "name": ["Ank Kumar", "HARSH sharma", "ankit VERMA"],
    "email": ["ank@example.com", "harsh@EXAMPLE.com", "ankit@Example.Com"]
}
df = pd.DataFrame(data)

# Convert to lowercase
df['name_lower'] = df['name'].str.lower()
print(df['name_lower'])
```

**Output:**
```
0      ank kumar
1    harsh sharma
2    ankit verma
Name: name_lower, dtype: object
```

```python
# Convert to uppercase
df['name_upper'] = df['name'].str.upper()

# Convert to title case
df['name_title'] = df['name'].str.title()

# Strip whitespace
df['name'] = df['name'].str.strip()

# Remove leading/trailing characters
df['name'] = df['name'].str.strip('.')

# Length of string
df['name_length'] = df['name'].str.len()
```

### **String Searching**

- **df['col'].str.contains('pattern')**: Check if contains substring
- **df['col'].str.contains('pattern', case=False)**: Case-insensitive search
- **df['col'].str.startswith('pattern')**: Check if starts with
- **df['col'].str.endswith('pattern')**: Check if ends with
- **df['col'].str.find('substring')**: Find position of substring (-1 if not found)
- **df['col'].str.match('pattern')**: Match regex pattern from start

**Example:**
```python
# Check if contains substring
df['has_kumar'] = df['name'].str.contains('Kumar', case=False)
print(df[['name', 'has_kumar']])
```

**Output:**
```
             name  has_kumar
0       Ank Kumar       True
1    HARSH sharma      False
2    ankit VERMA       False
```

```python
# Check if starts with
df['starts_with_a'] = df['name'].str.startswith('A', na=False)

# Check if ends with
df['ends_with_a'] = df['name'].str.endswith('a', na=False)

# Find position of substring
df['kumar_position'] = df['name'].str.find('Kumar')

# Match with regex
df['has_pattern'] = df['email'].str.match(r'.*@example\.com')
```

### **String Replacement**

- **df['col'].str.replace('old', 'new')**: Replace substring
- **df['col'].str.replace('pattern', 'new', regex=True)**: Replace using regex
- **df['col'].str.translate(table)**: Translate characters using table

**Example:**
```python
# Replace substring
df['name_replaced'] = df['name'].str.replace('Kumar', 'Singh')

# Replace with regex
df['email_clean'] = df['email'].str.replace(r'@.*\.com', '@domain.com', regex=True)

# Replace multiple patterns
df['name'] = df['name'].str.replace('Kumar|Sharma|VERMA', 'Redacted', regex=True)
```

### **String Splitting**

- **df['col'].str.split('delimiter')**: Split by delimiter (returns list)
- **df['col'].str.split('delimiter', expand=True)**: Split and expand to columns
- **df['col'].str.split('delimiter').str[0]**: Get specific part after split
- **df['col'].str.rsplit('delimiter', n=1)**: Split from right

**Example:**
```python
# Split by delimiter
df[['first_name', 'last_name']] = df['name'].str.split(' ', expand=True)
print(df[['first_name', 'last_name']])
```

**Output:**
```
  first_name last_name
0        Ank     Kumar
1      HARSH    sharma
2      ankit     VERMA
```

```python
# Split and get specific part
df['first_name'] = df['name'].str.split(' ').str[0]

# Split email to get domain
df['domain'] = df['email'].str.split('@').str[1]
```

### **String Concatenation**

- **df['col1'] + df['col2']**: Concatenate two columns
- **df['col'].str.cat(df['col2'], sep=' ')**: Concatenate with separator
- **df['col'].str.cat([df['col2'], df['col3']], sep=', ')**: Concatenate multiple columns

**Example:**
```python
# Concatenate columns
df['full_info'] = df['first_name'] + ' - ' + df['email']

# Using str.cat()
df['full_info'] = df['first_name'].str.cat(df['email'], sep=' | ')
```

### **Extracting with Regex**

- **df['col'].str.extract('(pattern)')**: Extract first match
- **df['col'].str.extract('(pattern1)(pattern2)')**: Extract multiple groups
- **df['col'].str.extractall('(pattern)')**: Extract all matches
- **df['col'].str.findall('pattern')**: Find all matches as list

**Example:**
```python
# Extract email username
df['username'] = df['email'].str.extract(r'(.*)@')

# Extract multiple groups
df[['username', 'domain']] = df['email'].str.extract(r'(.*)@(.*)')
```

---

## Visualization

### **Basic Plots**

- **df.plot()**: Line plot of all columns
- **df.plot(x='col1', y='col2')**: Line plot with specific x and y
- **df.plot(kind='line')**: Line plot (explicit)
- **df.plot(kind='bar')**: Bar plot
- **df.plot(kind='barh')**: Horizontal bar plot
- **df.plot(kind='hist')**: Histogram
- **df.plot(kind='box')**: Box plot
- **df.plot(kind='scatter', x='col1', y='col2')**: Scatter plot
- **df.plot(kind='area')**: Area plot
- **df.plot(kind='pie', y='col')**: Pie chart
- **df.plot(kind='density')**: Density plot (KDE)
- **df.plot(kind='hexbin', x='col1', y='col2')**: Hexbin plot

**Example:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
df = pd.DataFrame({
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'sales': [100, 150, 120, 180, 200],
    'expenses': [80, 90, 85, 100, 110]
})

# Line plot
df.plot(x='month', y='sales', kind='line', title='Monthly Sales')
plt.show()

# Multiple lines
df.plot(x='month', y=['sales', 'expenses'], kind='line')
plt.show()

# Bar plot
df.plot(x='month', y='sales', kind='bar', title='Sales by Month')
plt.show()

# Horizontal bar plot
df.plot(x='month', y='sales', kind='barh')
plt.show()

# Stacked bar plot
df.plot(x='month', y=['sales', 'expenses'], kind='bar', stacked=True)
plt.show()
```

### **Statistical Plots**

- **df['col'].plot(kind='hist', bins=n)**: Histogram with n bins
- **df[['col1', 'col2']].plot(kind='box')**: Box plot for multiple columns
- **df.plot(kind='kde')**: Kernel density estimate plot
- **pd.plotting.scatter_matrix(df)**: Scatter plot matrix
- **pd.plotting.parallel_coordinates(df, 'class_col')**: Parallel coordinates plot
- **pd.plotting.andrews_curves(df, 'class_col')**: Andrews curves

**Example:**
```python
# Histogram
df['sales'].plot(kind='hist', bins=10, title='Sales Distribution')
plt.show()

# Box plot
df[['sales', 'expenses']].plot(kind='box')
plt.show()

# Scatter plot
df.plot(x='sales', y='expenses', kind='scatter')
plt.show()

# Area plot
df.plot(x='month', y=['sales', 'expenses'], kind='area', alpha=0.5)
plt.show()

# Pie chart
df.set_index('month')['sales'].plot(kind='pie', autopct='%1.1f%%')
plt.show()
```

### **Advanced Plotting**

- **df.plot(color='color_name')**: Set color
- **df.plot(linewidth=n)**: Set line width
- **df.plot(marker='o')**: Add markers to line plot
- **df.plot(title='Title')**: Set plot title
- **df.plot(xlabel='X Label', ylabel='Y Label')**: Set axis labels
- **df.plot(legend=True/False)**: Show/hide legend
- **df.plot(grid=True/False)**: Show/hide grid
- **df.plot(figsize=(width, height))**: Set figure size
- **df.plot(ax=axes_object)**: Plot on specific axes (for subplots)

**Example:**
```python
# Subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

df.plot(x='month', y='sales', kind='line', ax=axes[0, 0], title='Line')
df.plot(x='month', y='sales', kind='bar', ax=axes[0, 1], title='Bar')
df.plot(x='month', y='sales', kind='area', ax=axes[1, 0], title='Area')
df['sales'].plot(kind='hist', ax=axes[1, 1], title='Histogram')

plt.tight_layout()
plt.show()

# Customization
df.plot(
    x='month', 
    y='sales',
    kind='line',
    color='red',
    linewidth=2,
    marker='o',
    title='Monthly Sales',
    xlabel='Month',
    ylabel='Sales ($)',
    legend=True,
    grid=True,
    figsize=(10, 6)
)
plt.show()
```

---

## Advanced Operations

### **Advanced Indexing**

- **df.set_index('col')**: Set column as index
- **df.set_index(['col1', 'col2'])**: Set MultiIndex
- **df.reset_index()**: Reset index to default integer index
- **df.reset_index(drop=True)**: Reset without keeping old index as column
- **df.xs(key, level)**: Cross-section from MultiIndex
- **df.swaplevel(i, j)**: Swap levels in MultiIndex
- **df.sort_index(level=0)**: Sort by specific index level

### **Reshaping Data**

- **df.melt(id_vars, value_vars)**: Unpivot from wide to long format
- **df.pivot(index, columns, values)**: Pivot from long to wide format
- **df.stack()**: Pivot columns to rows (create MultiIndex)
- **df.unstack()**: Pivot rows to columns (inverse of stack)
- **df.T**: Transpose DataFrame (swap rows and columns)
- **df.explode('col')**: Expand lists/arrays in column into separate rows
- **pd.wide_to_long(df, stubnames, i, j)**: Reshape from wide to long based on stub names

**Example:**
```python
# Melt (wide to long)
df_long = df.melt(id_vars=['month'], 
                   value_vars=['sales', 'expenses'],
                   var_name='category', 
                   value_name='amount')

# Pivot (long to wide)
df_wide = df_long.pivot(index='month', columns='category', values='amount')

# Transpose
df_transposed = df.T
```

### **Window Functions**

- **df.rolling(window).apply(func)**: Apply custom function to rolling window
- **df.rolling(window, on='date_col')**: Rolling on specific date column
- **df.expanding().mean()**: Expanding window (cumulative)
- **df.rolling(window).corr(other_series)**: Rolling correlation

### **Working with Categories**

- **df['col'].astype('category')**: Convert to categorical type
- **df['col'].cat.categories**: View categories
- **df['col'].cat.rename_categories(new_names)**: Rename categories
- **df['col'].cat.add_categories(['new'])**: Add new categories
- **df['col'].cat.remove_unused_categories()**: Remove unused categories

### **Dummy Variables**

- **pd.get_dummies(df, columns=['col'])**: Convert categorical to dummy/indicator variables
- **pd.get_dummies(df, prefix='prefix')**: Add prefix to dummy column names
- **pd.get_dummies(df, drop_first=True)**: Drop first category to avoid multicollinearity

**Example:**
```python
# Create dummy variables
df_dummies = pd.get_dummies(df, columns=['department'])
print(df_dummies)

# With prefix
df_dummies = pd.get_dummies(df, columns=['department'], prefix='dept')
```

### **Evaluation**

- **df.eval('new_col = col1 + col2')**: Evaluate expression and create new column
- **df.query('col > value')**: Query DataFrame using string expression
- **pd.eval('df1.col + df2.col')**: Evaluate expression across DataFrames

### **Sample & Permutation**

- **df.sample(n)**: Randomly sample n rows
- **df.sample(frac=0.5)**: Sample fraction of rows (0.5 = 50%)
- **df.sample(n, replace=True)**: Sample with replacement
- **np.random.permutation(df.index)**: Randomly permute index

---

## Performance Tips

### **Memory Optimization**

- **df.memory_usage(deep=True)**: Get detailed memory usage of DataFrame
- **df.info(memory_usage='deep')**: Show memory usage in info
- **df.astype('int32')**: Downcast int64 to int32 to save memory
- **df.astype('float32')**: Downcast float64 to float32
- **df['col'].astype('category')**: Convert to category for repeated values
- **pd.to_numeric(df['col'], downcast='integer')**: Automatically downcast numeric types

**Example:**
```python
# Check memory usage
print(df.memory_usage(deep=True))
print(df.info(memory_usage='deep'))

# Optimize data types
# Convert int64 to int32 (if values fit)
df['age'] = df['age'].astype('int32')

# Convert float64 to float32
df['salary'] = df['salary'].astype('float32')

# Convert to category for repeated values
df['department'] = df['department'].astype('category')

# Example of memory savings
before = df.memory_usage(deep=True).sum()
df['city'] = df['city'].astype('category')
after = df.memory_usage(deep=True).sum()
print(f"Memory saved: {before - after} bytes")
```

### **Efficient Operations**

**Best Practices:**
- ✅ Use vectorized operations instead of loops
- ✅ Use `.loc/.iloc` instead of chained indexing
- ✅ Use `.query()` for complex filtering on large datasets
- ✅ Use `inplace=True` to avoid creating copies (saves memory)
- ❌ Avoid iterating with `.iterrows()` (very slow)
- ✅ Use `.apply()` with `raw=True` for numeric operations
- ✅ Use `.eval()` for arithmetic operations on large datasets

**Example:**
```python
# Use vectorized operations instead of loops
# ❌ BAD (Slow)
for i in range(len(df)):
    df.at[i, 'bonus'] = df.at[i, 'salary'] * 0.1

# ✅ GOOD (Fast)
df['bonus'] = df['salary'] * 0.1

# Use .loc/.iloc instead of chained indexing
# ❌ BAD (May cause SettingWithCopyWarning)
df[df['age'] > 25]['salary'] = 50000

# ✅ GOOD
df.loc[df['age'] > 25, 'salary'] = 50000

# Use .query() for complex filtering (faster for large datasets)
# ✅ GOOD
result = df.query("age > 25 and salary < 50000")
```

### **Chunking for Large Files**

**Example:**
```python
# Process large CSV in chunks
chunk_size = 10000
chunks = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process each chunk
    filtered = chunk[chunk['age'] > 18]
    chunks.append(filtered)

# Combine all chunks
df = pd.concat(chunks, ignore_index=True)

# Or process without storing in memory
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process and save immediately
    result = chunk[chunk['age'] > 18]
    result.to_csv('output.csv', mode='a', header=False, index=False)
```

### **Using Dask for Huge Datasets**

```python
# For datasets too large for memory, use Dask
import dask.dataframe as dd

# Read with Dask (lazy loading)
ddf = dd.read_csv('huge_file.csv')

# Operations are lazy (not computed immediately)
result = ddf[ddf['age'] > 25].groupby('department')['salary'].mean()

# Compute when needed
result_computed = result.compute()
```

### **Parallel Processing**

```python
# Use parallel processing for apply operations
import multiprocessing as mp

def process_row(row):
    # Your processing logic
    return row['salary'] * 1.1

# Number of CPU cores
n_cores = mp.cpu_count()

# Split DataFrame
df_split = np.array_split(df, n_cores)

# Process in parallel
with mp.Pool(n_cores) as pool:
    df_processed = pd.concat(pool.map(lambda x: x.apply(process_row, axis=1), df_split))
```

---

## Quick Reference

### **DataFrame Creation**
- **pd.Series(data, index)**: Create 1D array
- **pd.DataFrame(data, index, columns)**: Create 2D table
- **pd.read_csv('file.csv')**: Read CSV file
- **pd.read_excel('file.xlsx')**: Read Excel file
- **pd.read_json('file.json')**: Read JSON file
- **pd.read_html('url')**: Read HTML tables
- **pd.read_sql(query, connection)**: Read from SQL
- **pd.read_parquet('file.parquet')**: Read Parquet file

### **Data Inspection**
- **df.head(n)**: First n rows
- **df.tail(n)**: Last n rows
- **df.info()**: Data types and memory
- **df.describe()**: Statistical summary
- **df.shape**: (rows, columns)
- **df.columns**: Column names
- **df.index**: Row index
- **df.dtypes**: Data types
- **df.memory_usage()**: Memory usage

### **Selection & Indexing**
- **df['column']**: Select column (Series)
- **df[['col1', 'col2']]**: Select columns (DataFrame)
- **df.loc[row_label]**: Select by label
- **df.iloc[row_index]**: Select by position
- **df.at[row, col]**: Single value by label (fast)
- **df.iat[row, col]**: Single value by position (fast)
- **df[df['age'] > 25]**: Boolean indexing
- **df.query("age > 25")**: SQL-like filtering

### **Data Cleaning**
- **df.isnull()**: Check for nulls
- **df.notnull()**: Check for non-nulls
- **df.dropna()**: Drop rows with nulls
- **df.fillna(value)**: Fill nulls
- **df.replace(old, new)**: Replace values
- **df.drop_duplicates()**: Remove duplicates
- **df.astype(type)**: Change data type
- **df.rename(columns={})**: Rename columns
- **df.drop(columns=[])**: Drop columns

### **Data Manipulation**
- **df['new_col'] = value**: Add column
- **df.insert(loc, col, val)**: Insert column at position
- **df.drop(columns=[])**: Remove columns
- **df.apply(func)**: Apply function
- **df.sort_values(by='col')**: Sort by column
- **df.sort_index()**: Sort by index

### **Aggregation & Grouping**
- **df.groupby('col')**: Group by column
- **df.agg(func)**: Aggregate
- **df['col'].value_counts()**: Count unique values
- **df.pivot_table()**: Create pivot table
- **df['col'].sum()**: Sum
- **df['col'].mean()**: Average
- **df['col'].median()**: Median
- **df['col'].min()**: Minimum
- **df['col'].max()**: Maximum
- **df['col'].std()**: Standard deviation
- **df['col'].count()**: Count non-null

### **Merging & Joining**
- **pd.concat([df1, df2])**: Concatenate
- **pd.merge(df1, df2)**: Merge (join)
- **df1.join(df2)**: Join on index

### **Time Series**
- **pd.to_datetime(series)**: Convert to datetime
- **pd.date_range(start, end)**: Create date range
- **df.resample('M').mean()**: Resample to monthly
- **df['col'].shift(1)**: Shift data
- **df['col'].rolling(7)**: Rolling window
- **df['date'].dt.year**: Extract year
- **df['date'].dt.month**: Extract month
- **df['date'].dt.day**: Extract day

### **String Operations**
- **df['col'].str.lower()**: Lowercase
- **df['col'].str.upper()**: Uppercase
- **df['col'].str.contains()**: Check if contains
- **df['col'].str.replace()**: Replace substring
- **df['col'].str.split()**: Split string
- **df['col'].str.strip()**: Remove whitespace
- **df['col'].str.len()**: String length

### **Visualization**
- **df.plot()**: Line plot
- **df.plot(kind='bar')**: Bar chart
- **df.plot(kind='hist')**: Histogram
- **df.plot(kind='box')**: Box plot
- **df.plot(kind='scatter')**: Scatter plot

### **Exporting Data**
- **df.to_csv('file.csv')**: Export to CSV
- **df.to_excel('file.xlsx')**: Export to Excel
- **df.to_json('file.json')**: Export to JSON
- **df.to_parquet('file')**: Export to Parquet
- **df.to_sql(table, conn)**: Export to database
- **df.to_html('file.html')**: Export to HTML

---

## Important Notes for Big Data Engineers

### 1. Memory Management
- Always use `dtype` optimization for large datasets
- Use `category` dtype for columns with repeated values
- Consider using `Dask` or `Modin` for datasets larger than RAM
- Use chunking when reading large files

### 2. Performance Best Practices
- Avoid loops; use vectorized operations
- Use `.loc` and `.iloc` instead of chained indexing
- Use `.query()` for complex filtering on large datasets
- Set `inplace=True` to avoid creating copies (saves memory)

### 3. File Formats for Big Data
- **CSV**: Universal but slow and large
- **Parquet**: Columnar, compressed, fast (recommended for big data)
- **Feather**: Fast for temporary storage
- **HDF5**: Good for hierarchical data

### 4. Common Pitfalls
- **SettingWithCopyWarning**: Always use `.loc` for assignments
- **Memory leaks**: Use `del df` and `gc.collect()` for large objects
- **Encoding issues**: Always specify encoding when reading/writing files
- **Datetime parsing**: Use `parse_dates` parameter in `read_csv()`

### 5. Integration with Big Data Tools
```python
# PySpark
spark_df = spark.createDataFrame(pandas_df)

# SQL Databases
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@localhost/db')
df.to_sql('table_name', engine)

# Apache Arrow (for fast data transfer)
import pyarrow as pa
table = pa.Table.from_pandas(df)
```

---

## Conclusion

Pandas is an essential tool for any data professional. This guide covers the most important operations with function descriptions for quick recall. Always refer to the [official documentation](https://pandas.pydata.org/docs/) for detailed information.

### Key Takeaways
1. Use vectorized operations for performance
2. Optimize data types to save memory
3. Use appropriate file formats (Parquet for big data)
4. Master groupby and aggregation for analysis
5. Practice with real datasets to build proficiency

Happy Data Wrangling! 🐼📊