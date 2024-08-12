import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset from the 'automobile' folder within your project
# Assuming 'Automobile.csv' is in a folder named 'automobile' in your project directory
df = pd.read_csv('Automobile.csv')

# Set up the page
st.title("Automobile Dataset Analysis")
st.write("This app provides an analysis of the Automobile dataset with various charts and interactive features.")

# Display dataset overview
st.header("Dataset Overview")
st.write("### First few rows of the dataset")
st.dataframe(df.head())

# Display summary statistics
st.write("### Summary Statistics")
st.write(df.describe(include='all'))

# Handle missing values for horsepower
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)

# MPG Distribution
st.header("MPG Distribution")
mpg_min, mpg_max = st.slider("Select MPG Range", float(df['mpg'].min()), float(df['mpg'].max()), (float(df['mpg'].min()), float(df['mpg'].max())))
filtered_df = df[(df['mpg'] >= mpg_min) & (df['mpg'] <= mpg_max)]
st.write(f"Cars with MPG between {mpg_min} and {mpg_max}")
st.dataframe(filtered_df)

# MPG Histogram
st.subheader("MPG Histogram")
fig, ax = plt.subplots()
ax.hist(filtered_df['mpg'], bins=15, color='skyblue', edgecolor='black')
ax.set_xlabel('MPG')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Scatter Plot: Horsepower vs. MPG
st.header("Horsepower vs. MPG")
cylinders = st.selectbox("Select Number of Cylinders", sorted(df['cylinders'].unique()))
filtered_df_cyl = df[df['cylinders'] == cylinders]
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_df_cyl, x='horsepower', y='mpg', hue='origin', ax=ax, palette='Set2')
ax.set_xlabel('Horsepower')
ax.set_ylabel('MPG')
ax.set_title(f'Horsepower vs. MPG (Cylinders: {cylinders})')
st.pyplot(fig)

# Correlation Heatmap
st.header("Correlation Heatmap")

# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['float64', 'int64'])  

# Calculate the correlation matrix
corr = numeric_df.corr()

# Plot the heatmap
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation between Numerical Features')
st.pyplot(fig)

# Box Plot: MPG by Origin
st.header("MPG by Origin")
fig, ax = plt.subplots()
sns.boxplot(data=df, x='origin', y='mpg', ax=ax, palette='Set3')
ax.set_title('MPG Distribution by Origin')
st.pyplot(fig)

# Pair Plot
st.header("Pair Plot of Key Features")
features = st.multiselect("Select features to include in the pair plot", ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration'], default=['mpg', 'displacement', 'horsepower'])
if features:
    sns.pairplot(df[features + ['origin']], hue='origin', palette='Set2')
    st.pyplot(plt)

# Conclusion
st.header("Conclusion")
st.write("This analysis provides insights into the relationship between various attributes of automobiles, such as horsepower, weight, and MPG. You can explore further by selecting different cylinders, adjusting MPG ranges, and analyzing different feature combinations.")
