# YouTube Video Data Analysis

## Introduction
In this analysis, we explored YouTube video data to understand key trends, correlations, and factors influencing video popularity. We cleaned the dataset, performed exploratory data analysis (EDA), visualized relationships, and built a predictive model for estimating video views based on engagement metrics.

## Data Collection
The dataset used in this analysis is `USvideos.csv`, which contains information about YouTube videos, including:
- **Title**
- **Views**
- **Likes**
- **Dislikes**
- **Comment Count**
- **Description**

## Data Cleaning
- The `description` column contained missing values, which were replaced with an empty string.
- New features such as `title_length` (length of video title) and `contains_capitalized` (whether the title contains capitalized words) were created to extract insights.

## Exploratory Data Analysis (EDA)

### Descriptive Statistics
Basic statistical properties of the dataset were examined using `df.describe()` to summarize distributions of numerical columns.

### Feature Correlation
A **heatmap** was plotted to analyze correlations between numerical features:
```python
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
```
**Findings:**
- Views showed strong positive correlation with **likes and comment count**.
- Dislikes had a weaker correlation with views, suggesting dislikes alone are not a major factor.

### Title Length Analysis
We analyzed the distribution of title lengths to understand how long video titles impact engagement:
```python
df["title_length"] = df["title"].apply(len)
sns.histplot(df["title_length"], bins=30, kde=True, color="blue")
plt.xlabel("Title Length")
plt.ylabel("Count")
plt.title("Distribution of Title Lengths")
plt.show()
```
**Findings:**
- Most titles were between 20-80 characters.
- Extremely short or long titles were rare.

### Word Cloud of Titles
To understand common words in video titles, we generated a **word cloud**:
```python
from wordcloud import WordCloud

title_words = " ".join(df["title"])
wc = WordCloud(width=1200, height=600, background_color="white").generate(title_words)
plt.figure(figsize=(15, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of YouTube Video Titles")
plt.show()
```
**Findings:**
- Common words included "official", "trailer", "new", indicating that entertainment and promotional content dominate trending videos.

### Views vs. Likes
A scatter plot was created to visualize the relationship between **views** and **likes**:
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["views"], y=df["likes"], alpha=0.5, color='red')
plt.xlabel("Views")
plt.ylabel("Likes")
plt.title("Views vs Likes Scatter Plot")
plt.xscale("log")
plt.yscale("log")
plt.show()
```
**Findings:**
- A **strong positive correlation** was observed, meaning popular videos tend to receive more likes.

## Predictive Modeling
We used a **Linear Regression model** to predict **video views** based on engagement metrics (`likes`, `dislikes`, `comment_count`).

### Model Training
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

features = df[["likes", "dislikes", "comment_count"]]
target = df["views"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
```
### Model Evaluation
```python
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")
```

**Results:**
- **R-squared Score**: **0.775** (The model explains **77.5% of the variance** in video views).
- **Mean Absolute Error (MAE)**: Moderate error, indicating potential improvements by adding more features.

## Conclusion
- **Likes and comment count** are strong indicators of video popularity.
- **Dislikes alone** do not significantly impact view count.
- **Linear Regression provides reasonable accuracy**, but advanced models like **Random Forest** or **XGBoost** could improve predictions.
- Further enhancements include analyzing **video categories, upload times, and engagement trends over time**.

This analysis provides a foundation for understanding **YouTube video engagement trends** and building **better predictive models**.

