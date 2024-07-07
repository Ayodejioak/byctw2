Video Game Sales Analysis

Overview:
This project analyzes a comprehensive dataset of video game sales to uncover market trends and provide insights for a gaming company preparing to launch a new video game. The analysis focuses on understanding popular genres, the impact of review scores on sales, and how sales have evolved over time.

Dataset:
The dataset contains information on over 55,000 video games, including:
- Game details (Name, Platform, Genre, ESRB Rating)
- Sales figures (Global, North America, Europe, Japan, Other regions)
- Review scores (Critic and User)
- Release Year

Source: vgchartz.com (as of April 2019)

Project Structure:
- `byct_w2.py`: Main Python script containing all analysis code
- `byct_w2.ipynb` : Python script with analysis code
- `VideoGameSales.csv`: Original dataset
- `requirements.txt`: List of required Python libraries
- `README.md`: This file, containing project information and instructions


Analysis Steps
1. Data Cleaning and Exploration
- Loaded and inspected the dataset
- Handled missing values by dropping rows with null entries
- Created histograms for key numerical columns (Critic Score, User Score, Global Sales)

2. Basic Grouping and Analysis
- Analyzed top genres by total global sales
- Calculated correlation between critic scores and global sales
- Visualized the trend of average global sales over time

3. Advanced Analysis and Visualization
- Compared sales performance of the Action genre across different platforms
- Built a linear regression model to predict global sales based on critic and user scores
- Created an interactive scatter plot using Plotly to explore the relationship between critic scores and global sales across genres and years


Key Findings:
1. The top-selling genre is Action
2. There is a moderate positive correlation (r = 0.50) between critic scores and global sales
3. Global sales have shown a fluctuating trend over time, with notable peaks around 2000 and 2010-2015, followed by a sharp decline in recent years.
4. The PS 3 seems to perform best for Action games in terms of global sales
5. Our predictive model for global sales based on review scores has an R-squared value of 0.37
6. The Mean Squared Error of our predictive model is 2.85, which gives an idea of the average prediction error in millions of units sold.

For detailed results and visualizations, please refer to the Jupyter notebook.

Challenges and Solutions
During this analysis, I encountered and addressed the following challenges:
1. Challenge: Large dataset impacting performance
   Solution: Optimized code using vectorized operations in pandas
   
2. Challenge: Outliers in sales data skewing visualizations
   Solution: Used log transformation for sales data in certain plots

3. Challenge: Interpreting the relationship between multiple variables
   Solution: Utilized interactive visualizations with Plotly to allow for dynamic exploration

4. Challenge: Missing data - Some entries lacked critic or user scores.
   Solution: I chose to drop these rows to ensure data quality for our analysis.

How to Run
1. Ensure you have Python 3.8+ installed
2. Clone this repository
3. Navigate to the project directory
4. Install required libraries: `pip install -r requirements.txt`
5. Run the analysis script: `byct_w2.py`

Required Libraries
- pandas==1.3.3
- matplotlib==3.4.3
- seaborn==0.11.2
- scikit-learn==0.24.2
- plotly==5.3.1

Future Work
- Incorporate machine learning models to predict game success based on multiple features
- Analyze the impact of ESRB ratings on sales across different regions
- Investigate seasonal trends in game releases and their effect on sales

Author
Ayodeji Ogunjinmi

Acknowledgments
- Dataset provided by vgchartz.com
- Inspired by the growing importance of data-driven decision making in the gaming industry
- Instructor Samuel Nnitiwe Theophilus
