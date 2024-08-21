# Movie Recommendation system

<table style="border: none; border-collapse: collapse;">
  <tr style="border: none;">
     <td style="border: none; vertical-align: top;">
      Happy to see you there, Welcome to this repository here we will see the movie recommendation based on the rating , tags using various ML algorithm.
    </td>
    <td style="border: none;">
      <img src="https://media-s3-us-east-1.ceros.com/gumgum/images/2018/08/30/a4ef384f2a79aeee4847721ec28d8720/machine-training-animation-1.gif" alt="ML GIF" width="300"/>
    </td>
  </tr>
</table>



## 📊 Project Overview

This project implements a hybrid recommendation system that combines content-based and collaborative filtering techniques. It recommends movies based on both the content of movies and user preferences.

### Import Libraries::
   - **pandas**: For handling and analyzing data.
   - **scikit-learn**: For calculating similarities between movies.
   - **numpy**: For numerical operations.

### Load Data::
   - Read three datasets: `movies.csv`, `tags.csv`, and `ratings.csv`. 
   - `movies.csv` contains movie details.
   - `tags.csv` contains movie tags.
   - `ratings.csv` contains user ratings for movies.

### Prepare Content-Based Filtering:
   - **Combine Tags and Genres**: Merge movie tags with movie genres to create a description for each movie.
   - **Compute Content Similarity**: Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert movie descriptions into numerical vectors and calculate how similar each movie is to every other movie based on these descriptions.

### Prepare Collaborative Filtering:
   - **Create User-Movie Matrix**: Make a matrix where rows represent users, columns represent movies, and cells contain ratings given by users.
   - **Filter Movies**: Ensure that only movies present in both the user-movie matrix and the content similarity matrix are used.
   - **Compute Collaborative Similarity**: Calculate how similar movies are based on user ratings.

### Hybrid Recommendation:
   - **Combine Scores**: Use both content-based and collaborative filtering scores to recommend movies. The `alpha` parameter controls the weight given to each type of score.
   - **Get Recommendations**: Sort movies by their combined score and return the top recommendations.

### Example Usage:
  - Call the function with a movie title and a user ID to get a list of recommended movies.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Libraries:** Scikit-learn, SVM, Decision Tree, Random Forest, Pandas, NumPy, Matplotlib, Seaborn
- **Data Processing:** Pandas, NumPy
- **Modeling:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook, Anaconda

## 🌐 Environment Setup

To replicate the environment used for this project, ensure you have the following installed:
- **Python 3.x**
- **Jupyter Notebook**
- **Anaconda** for package management

### 🚀 Skills Gained

Through this internship and the development of this project, I have gained hands-on experience in:

- **Machine Learning:** Understanding and implementing various machine learning models, including ensemble learning techniques.
- **Data Preprocessing:** Cleaning, transforming, and preparing data for analysis.
- **Feature Engineering:** Creating and selecting features that enhance model performance.
- **Model Evaluation:** Analyzing model performance using various metrics and fine-tuning hyperparameters.
- **Visualization:** Using data visualization tools to gain insights from data and model predictions.
- **Team Collaboration:** Collaborating with team members, using version control (Git), and documenting the project effectively.

### ⏳ Internship Experience

Over the course of this internship, I dedicated 8 weeks to developing this project. This experience significantly enhanced my understanding of advanced machine learning concepts, particularly in ensemble learning. It also honed my skills in working with real-world datasets and translating complex data into actionable insights.
