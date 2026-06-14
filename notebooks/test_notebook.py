import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Importing the necessary libraries:
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    import plotly.express as px
    import pickle
    import shap
    import seaborn as sns

    return (
        LinearRegression,
        mean_squared_error,
        mo,
        np,
        pd,
        pickle,
        plt,
        px,
        r2_score,
        shap,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # <center> The basics of Regression in Python

    The following notebook is part of a training done by Silva Bashllari, MSc through PyLadies Prishtina community on the basics of regression tasks through linear regression (simple and multiple), Ridge & Lasso Regressions.

    Expected Level of Participants: Beginners with same core skills in high-school level maths and basic Python skills.

    For the purposes of this work, we will use the following toy dataset:
        1. X-var-1: exam score on a professional test from 0 to 100.
        2. X-var-2: years of experience at work.
        3. Y-var: Salary.

    Our final goal is to <b> predict the salary.</b>

    We will have to make our decisions & experiments on what is the best approach towards that goal, meaning with which variables and models.

    The performance of this will be evaluated based on $R^2$ and $RMSE$ (Root Square Mean Error).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## <center> Step 1: Understanding, Cleaning and Preprocessing the dataset
    """)
    return


@app.cell
def _(pd):
    data = pd.read_excel("/home/siinvictus/projects/stupid_project/data/salary_data.xlsx")
    return (data,)


@app.cell
def _(data):
    data.head(10)
    return


@app.cell
def _(data):
    data.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Renaming the columns to make it easier to target them:
    """)
    return


@app.cell
def _(data):
    data.columns
    return


@app.cell
def _(data):
    data_rn = data.rename(columns={'#': 'index', 'Exam Score (0–100)': 'exam_score', 'Years of Experience': 'years_exp', 'Salary (€)': 'salary'})
    data_rn.head(10)
    return (data_rn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Clearly above we have 2 null (missing values) in the years of experience colum. We have to handle them somehow, let's first visualize them:
    """)
    return


@app.cell
def _(data_rn):
    data_rn[data_rn['years_exp'].isna()]
    return


@app.cell
def _(data_rn):
    data_rn.describe()
    return


@app.cell
def _(data_rn):
    print(f' The median for exam score is {data_rn['exam_score'].median()}, for years of experience is {data_rn['years_exp'].median()} and for salary {data_rn['salary'].median()} ')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two rows that have missing values have their exam scores 74.3 and 71.7, respectively, very close to the total exam score average that we see from <b> data.describe </b> of 70.7 and to the median. In terms of the salaries, they do differ from the average salary of 69,584 and from the median, one being 82,282 and the other 54,900.

    2 rows in a dataset of 120 rows are 2/120 = 0.016, so 1.6% of the dataset, so we can just make a heuristic decision to drop them.
    """)
    return


@app.cell
def _(data_rn):
    data_nonull = data_rn.dropna()
    data_nonull.shape
    return (data_nonull,)


@app.cell
def _(data_nonull, plt):
    plt.hist(data_nonull['years_exp'], color='red', edgecolor = 'black', bins = 15)
    plt.title('The histogram of Years of Experience')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We observe widely rangning years of experience, with two peaks let's say, one around 5 and the other around 12.5.
    """)
    return


@app.cell
def _(data_nonull, plt):
    plt.hist(data_nonull['exam_score'], color='blue', edgecolor = 'black', bins = 15)
    plt.title('The histogram of Exam Scores')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    W.r.t the exam scores, we can clearly see that lower scores tend to be more likely with the two equal peaks being around 50s and 60s.
    """)
    return


@app.cell
def _(data_nonull, plt):
    plt.hist(data_nonull['salary'], color='green', edgecolor = 'black', bins = 15)
    plt.title('The histogram of Salary')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The salary follows an almost normal distribution with a few peaks that go beyond what would be the normal curve but the good thing is that the center appears to also be the most frequent.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From the three plots above we can observe <b> there appear to be no outliners </b> to handle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In any of the variables, at least from a visual p.o.v we don't observe any completely of out the blue outliers that might harm the analysis or shift dramatically the regression lines.
    """)
    return


@app.cell
def _(data_nonull, plt):
    plt.scatter(data_nonull['years_exp'], data_nonull['salary'], color='red', marker='x')
    plt.title('The scatter plot of years of experience and salary')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()
    return


@app.cell
def _(data_nonull, plt):
    plt.scatter(data_nonull['exam_score'], data_nonull['salary'], color='blue', marker='x')
    plt.title('The scatter plot of years of exam scores and salary')
    plt.xlabel('Exam Scores')
    plt.ylabel('Salary')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From the scatter plots themselves, we can clearly observe that the first variable above, <b> the years of experience </b> builds a much cleaner pattern with the salary than does the exam scores. We can verify this further by computing the correlation of each variable with our target variable, the salary. The correlation metric we will use is the Pearson Correlation:

    $$
    r = \frac{\text{cov}(X, Y)}{\sigma_X \cdot \sigma_Y}
    $$

    where $\sigma_X$ is the standard deviation of variable $x$ and ${cov}(X,Y)$ is the covariance between two variables, $x$ and $y$ measured as follows:

    $$
    \text{cov}(X, Y) = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{n-1}
    $$
    """)
    return


@app.cell
def _(data_nonull):
    corr_matrix = data_nonull[['years_exp', 'exam_score', 'salary']].corr()
    return (corr_matrix,)


@app.cell
def _(corr_matrix):
    type(corr_matrix)
    return


@app.cell
def _(corr_matrix):
    corr_matrix
    return


@app.cell
def _(corr_matrix, sns):
    sns.heatmap(corr_matrix, 
                cmap='cubehelix',
                center=0,
                annot=True,
                #fmt = '.1g'  -- this makes it rounded to 1 decimal point
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From the correlation matrix above, we can clearly see that the intuition we got from the visualization is confirmed by the results of the correlation metric, where the years of experience and the salary have a high positive correlation, $corr = 0.85$ and the same one with exam scores shows a lesser but existent postive correlation of 0.31.

    The two input variables, exam score and years of experience have an almost equal to 0 correlation with one another which is good because this means they carry <b> independent information </b>. If they were to be correlated with one another, the model would struggle to differentiate the effect of one and the other in the salary. We can thus say that our model does not struggle from <b> multicollinearity </b>.
    """)
    return


@app.cell
def _(data_nonull, plt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data_nonull['exam_score'], data_nonull['years_exp'], data_nonull['salary'], color='green')

    ax.set_xlabel('Exam Score')
    ax.set_ylabel('Years of Experience')
    ax.set_zlabel('Salary')
    ax.set_title('Exam Score & Experience vs Salary')

    plt.show()
    return


@app.cell
def _(data_nonull, mo, px):
    figu = px.scatter_3d(
        data_nonull,
        x='exam_score',
        y='years_exp',
        z='salary',
        color='salary',
        title='Exam Score & Experience vs Salary'
    )

    mo.ui.plotly(figu)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## <center> Train-Test Split

    The way the pipeline usually goes it to create a test set (a standard rule of thumb taking 80%) of the data and the other (20%) we use to test the model, given that we know the ground truth and we can get an idea how well our model will perform in a real-world scenario.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## <center> Model 1: Simple Linear Regression

    Our first model will be a simple linear regression that when you think about it it's just the simple line equation:
    $ax + b = y$.

    In our case $x$ will be the years of experience in one case and the exam scores in the other case and as we observed from our previous analysis, we expect the model with the years of experience as an input to yield a better predicting power.

    Thus, the goal of our model, given that it will be derived from our dataset, is to find the correct "a" and "b" that best outputs the <b> line </b> that best describes the relationship between the two variables in play.

    Which can be <b> the best line </b> ?

    It could be that which minimizes the error, thus the distance of each point from the line. In this case, the <span style='color: steelblue'> mean squared distance </span> is taken into consideration in order to not consider the negative distances (if a point is above or below the line) and its mean to understand the average error across all points.

    The formula used for the standard Ordinary Least Squares to find the a and b is:

    $$
    a = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}, \quad b = \bar{y} - a\bar{x}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model 1_A:

    - x -> years of experience
    - y-> salary
    """)
    return


@app.cell
def _(data_nonull, train_test_split):
    X_1 = data_nonull[[ 'years_exp']]
    y_1 = data_nonull['salary']

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)
    return X_test1, X_train1, y_test1, y_train1


@app.cell
def _(X_train1):
    X_train1.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Why 94? Can you guess?
    """)
    return


@app.cell
def _(LinearRegression, X_train1, y_train1):
    model_a1 = LinearRegression()
    model_a1.fit(X_train1, y_train1)
    return (model_a1,)


@app.cell
def _(model_a1):
    print(f"\n--- Years of Experience Model ---")
    print(f"Coefficient (a): {model_a1.coef_[0]:.2f}")
    print(f"Intercept   (b): {model_a1.intercept_:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <b> Interpretation </b>: A one year increase in experience is associated with a 2,409.72€ increase in salary. If a candidate has zero years of experience, the model predicts a baseline salary of 46,624.57€.
    """)
    return


@app.cell
def _(X_test1, mean_squared_error, model_a1, np, r2_score, y_test1):
    y_pred1 = model_a1.predict(X_test1)

    r2_a1   = r2_score(y_test1, y_pred1)
    rmse_a1 = np.sqrt(mean_squared_error(y_test1, y_pred1))

    print(f"R²:   {r2_a1:.3f}")
    print(f"RMSE: {rmse_a1:.2f} €")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <b> Interpretation </b>: About 60% of the variation of the output variable salary is explained by the years of experience.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model 1_B:
    - x -> exam scores
    - y -> salary
    """)
    return


@app.cell
def _(data_nonull, train_test_split):
    X_1b = data_nonull[[ 'exam_score']]
    y_1b = data_nonull['salary']

    X_train1b, X_test1b, y_train1b, y_test1b = train_test_split(X_1b, y_1b, test_size=0.2, random_state=42)
    return X_test1b, X_train1b, y_test1b, y_train1b


@app.cell
def _(LinearRegression, X_train1b, y_train1b):
    model_b1 = LinearRegression()
    model_b1.fit(X_train1b, y_train1b)
    return (model_b1,)


@app.cell
def _(model_b1):
    print(f"\n--- Exam Scores Model ---")
    print(f"Coefficient (a): {model_b1.coef_[0]:.2f}")
    print(f"Intercept   (b): {model_b1.intercept_:.2f}")
    return


@app.cell
def _(X_test1b, mean_squared_error, model_b1, np, r2_score, y_test1b):
    y_pred1b = model_b1.predict(X_test1b)

    r2_b1   = r2_score(y_test1b, y_pred1b)
    rmse_b1 = np.sqrt(mean_squared_error(y_test1b, y_pred1b))

    print(f"R²:   {r2_b1:.3f}")
    print(f"RMSE: {rmse_b1:.2f} €")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <b> Interpretation </b>: We can observe above that the model tells us that increasing the exam score with 1 point leads to a 316 euros increase in salary. If the exam score is 0, the salary is expected to be 47,195 euros but given that we have no one with an exam score of 0 that particular parameter is not to be taken much into account. Furthermore, we can see that the $R^2$ is very small, thus only about 13.3% of the variation on the salary is explained by the exam scores an the RMSE is 12,619, way bigger compared to the first model of only 8479.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <center> Model 2: Multiple Linear Regression
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the simple linear regression models above, we used a single feature to predict salary.
    Multiple linear regression extends this by using **both features simultaneously**:

    $$
    \hat{y} = w_1 x_1 + w_2 x_2 + b
    $$

    where $x_1$ is the exam score, $x_2$ is the years of experience, $w_1$ and $w_2$ are their
    respective coefficients, and $b$ is the intercept.

    The intuition is the same as before — find the $w_1$, $w_2$, and $b$ that minimise the
    mean squared error across all training points — but now the model fits a **plane** in 3D
    space rather than a line in 2D. This is exactly what you visualised in the 3D scatter plot above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##
    """)
    return


@app.cell
def _(data_nonull, train_test_split):
    X = data_nonull[[ 'years_exp', 'exam_score']]
    y = data_nonull['salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, X_test, X_train, y, y_test, y_train


@app.cell
def _(LinearRegression, X_train, y_train):
    model_multi = LinearRegression()
    model_multi.fit(X_train, y_train)
    return (model_multi,)


@app.cell
def _(X_test, mean_squared_error, model_multi, np, r2_score, y_test):
    y_pred_multi = model_multi.predict(X_test)

    r2_multi   = r2_score(y_test, y_pred_multi)
    rmse_multi = np.sqrt(mean_squared_error(y_test, y_pred_multi))
    return r2_multi, rmse_multi, y_pred_multi


@app.cell
def _(y_pred_multi):
    y_pred_multi
    return


@app.cell
def _(y_test):
    y_test
    return


@app.cell
def _(model_multi, r2_multi, rmse_multi):
    print(f"Coefficients:")
    print(f"  exam_score  (w1): {model_multi.coef_[0]:.2f}")
    print(f"  years_exp   (w2): {model_multi.coef_[1]:.2f}")
    print(f"  intercept   (b):  {model_multi.intercept_:.2f}")

    print(f"Evaluation metrics:")
    print(f"\nR²:   {r2_multi:.3f}")
    print(f"RMSE: {rmse_multi:.2f} €")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Clearly, the multiple regression model does a much better job than the simple ones to predict the salary, by simoultaneously increasing the $R^2$ score to aboyt 93%, thus indicating that 93% of the variation in the salary is explained by the years of experience and the exam scores. At the same time, this second model lowers the error by more than half compared to the best performin model above.

    The train/test split is only for evaluation purposes. We hold out the test set to get an honest estimate of how well the model generalises. But once we're satisfied with the model, we retrain on the full dataset before saving, so the final model has learned from as much data as possible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <center> The final model
    """)
    return


@app.cell
def _(LinearRegression, X, y):
    model_final = LinearRegression()
    model_final.fit(X, y)  # X and y are the full dataset, not just train
    return (model_final,)


@app.cell
def _(model_final):
    print(model_final.coef_, model_final.intercept_)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The formula then is the following:

    $$
    \hat{salary} = 12303.14 + 2600.94 * yearsExperience + 454.22 * examScore
    $$

    This means that, keeping everything else static:
    - 1 more year of experience increases the salary by 2600.94 euros;
    - 1 more point in the exam score increases the salary by 454.22 euros;
    - if the years of experience and the exam score is 0, the expected salary is 12303.14 euros.
    """)
    return


@app.cell
def _(X, model_final):
    y_pred_final = model_final.predict(X)
    return (y_pred_final,)


@app.cell
def _(plt, y, y_pred_final):
    plt.scatter(y, y_pred_final, color='purple', alpha=0.6)

    # perfect prediction line (y = y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--')


    plt.title('Actual vs Predicted Salary')
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.show()
    return


@app.cell
def _(model_final, pickle):
    with open("../models/mult_lin_reg.pkl", "wb") as f:
        pickle.dump(model_final, f)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example Prediction:

    Imagine in our webpage someone can put their current years of experience and their performance on a given acceptance test in a specific company and we can thus predict their expected salary.

    Remember the order of inputs: 1. Years of Experience 2. Exam Scores.
    """)
    return


@app.cell
def _(model_final):
    model_final.predict([[10, 60.7]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model predicts a salary of 65,884.07 euros/year.
    However, if the person studies a bit harder and manages to increase the exam score by 10 points:
    """)
    return


@app.cell
def _(model_final):
    model_final.predict([[10, 67.7]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    He/she might see a slight improvement in the salary, with it reaching 69.063.65 euros/year.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## <center> Adding Interpretability through SHAP

    SHAP stands for SHapley Additive exPlanations. <br>

    It combines two ideas: Shapley values from game theory and the fact that the contributions are additive — they sum up exactly to the prediction.

    It can be very helpful in dealing with "black box" models, like neural nets, when we don't know exactly how every input variable has impacted the output.

    However, it is easier to learn it with simple models like the ones above.

    Remember, from the multiple linear regression, we have the following formula:

    $$
    salary=12303.14+2600.94∗yearsExperience+454.22∗examScore
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### First, let's do it step-by-step "by-hand" (yes, irony is recognized) 🐼
    """)
    return


@app.cell
def _(data_nonull):
    data_nonull.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have:
    - mean salary: 69,601.44
    - mean exam_score: 70.72
    - mean years_exp: 9.76
    """)
    return


@app.cell
def _(data_nonull):
    mean_exam_score = data_nonull['exam_score'].mean()
    mean_years_exp = data_nonull['years_exp'].mean()
    print(f'The average exam score is {mean_exam_score} and the average years of experience are {mean_years_exp}')
    return mean_exam_score, mean_years_exp


@app.cell
def _(mean_exam_score, mean_years_exp):
    # We compute the expected value of the salary from our model, that is the mean value
    expected_salary_from_mr = 12303.14 + 2600.94*mean_years_exp + 454.22*mean_exam_score
    print(f'The average expected salary is {expected_salary_from_mr}')
    return (expected_salary_from_mr,)


@app.cell
def _(mean_exam_score, mean_years_exp):
    2600.94525765*mean_years_exp +  454.22532652 * mean_exam_score + 12303.147145739575
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's consider we have a specfic person with:
    - Years of experience: 10
    - Exam score: 57

    We then use our model to predict the salary.
    """)
    return


@app.cell
def _(model_final):
    model_final.predict([[10, 57]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Clearly we can see the predicted salary is: 64203.44
    """)
    return


@app.cell
def _(model_final):
    pred_sal_person1 = model_final.predict([[10, 57]])
    pred_sal_1 = pred_sal_person1[0]
    return (pred_sal_1,)


@app.cell
def _(expected_salary_from_mr, pred_sal_1):
    difference_person1 = pred_sal_1 - expected_salary_from_mr
    print(f'This person gets {difference_person1:.2f} than the average prediction.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we want to understand which of the variables, namely years of experience and salary, contributed by how much in "losing out" or "gaining" that extra money.

    For this we have to compute <b> the SHAP value for each feature</b>.

    $$
    SHAP_i = coef_i(x_i - \bar{x_i})
    $$
    """)
    return


@app.cell
def _():
    shap_experience = 2600.94 * (10-9.76)
    shap_exams = 454.22 * (57-70.72)
    print(f'Person 1 got {shap_experience:.2f} euros from the years of experience and {shap_exams:.2f} euros from the exam scores.')
    return shap_exams, shap_experience


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Verifiying it with the baseline:
    """)
    return


@app.cell
def _(expected_salary_from_mr, shap_exams, shap_experience):
    expected_salary_from_mr +shap_experience+shap_exams  #compared to predicted: 64203.44 -- some 10 euros missing due to approximation errors? but the point is clear.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Now, using the python library <b>shap
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1. Computing the SHAP values
    """)
    return


@app.cell
def _(X, model_final, shap):
    masker = shap.maskers.Independent(X, max_samples=118)
    explainer = shap.LinearExplainer(model_final, masker)
    shap_values = explainer(X)

    print("masker mean:", explainer.masker.data.mean(axis=0))
    print("base_values:", shap_values[0].base_values)
    return explainer, shap_values


@app.cell
def _(shap_values):
    print(f'The type of shap values {type(shap_values)} and internally {type(shap_values.values)}')
    return


@app.cell
def _(np, shap_values):
    print(np.shape(shap_values.values)) #so each row has 2 shap values, 1 for each feature.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 2. Waterfall & bar plots.
    """)
    return


@app.cell
def _(shap, shap_values):
    # explain one specific person
    shap.plots.waterfall(shap_values[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can clearly see that for the first person taken into consideration here, we observe:
    - The expected salary that is the mean salary (same as we computed "by hand" above): 69,601.44 Euros
    - This person's predicted salary: <b> 83,793.857 </b> Euros
    - And how much each of the two features contributes to that difference from the mean salary:
          - the fact that the years of experience are 16.1, it adds 16,703.36 euros to the salary.
          - the fact that the exam score 65.2, lowers the salary by 2,510.94 euros.
    """)
    return


@app.cell
def _(X, expected_salary_from_mr, model_final, shap_values):
    print("By-hand baseline:", expected_salary_from_mr)
    print("Mean of model predictions over X:", model_final.predict(X).mean())
    print("SHAP base_values:", shap_values[0].base_values)
    return


@app.cell
def _(X, explainer):
    print("X shape:", X.shape)
    print("X mean:\n", X.mean())
    print("masker mean:\n", explainer.masker.data.mean(axis=0) if hasattr(explainer.masker, 'data') else "no masker.data")
    return


@app.cell
def _(shap, shap_values):
    shap.plots.force(shap_values[0], matplotlib=True) #it required matplotlib=True otherwise doesn't allow it.
    return


@app.cell
def _(shap, shap_values):
    shap.plots.bar(shap_values)
    #shows absolute mean shap values
    # because for each observation, that is for each person, there will be a shap value for each feature,
    # the bar graph below gives the abs mean over all obs.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How to compute the mean(|shap value|) above and how to interpret the results?

    For each feature, we have:

    $$
     \text{importance}_j = \frac{1}{n} \sum_{i=1}^{n} (|\text{SHAP}_{i,j}|)
    $$

    So, the importance of feature j is measured as the average of all the SHAP values for that feature for every person.

    In our case:
    - The years of experience changes the salary on average by $+- 13114.41$ euros
    - The exam score changes the salary on average by $+-6513.73$ euros
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3. Beeswarm Plot
    """)
    return


@app.cell
def _(shap, shap_values):
    # global summary
    # we can see which have large pos or large negative values
    # for both of the features, as the feature values increase, the shap values increase
    shap.plots.beeswarm(shap_values)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4. Dependence Plots

    Quite useful if the features have a nonlinear relationship with the target value.
    """)
    return


@app.cell
def _(shap, shap_values):
    shap.plots.scatter(shap_values)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## <center> Regularization Techniques
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ridge Regression
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lasso Regression
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Elastic Net
    """)
    return


if __name__ == "__main__":
    app.run()
