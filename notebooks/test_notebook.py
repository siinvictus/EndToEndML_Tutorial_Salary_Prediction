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
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    import plotly.express as px
    import pickle
    import shap

    return (
        ElasticNet,
        Lasso,
        LinearRegression,
        Ridge,
        StandardScaler,
        mean_squared_error,
        mo,
        np,
        pd,
        pickle,
        plt,
        px,
        r2_score,
        shap,
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
    data = pd.read_excel("data/salary_data.xlsx")
    return (data,)


@app.cell
def _(data):
    data.head()
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
    data_nonull[['years_exp', 'exam_score', 'salary']].corr()
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


@app.cell
def _(X, model_final, shap):
    masker = shap.maskers.Independent(X, max_samples=118)
    explainer = shap.LinearExplainer(model_final, masker)
    shap_values = explainer(X)

    print("masker mean:", explainer.masker.data.mean(axis=0))
    print("base_values:", shap_values[0].base_values)
    return explainer, shap_values


@app.cell
def _(shap, shap_values):
    # explain one specific person
    shap.plots.waterfall(shap_values[0])
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
    # global summary
    shap.plots.beeswarm(shap_values)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## <center> Regularization Techniques
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regularization Techniques

    Until now, our linear regression model had only one main objective: find the coefficients that minimize the prediction error.

    In multiple linear regression, our salary prediction followed the form:

    \[
    \hat{salary} = b + w_1 \cdot years\_exp + w_2 \cdot exam\_score
    \]

    The coefficients \(w_1\) and \(w_2\) tell us how much each feature contributes to the final salary prediction.

    However, ordinary linear regression gives the model full freedom to choose these coefficients. This is not always a problem, especially in a small and clean dataset like ours. But in real datasets, we may have many features, noisy data, duplicated information, or strongly correlated variables. In those cases, the model may start relying too much on certain variables and produce coefficients that are too large or unstable.

    Regularization is a way of saying:

    > "Fit the data, but do not let the coefficients become unnecessarily large."

    So instead of minimizing only the prediction error, regularized models minimize:

    \[
    prediction\ error + penalty
    \]

    The penalty is applied to the coefficients. This means the model is rewarded for predicting well, but punished if it uses overly large coefficients to do so.

    This is useful because smaller and more controlled coefficients often lead to models that generalize better to unseen data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why do we scale before regularization?

    Regularization penalizes the size of coefficients. Because of that, the scale of the input variables matters.

    For example, `exam_score` ranges from 0 to 100, while `years_exp` is much smaller in range. If we apply regularization directly, the model may punish one coefficient more than another simply because the features are measured in different units.

    To avoid this, we standardize the input features so that each feature has:

    \[
    mean = 0,\quad standard\ deviation = 1
    \]

    This allows Ridge, Lasso, and Elastic Net to treat the features more fairly.
    """)
    return


@app.cell
def _(StandardScaler, X, X_test, X_train):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_scaled = scaler.fit_transform(X)
    return X_test_scaled, X_train_scaled


@app.cell
def _(mean_squared_error, np, r2_score):
    def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\n--- {model_name} ---")
        print(f"R²:   {r2:.3f}")
        print(f"RMSE: {rmse:.2f} €")

        return r2, rmse, model

    return (evaluate_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ridge Regression

    Ridge Regression is the first regularized version of linear regression that we will test.

    Ordinary Linear Regression minimizes:

    \[
    RSS = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
    \]

    Ridge Regression adds a penalty to this:

    \[
    RSS + \alpha \sum_{j=1}^{p} w_j^2
    \]

    The second part is the regularization penalty. It squares each coefficient and adds them together.

    The hyperparameter \(\alpha\) controls how strong the penalty is:

    - If \(\alpha = 0\), Ridge behaves like ordinary linear regression.
    - If \(\alpha\) is small, the model is only lightly regularized.
    - If \(\alpha\) is large, the model is strongly discouraged from using large coefficients.

    Ridge does not usually make coefficients exactly zero. Instead, it shrinks them. This makes Ridge useful when we believe most features are useful, but we still want to control the model's flexibility.
    """)
    return


@app.cell
def _(
    Ridge,
    X,
    X_test_scaled,
    X_train_scaled,
    evaluate_model,
    y_test,
    y_train,
):
    ridge_model = Ridge(alpha=1.0)

    r2_ridge, rmse_ridge, ridge_model = evaluate_model(
        ridge_model,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        "Ridge Regression"
    )

    print("Ridge coefficients:")
    for feature, coef in zip(X.columns, ridge_model.coef_):
        print(f"{feature}: {coef:.2f}")

    print(f"Intercept: {ridge_model.intercept_:.2f}")
    return r2_ridge, rmse_ridge


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    The Ridge coefficients are based on scaled features, so they should not be interpreted in the exact same euro-per-unit way as the original linear regression coefficients.

    Instead, the main thing we observe is how Ridge controls the coefficients compared to ordinary linear regression. Ridge keeps both features in the model, but applies pressure on the coefficients so they do not become unnecessarily large.

    In our case, since the dataset has only two features and they are not strongly correlated, we do not expect Ridge to dramatically outperform ordinary linear regression. The value of Ridge becomes more visible when the dataset has many features or multicollinearity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lasso Regression
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lasso Regression

    Lasso Regression is another regularized version of linear regression.

    Its objective is:

    \[
    RSS + \alpha \sum_{j=1}^{p} |w_j|
    \]

    The difference is that Lasso uses the absolute value of the coefficients instead of the squared value.

    This small mathematical change creates an important behavior: Lasso can shrink some coefficients all the way to zero.

    When a coefficient becomes zero, the feature is effectively removed from the model.

    So Lasso is useful not only for controlling overfitting, but also for feature selection.
    """)
    return


@app.cell
def _(
    Lasso,
    X,
    X_test_scaled,
    X_train_scaled,
    evaluate_model,
    y_test,
    y_train,
):
    lasso_model = Lasso(alpha=0.1, max_iter=10000)

    r2_lasso, rmse_lasso, lasso_model = evaluate_model(
        lasso_model,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        "Lasso Regression"
    )

    print("Lasso coefficients:")
    for feat, co in zip(X.columns, lasso_model.coef_):
        print(f"{feat}: {co:.2f}")

    print(f"Intercept: {lasso_model.intercept_:.2f}")
    return r2_lasso, rmse_lasso


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    Lasso is stricter than Ridge because it can force weak coefficients to become exactly zero.

    In this dataset, we only have two predictors: years of experience and exam score. Since both variables have some relationship with salary, Lasso may keep both of them depending on the value of \(\alpha\).

    However, if we increase \(\alpha\), Lasso becomes more aggressive. This allows us to observe which feature the model considers more essential for predicting salary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How does alpha change the Lasso model?

    Instead of using only one value of \(\alpha\), we can test several values and observe how the coefficients change.

    This helps us understand Lasso as a coefficient selection mechanism rather than just another regression model.
    """)
    return


@app.cell
def _(
    Lasso,
    X_test_scaled,
    X_train_scaled,
    mean_squared_error,
    np,
    pd,
    r2_score,
    y_test,
    y_train,
):
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]

    lasso_results = []

    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        lasso_results.append({
            "alpha": alpha,
            "r2": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "years_exp_coef": model.coef_[0],
            "exam_score_coef": model.coef_[1]
        })

    lasso_results_df = pd.DataFrame(lasso_results)
    lasso_results_df
    return alphas, lasso_results_df, model, y_pred


@app.cell
def _(lasso_results_df, plt):
    plt.figure(figsize=(8, 5))
    plt.plot(lasso_results_df["alpha"], lasso_results_df["years_exp_coef"], marker="o", label="years_exp")
    plt.plot(lasso_results_df["alpha"], lasso_results_df["exam_score_coef"], marker="o", label="exam_score")
    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Coefficient value")
    plt.title("Lasso coefficient shrinkage as alpha increases")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From the plot above, we can see how increasing \(\alpha\) puts more pressure on the coefficients.

    At low values of \(\alpha\), Lasso behaves similarly to ordinary linear regression. As \(\alpha\) increases, the coefficients shrink. If \(\alpha\) becomes large enough, one or more coefficients may become exactly zero.

    This is the main conceptual difference between Ridge and Lasso: Ridge shrinks coefficients smoothly, while Lasso can remove features from the model completely.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How does alpha change the Ridge model?

    We repeat the same experiment with Ridge. The expectation is different: Ridge should shrink coefficients as \(\alpha\) increases, but it should not usually force them exactly to zero.
    """)
    return


@app.cell
def _(
    Ridge,
    X_test_scaled,
    X_train_scaled,
    alphas,
    mean_squared_error,
    model,
    np,
    pd,
    r2_score,
    y_pred,
    y_test,
    y_train,
):
    ridge_results = []

    for ralpha in alphas:
        rmodel = Ridge(alpha=ralpha)
        rmodel.fit(X_train_scaled, y_train)
        y_pred_r = model.predict(X_test_scaled)

        ridge_results.append({
            "alpha": ralpha,
            "r2": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "years_exp_coef": model.coef_[0],
            "exam_score_coef": model.coef_[1]
        })

    ridge_results_df = pd.DataFrame(ridge_results)
    ridge_results_df
    return (ridge_results_df,)


@app.cell
def _(plt, ridge_results_df):
    plt.figure(figsize=(8, 5))
    plt.plot(ridge_results_df["alpha"], ridge_results_df["years_exp_coef"], marker="o", label="years_exp")
    plt.plot(ridge_results_df["alpha"], ridge_results_df["exam_score_coef"], marker="o", label="exam_score")
    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Coefficient value")
    plt.title("Ridge coefficient shrinkage as alpha increases")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Ridge coefficients also shrink as \(\alpha\) increases. However, unlike Lasso, Ridge does not usually set coefficients exactly to zero.

    This means Ridge keeps both variables in the model, but reduces their strength. In other words, Ridge controls the volume of each feature rather than muting features completely.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Elastic Net
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Elastic Net

    Elastic Net combines Ridge and Lasso.

    Its objective is:

    \[
    RSS + \alpha \left( l1\_ratio \sum |w_j| + (1 - l1\_ratio)\sum w_j^2 \right)
    \]

    This means Elastic Net has two controls:

    - \(\alpha\): how strong the total regularization is
    - \(l1\_ratio\): how much of the penalty behaves like Lasso versus Ridge

    If:

    \[
    l1\_ratio = 1
    \]

    Elastic Net behaves like Lasso.

    If:

    \[
    l1\_ratio = 0
    \]

    Elastic Net behaves like Ridge.

    If the value is between 0 and 1, the model combines both behaviors.

    Elastic Net is useful when we want feature selection, but we also want more stability when features are correlated.
    """)
    return


@app.cell
def _(
    ElasticNet,
    X,
    X_test_scaled,
    X_train_scaled,
    evaluate_model,
    y_test,
    y_train,
):
    elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)

    r2_elastic, rmse_elastic, elastic_model = evaluate_model(
        elastic_model,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        "Elastic Net Regression"
    )

    print("Elastic Net coefficients:")
    for efeature, ecoef in zip(X.columns, elastic_model.coef_):
        print(f"{efeature}: {ecoef:.2f}")

    print(f"Intercept: {elastic_model.intercept_:.2f}")
    return r2_elastic, rmse_elastic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    Elastic Net behaves as a compromise between Ridge and Lasso.

    The Lasso part allows it to reduce unnecessary features, while the Ridge part makes the model more stable. This is especially helpful when we have many predictors and some of them are correlated with each other.

    In our dataset, Elastic Net may not dramatically improve performance because the dataset is small and has only two input variables. However, it is important conceptually because it shows how regularization techniques can be combined depending on the modeling problem.
    """)
    return


@app.cell
def _(
    pd,
    r2_elastic,
    r2_lasso,
    r2_multi,
    r2_ridge,
    rmse_elastic,
    rmse_lasso,
    rmse_multi,
    rmse_ridge,
):
    comparison = pd.DataFrame({
        "Model": [
            "Multiple Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Elastic Net"
        ],
        "R²": [
            r2_multi,
            r2_ridge,
            r2_lasso,
            r2_elastic
        ],
        "RMSE": [
            rmse_multi,
            rmse_ridge,
            rmse_lasso,
            rmse_elastic
        ]
    })

    comparison
    return (comparison,)


@app.cell
def _(comparison, plt):
    plt.figure(figsize=(8, 5))
    plt.bar(comparison["Model"], comparison["RMSE"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("RMSE (€)")
    plt.title("Model comparison by RMSE")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
