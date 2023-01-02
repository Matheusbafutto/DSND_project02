# Disaster Response Pipeline Project

This is an implementation for the capstone project of Udacity's Data Science Nanodegree **Data Engineering** module. It implements an ETL and a machine learning pipeline to classify tweets based on 36 disaster categories.

It also builds a web app capable of displaying insights about the dataset. The web app allows users to input "tweets" of their own and visualize predicted disaster categories for the text they provided.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### file structure
- app
    - templates
        - go.html: template for custom text predictions page
        - master.html: template for homepage with dataset visualizations
    - run.py: main execution script for web app bootstrap
- data
    - disaster_categories.csv: dataset with disaster categories for each tweet
    - disaster_categories.csv: dataset with input tweet messages
    - DisasterResponse.db: sqlite database with merged and cleaned master dataset (stored in `tweets` table)
- jupyter_notebooks
    - ETL Pipeline Preparation.ipynb: jupyter notebook used to draft ETL script
    - ML Pipeline Preparation.ipynb: jupyter notebook used to draft ML script
- models: folder containing various output ML models from the ML pipeline
- process_data.py: script for creating sqlite databse file used in the ML pipeline
- train_classifier.py: script for training best ML model from ML notebook on sqlite dataset and output a new model

Remaining files are either configuration or documentation

# Reference
- https://scikit-learn.org/stable/index.html
- https://pandas.pydata.org/docs/index.html
- https://numpy.org/doc/
- Udacity Data Scientist Nanodegree coursework
- Udacity Knowledge portal
- https://plotly.com/javascript/
- https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
- https://stackoverflow.com/questions/40524790/valueerror-this-solver-needs-samples-of-at-least-2-classes-in-the-data-but-the
- https://stackoverflow.com/questions/71902957/how-to-use-gridsearchcv-with-multioutputclassifiermlpclassifier-pipeline
- https://towardsdatascience.com/understanding-the-n-jobs-parameter-to-speedup-scikit-learn-classification-26e3d1220c28
- macworld.com/article/194771/105audioterm.html
- https://stackoverflow.com/questions/61893719/importerror-cannot-import-name-joblib-from-sklearn-externals
- https://www.programiz.com/python-programming/datetime/current-time
- https://www.geeksforgeeks.org/python-how-to-get-function-name/#:~:text=Method%201%3A%20Get%20Function%20Name,also%20for%20documentation%20at%20times.
- https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
