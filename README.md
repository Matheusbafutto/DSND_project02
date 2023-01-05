# Disaster Response Pipeline Project

This is an implementation for the capstone project of Udacity's Data Science Nanodegree **Data Engineering** module. It implements an ETL and a machine learning pipeline to classify tweets based on 36 disaster categories.

It also builds a web app capable of displaying insights about the dataset. The web app allows users to input "tweets" of their own and visualize predicted disaster categories for the text they provided.

## Instructions:

### With docker
In order to get the final webapp to run as expected, we would need to setup conda and also call the processing and training scripts with matching path arguments for csv's, models and sql files. To make bootstrapping the app easy, a Dockerfile has been added so that all script calls and arguments are handled behind the scenes. The image also contains a conda installation so we dont need to setup conda on the local machine either. To run the app with docker:

1. Make sure you have (docker setup on your environment)[https://docs.docker.com/get-docker/]
2. Build the image by running `docker build -t dsnd_project02 .` at the root of the project
3. Start a new container from the image with `docker run -p 3001:3001 --name DSND_pjct02_api dsnd_project02`
4. navigate on your local browser to `http://localhost:3001`

**Disclaimer**: the webapp on the final container loads pickled ML models which are pretty big. I plan on optimizing the size of the model used on this app in the future but in the meantime, the container may require additional RAM to work properly (my experience, 10 GB worked fine).

### Without docker
This project assumes you are using (conda)[https://docs.conda.io/en/latest/] as your python managing library. Before you begin make sure you have conda setup and your conda environment contains all dependencies specified in `env.yml`. You can also create a new conda environment with all required dependencies by running `conda env create -f env.yml`.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## file structure
- app
    - templates
        - go.html: template for custom text predictions page
        - master.html: template for homepage with dataset visualizations
    - run.py: main execution script for web app bootstrap
- data
    - disaster_categories.csv: dataset with disaster categories for each tweet
    - disaster_categories.csv: dataset with input tweet messages
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
- https://pythonspeed.com/articles/activate-conda-dockerfile/
- https://docs.docker.com/
- https://hub.docker.com/r/continuumio/miniconda3
- https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
- https://stackoverflow.com/questions/38882654/docker-entrypoint-running-bash-script-gets-permission-denied
- https://stackoverflow.com/questions/42494853/standard-init-linux-go178-exec-user-process-caused-exec-format-error
- https://stackoverflow.com/questions/66785929/docker-container-exited-with-code-247-when-getting-data-from-gcp
- https://forums.docker.com/t/dockerfile-getting-failed-to-compute-cache-key-in-the-build/125911/14
