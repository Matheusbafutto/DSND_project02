# base image to setup conda
FROM continuumio/miniconda3:latest as builder

COPY env.yml .

RUN conda env create -f env.yml

RUN conda env list

RUN echo "conda activate DSND_project02" > ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN echo "Testing conda environment activation"
RUN python -c "import flask"
RUN python -c "import pandas"
RUN python -c "import sqlalchemy"
RUN python -c "import plotly"

# image for preprocessing data
FROM builder as cleaner

COPY ./data ./data
COPY ./process_data.py ./process_data.py


RUN python process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv ./data/DisasterResponse.db
RUN ls
# try running ls to see if sql file is generated

# image for training model
FROM builder as trainer

COPY --from=cleaner ./data/DisasterResponse.db ./data/DisasterResponse.db
COPY ./train_classifier.py ./train_classifier.py

RUN python train_classifier.py ./data/DisasterResponse.db ./vec_tfidf_RandomForest.pkl

# image for serving webapp
FROM builder as app

COPY ./app ./app
COPY --from=trainer ./vec_tfidf_RandomForest.pkl ./models/vec_tfidf_RandomForest.pkl
COPY ./bootstrap.sh ./bootstrap.sh
COPY --from=cleaner ./data/DisasterResponse.db ./data/DisasterResponse.db

RUN chmod +x "/bootstrap.sh"
CMD ["./bootstrap.sh"]

EXPOSE 3001
