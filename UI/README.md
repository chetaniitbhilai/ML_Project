# Project: Metro Line Prediction and Data Analysis

This project involves machine learning and data processing to predict metro line routes based on given data. The repository includes scripts for training models, processing data, and rendering visualizations on a web interface.

---

## Directory Structure

### **Files and Directories**

- **cache/**  
  A directory containing intermediate or cached files to plot the metro stations on the map. It contains the JSON files for the coordinates plotting 

- **amenities_generic.csv**  
  The raw dataset of amenities used for analysis generated from the overpy library. (Code for this is available in data_train.py)

- **amenities_generic_filtered.csv**  
  A filtered version of the amenities dataset removing the unknown values. (Code for this is available in data_train.py)

- **final_combined_amenities.csv**  
  The final processed dataset combining various amenities data for model training that contains the count of the road, suburb and district which are just some additional features we are using for calculating the population our primary feature.

- **updated_amenities_generic.csv**  
  An updated version of the amenities data which contains the suburb, road and district which we are genreating using opencage API.

- **updated_data.csv**  
  An updated dataset and our final dataset that we are using for model training which also contains the population that is our primary feature.

- **app.py**  
  The main Python script to run the application, using Flask backend that create a pipeline to run the data_train and model.

- **data_train.py**  
  Script for training the machine learning model and refining it and adding the features.

- **model.py**  
  Contains the machine learning model implementation of GCN that predicts the metro stations and it also makes a graph and plot that on the real map of the required city.

- **index.html**  
  The main HTML file for the web-based visualization interface.

- **predicted_metro_line_map.html**  
  A generated HTML file showing the predicted metro line map using openwather API.

- **README.md**  
  This documentation file.

---
