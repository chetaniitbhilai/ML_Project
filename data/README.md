# Data Directory Structure

The **data** folder in this project organizes all datasets used for processing, model training, and testing. This directory ensures modularity and clear categorization of raw, refined, and auxiliary data.

---

## Subdirectories

### **1. dataset/**
Contains the original raw datasets used as input for preprocessing and analysis created using overpy library of Python.

**Overpy** - The overpy library in Python is used to interact with the OpenStreetMap (OSM) Overpass API, which allows users to query and retrieve geographic data from OSM. With overpy, you can send complex queries to Overpass API to extract data like nodes, ways, and relations, which represent various map elements such as points of interest, streets, and areas. It provides a straightforward interface for parsing and accessing these elements' details, including tags, geometry, and other metadata. This library is useful for applications involving geographic data analysis, mapping, and geospatial research.

- **Purpose**: Acts as the primary source for all data processing.
- **Example Content**: Raw CSV files, or any other format containing unprocessed metro line or amenities data.

---

### **2. Feature_addition/**
Holds additional features or data sources that augment the base datasets using opencage API.

**OpenCage API** - The OpenCage API is a geocoding service that provides powerful capabilities for converting geographic coordinates (latitude and longitude) into human-readable addresses (reverse geocoding) and vice versa (forward geocoding). It aggregates data from multiple sources, including OpenStreetMap, making it highly reliable and accurate across different regions. The API is often used for applications that require location-based services, such as mapping, location search, address validation, and more. It offers flexible query options, supports international addresses, and handles ambiguous or incomplete queries gracefully, making it a popular choice for developers working with geospatial data.

- **Purpose**: To enhance the predictive capability of the model by adding supplementary attributes (e.g., proximity, location-specific details).
- **Example Content**: 
  - Datasets for additional features like geographic markers.

---

### **3. population/**
Contains datasets related to population that we made using a formula based on the suburb_count,road_count and district_count.

- **Purpose**: Helps incorporate demographic factors into the prediction model.
- **Example Content**: 
  - Population density or metro usage trends by population segment.

---

### **4. refined_dataset/**
Includes preprocessed and cleaned datasets.

- **Purpose**: Optimized datasets for machine learning training and evaluation.
- **Example Content**: 
  - Cleaned amenities data.
  - Merged and formatted datasets ready for use in `data_train.py`.

---

### **5. test_data/**
Stores data specifically set aside for testing model performance made with refernce to Delhi as a center.

- **Purpose**: Ensures unbiased evaluation of the model's predictions.
- **Example Content**: 
  - Validation and test split datasets.
  - Sample input data for test cases.

---

## How to Use the Data Folder

1. **Accessing Raw Data**  
   Use the `dataset/` folder to source original data for any processing needs.

2. **Adding Features**  
   Supplement your base dataset with data from the `Feature_addition/` directory.

3. **Refining Data**  
   Refine datasets and store them in the `refined_dataset/` folder for future use in training.

4. **Testing the Model**  
   Evaluate model performance using the datasets in the `test_data/` folder.

---

## Notes

- Ensure any new data follows the folder structure for consistency.
- Preprocessing scripts should output refined data to the `refined_dataset/` folder.
- All data used for testing should be exclusive to the `test_data/` directory to prevent data leakage.

---

## License

This project is licensed under the MIT License. Refer to the root-level `README.md` for details.