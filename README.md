# ML_Project
Team Name - Outliers <br>
<br>
Branch - Master <br>
<br>
📦 Root<br>
 ┣ 📂 UI<br>
 ┃ ┣ 📄 amenities_generic.csv ────────────────── Raw amenities data<br>
 ┃ ┣ 📄 amenities_generic_filtered.csv ───────── Filtered version of amenities data<br>
 ┃ ┣ 📄 app.py ─────────────────────────────────┐<br>
 ┃ ┣ 📄 data_train.py ───────────────────────────┼── UI integration files with model<br>
 ┃ ┣ 📄 model.py ───────────────────────────────┘<br>
 ┃ ┣ 📄 index.html ────────────────────────────── Web interface<br>
 ┃ ┣ 📄 final_combined_amenities.csv ─────────── Final processed amenities data<br>
 ┃ ┣ 📄 output_graph.png ──────────────────────── Visualization output<br>
 ┃ ┣ 📄 updated_amenities_generic.csv ────────── Updated version of amenities data<br>
 ┃ ┗ 📄 updated_data.csv ─────────────────────── Latest data update<br>
 ┃<br>
 ┣ 📂 data<br>
 ┃ ┣ 📂 Feature_addition ─────────────────────── Addition of features and new generic model<br>
 ┃ ┣ 📂 dataset ───────────────────────────────── Modifying DatasetCreation python script<br>
 ┃ ┣ 📂 datset_station ──────────────────────── Add files via upload<br>
 ┃ ┣ 📂 population ────────────────────────────── Population data added<br>
 ┃ ┣ 📂 refined_dataset ─────────────────────── Updated refining code for duplicates<br>
 ┃ ┗ 📂 test_data ─────────────────────────────── Refined directory structure<br>
 ┃<br>
 ┣ 📂 models<br>
 ┃ ┣ 📂 clustering/City_plots ────────────────── Initial clustering approach<br>
 ┃ ┣ 📂 model_test ───────────────────────────── Individual team member models<br>
 ┃ ┗ 📂 final_model ─────────────────── Final generic model implementation<br>
 ┃<br>
 ┗ 📂 utilities<br>
 ┃ ┣ 📂 dummy_city ───────────────────────────── Test implementation with dummy city data<br>
 ┃ ┗ 📂 rohan_new ─────────────────────────────── Real map plotting implementation<br>
 ┃<br>
 ┗ 📄 README.md - Contents of the directory and how to run the model <br>
<br>
# Running the UI Version of the Model

To set up and run the UI and backend components for the project, follow these steps:

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/chetaniitbhilai/ML_Project.git
cd UI
```

### Step 2: Launch the UI
**Open index.html in your preferred browser.<br>
You may use a local development server (e.g., Live Server in VS Code) to go live.<br>
UI Preview:<br>**

![UI Screenshot](https://github.com/user-attachments/assets/1021d28c-37ae-4eb5-8d71-e9c6ceef7e82) <br>


### Step 3: Run the Backend
```bash
In UI directory 
python/python3 app.py
```

### Step 4: 
<ul>
  <li>Input: Enter coordinates and radius values in the frontend form.</li>
  <li>Action: Click on the "Send to Backend" button.</li>
  <li>Process: The backend will process the request. Please allow approximately 10 minutes for the operation to complete.</li>
</ul>


