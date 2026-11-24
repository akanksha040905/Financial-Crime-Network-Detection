# Financial Crime Network Detection Using Graph Analytics and Machine Learning

## Project Overview

Financial crimes such as money laundering are often carried out through multiple transactions between different bank accounts. Traditional methods that analyse individual transactions may fail to identify complex relationships and suspicious patterns between accounts.

This project focuses on detecting suspicious financial activities by analysing transaction networks using graph analytics and machine learning techniques.

The system analyses transaction data, identifies relationships between accounts, extracts network-based features, and helps detect potentially suspicious financial activities through an interactive dashboard.

## Objectives

- Analyse financial transaction data to identify suspicious activities.
- Represent financial transactions as account-to-account networks.
- Apply graph analytics to study relationships between accounts.
- Identify suspicious patterns such as smurfing, structuring, and circular transaction networks.
- Use machine learning to support suspicious account detection.
- Provide explainability insights for detected suspicious activities.
- Present results through an interactive dashboard.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Development Environment | Google Colab, Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Graph Analytics | NetworkX |
| Data Visualization | Matplotlib, Plotly (if used) |
| Dashboard | Python-based dashboard |
| Version Control | Git, GitHub |

## Project Structure

```text
Financial-Crime-Network-Detection/
│
├── Code/
│   └── Financial_Crime_Detection.ipynb
│
├── Dashboard/
│   ├── fraud_dashboard.py
│   ├── fraud_core.py
│   ├── graph_analytics.py
│   ├── report_generator.py
│   ├── generate_evaluator_datasets.py
│   ├── sample_data_generator.py
│   └── requirements.txt
│
├── Dataset/
│   └── Transaction CSV files
│
├── Documentation/
│   └── Project_Report.pdf
│
├── Results/
│   └── Graphs and Visualizations
│
└── README.md
```

## Key Features

### 1. Transaction Data Analysis
Processes financial transaction records and prepares the data for further analysis.

### 2. Graph Analytics
Represents financial transactions as networks to examine relationships and interactions between accounts.

### 3. Suspicious Activity Detection
Analyses transaction patterns to identify potentially suspicious financial activities.

### 4. Machine Learning
Uses machine learning techniques to support the identification and classification of suspicious accounts.

### 5. Explainability
Provides insights into the factors and network characteristics contributing to suspicious account predictions.

### 6. Interactive Dashboard
Provides a dashboard interface to explore transaction data, analyse suspicious activities, and view detection results.

### 7. Report Generation
Supports generating reports containing relevant analysis and detection results.

## Types of Suspicious Patterns

- Smurfing and Structuring
- Circular Transaction Networks
- Suspicious Account Relationships
- Unusual Transaction Activities

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Folder

```bash
cd Financial-Crime-Network-Detection
```

### Step 3: Install Dependencies

```bash
pip install -r Dashboard/requirements.txt
```

### Step 4: Run the Dashboard

The exact command depends on the dashboard framework used.

For a Streamlit dashboard:

```bash
streamlit run Dashboard/fraud_dashboard.py
```

For a standard Python application:

```bash
python Dashboard/fraud_dashboard.py
```

Ensure the required datasets and supporting Python modules are available in the expected locations.

## Dataset

The project includes transaction datasets used for analysis, testing, and evaluation.

The Dataset folder contains sample transaction records and test datasets representing different financial transaction scenarios.

## Documentation

The Documentation folder contains the detailed project report, including the problem statement, objectives, methodology, implementation, results, and conclusion.

## Results

The Results folder contains relevant graphs, visualizations, and evaluation outputs generated during the project.

## Future Enhancements

- Integration with real-time financial transaction streams.
- Improved suspicious transaction detection techniques.
- Enhanced graph-based anomaly detection.
- Advanced explainability and visualization features.
- Deployment of the dashboard for wider accessibility.

## Disclaimer

This project is developed for academic and research purposes. The detection results are intended to support financial crime analysis and should not be treated as definitive evidence of financial wrongdoing.

## Acknowledgement

This project is developed as part of an academic research project at Presidency University.

**Institution:** Presidency University, Bengaluru, Karnataka, India.
