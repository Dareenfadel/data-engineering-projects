# Data Engineering Projects

This repository, created by **Dareen Fadel**, showcases a series of data engineering projects that demonstrate essential skills in data exploration, transformation, and pipeline creation. Each milestone builds upon the previous one, highlighting the use of modern data engineering tools and practices.

---

## Table of Contents
- [Overview](#overview)
- [Milestones](#milestones)
  - [Milestone 1: EDA, Cleaning, Data Transformation, and Feature Engineering](#milestone-1-eda-cleaning-data-transformation-and-feature-engineering)
  - [Milestone 2: Packaging, Databases, and Streaming](#milestone-2-packaging-databases-and-streaming)
  - [Milestone 3: Advanced Data Cleaning and Analysis](#milestone-3-advanced-data-cleaning-and-analysis)
  - [Milestone 4: ETL Pipeline and Dashboard](#milestone-4-etl-pipeline-and-dashboard)
- [Technologies Used](#technologies-used)
- [How to Use](#how-to-use)


---

## Overview

This repository is structured into four key milestones:
1. Exploratory Data Analysis (EDA) and data cleaning.
2. Packaging results into Docker containers and integrating with PostgreSQL and Kafka.
3. Advanced data cleaning, analysis, and feature engineering using Apache Spark.
4. Building an ETL pipeline with Apache Airflow and visualizing results using Plotly.

---

## Milestones

### Milestone 1: EDA, Cleaning, Data Transformation, and Feature Engineering
**Objectives:**
- Perform Exploratory Data Analysis (EDA) to understand the dataset.
- Clean and preprocess the data, including handling missing values.
- Apply data transformation and feature engineering techniques.
- Create lookup tables for encoding and future transformations.

### Milestone 2: Packaging, Databases, and Streaming
**Objectives:**
1. Package the milestone into a Docker image for easy deployment.
2. Save cleaned datasets and lookup tables to:
   - A PostgreSQL database.
   - Local storage as CSV or Parquet files.
3. Utilize Kafka to receive a data stream, process the messages, and store them in the database.

### Milestone 3: Advanced Data Cleaning and Analysis
**Objectives:**
1. Load the dataset and perform tasks using **Apache Spark**:
   - Basic cleaning:
     - Rename columns.
     - Detect, check, and handle missing values.
   - Advanced cleaning and analysis.
   - Feature engineering to create new columns.
   - Encode categorical columns using a lookup table.
2. Save cleaned datasets and lookup tables as:
   - Local files (CSV or Parquet).
   - PostgreSQL database (Bonus).

### Milestone 4: ETL Pipeline and Dashboard
**Objectives:**
1. Create an ETL pipeline using Apache Airflow for automated data processing.
2. Design an interactive dashboard using Plotly to visualize the processed data and insights.

---

## Technologies Used
- **Python**: Core language for data processing and feature engineering.
- **Apache Spark**: Used in Milestone 3 for efficient data processing and transformation.
- **Docker**: For containerization and easy deployment.
- **PostgreSQL**: To store processed data and lookup tables.
- **Kafka**: For real-time data streaming.
- **Apache Airflow**: To create and manage ETL pipelines.
- **Plotly**: To design interactive dashboards in milestone 4.

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Dareenfadel/data-engineering-projects.git
   cd data-engineering-projects
