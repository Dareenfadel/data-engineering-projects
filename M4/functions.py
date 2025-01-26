


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine

pd.set_option('display.max_columns', None)


def tidycol(fintech_df):
    fintech_df.columns = (fintech_df.columns
    .str.strip()                     # Remove leading/trailing spaces
    .str.lower()                     # Convert to lowercase
    .str.replace(' ', '_')           # Replace spaces with underscores
    .str.replace(r'[^a-zA-Z0-9_]', '', regex=True))
    print(fintech_df.columns)
    return fintech_df

def set_index(fintech_df):
    fintech_df.set_index("loan_id",inplace=True)
    return fintech_df


def tidy_col_and_set_index(df):
    df=tidycol(df)
    df=set_index(df)
    return df


def dups(fintech_df):
    duplicates = fintech_df[fintech_df.duplicated()]
    print(f'Duplicates found:\n{duplicates}')
    duplicates = fintech_df['customer_id'].duplicated()
    duplicate_rows = fintech_df[duplicates]
    print(duplicate_rows)


def type_unique_values(fintech_df):
    fintech_df["type"].unique()



def incorrec_Data(fintech_df):
    incorrect_data = fintech_df[(fintech_df['annual_inc'] < 0) | (fintech_df['loan_amount'] < 0)|(fintech_df["funded_amount"] < 0)|(fintech_df["avg_cur_bal"] < 0)|(fintech_df["tot_cur_bal"] < 0)
                            |(fintech_df["int_rate"] < 0)|(fintech_df['annual_inc_joint'] < 0)]
    print(f'Incorrect data found:\n{incorrect_data}')


def numeric_has_not_numbers(fintech_df):
    numeric_columns = [
    'annual_inc', 
    'annual_inc_joint', 
    'avg_cur_bal', 
    'tot_cur_bal', 
    'loan_amount', 
    'funded_amount', 
    'int_rate']
    for col in numeric_columns:
        filtered_df = fintech_df[fintech_df[col].notna()]
        non_numeric_values = filtered_df[~filtered_df[col].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
    
        print(f"\nNon-numeric values in '{col}':")
        print(non_numeric_values)


def print_Data_types(fintech_df):
    data_types = fintech_df.dtypes
    print("Data Types:\n", data_types)



# %%
def change_term_to_numerical(fintech_df):
    fintech_df['term_no'] = fintech_df['term'].str.replace(' months', '').astype(int)
    fintech_df.head()
    return fintech_df

def observe_emp_length(fintech_df):
    fintech_df["emp_length"].unique()


def emp_length_map(fintech_df):
    emp_length_mapping = {
    '< 1 year': 0,  
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10  }
    fintech_df["emp_length_no"] = fintech_df["emp_length"].map(emp_length_mapping)
    return fintech_df

def check_grade_consistent(fintech_df):
    is_all_between = fintech_df['grade'].between(1, 35).all()

    if is_all_between:
        print("All values in 'grade' are between 1 and 35 inclusive.")
    else:
        print("Not all values in 'grade' are between 1 and 35 inclusive.")


def inconsistent_data(df):
    dups(df)
    type_unique_values(df)
    observe_emp_length(df)
    print_Data_types(df)
    incorrec_Data(df)
    numeric_has_not_numbers(df)
    check_grade_consistent(df)


def change_type_values(fintech_df):
    fintech_df["type"]=fintech_df["type"].replace("Individual","INDIVIDUAL")
    fintech_df["type"]=fintech_df["type"].replace("Joint App","JOINT")
    fintech_df["type"].unique()
    return fintech_df


def remove_dups(fintech_df):
    fintech_df_cleaned = fintech_df.drop_duplicates()
    print(f"Original number of rows: {len(fintech_df)}")
    print(f"Number of rows after removing duplicates: {len(fintech_df_cleaned)}")
    return fintech_df_cleaned



def numerical_col_fix(fintech_df):
    # List of numeric columns to process
    numeric_columns = [
        'annual_inc', 
        'annual_inc_joint', 
        'avg_cur_bal', 
        'tot_cur_bal', 
        'loan_amount', 
        'funded_amount', 
        'int_rate'
    ]
    
    # Function to tidy up numeric columns
    def tidy_numeric_columns_and_print_changes(df, columns):
        changes = {}  # Dictionary to store changes for each column
        for column in columns:
            if column in df.columns:
                original_values = df[column].copy()  # Backup original values
                df[column] = df[column].abs()  # Make values absolute
                df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert to numeric (NaN for errors)
                
                # Store original and modified values
                changes[column] = {
                    'original': original_values,
                    'modified': df[column]
                }
        return df, changes
    
    # Tidy the fintech DataFrame and get changes
    tidied_fintech_df, changes = tidy_numeric_columns_and_print_changes(fintech_df, numeric_columns)
    
    # Print changes for each column
    for column, change in changes.items():
        print(f"\nChanges in column '{column}':")
        
        # Create a DataFrame to compare original and modified values
        comparison_df = pd.DataFrame({
            'Original': change['original'],
            'Modified': change['modified']
        })
        
        # Filter the DataFrame to show only rows where the value changed (excluding NaNs)
        filtered_comparison_df = comparison_df[
            (comparison_df['Original'] != comparison_df['Modified']) &  # Where the values are different
            (comparison_df['Original'].notnull())  # Exclude NaNs
        ]
        
        if not filtered_comparison_df.empty:
            print(filtered_comparison_df)  # Print the filtered comparison
        else:
            print("No changes in this column.")

    return tidied_fintech_df


def fix_grade(fintech_df):
    fintech_df['grade'] = fintech_df['grade'].where(fintech_df['grade'].between(1, 35), np.nan)
    return fintech_df


def lowercase_values(fintech_df):
    fintech_df['purpose']=fintech_df['purpose'].str.lower()
    fintech_df['home_ownership']=fintech_df['home_ownership'].str.lower()
    fintech_df['emp_title']=fintech_df['emp_title'].str.lower()
    fintech_df['verification_status']=fintech_df['verification_status'].str.lower()
    fintech_df['loan_status']=fintech_df['loan_status'].str.lower()
    fintech_df['type']=fintech_df['type'].str.lower()
    fintech_df['description']=fintech_df['description'].str.lower()
    return fintech_df


def inconsistent_handle(df):
    df=change_type_values(df)
    df=remove_dups(df)
    df=numerical_col_fix(df)
    df=fix_grade(df)
    df=lowercase_values(df)
    df=emp_length_map(df)
    df=change_term_to_numerical(df)
    return df



def log_and_visualize(df, col):
    adjusted_col = df[col]
    # Check if there are non-positive values before applying log
    if (adjusted_col < 0).any():
        print(f"Column '{col}' contains non-positive values, log transformation skipped.")
        return adjusted_col, None

    adjusted_col = adjusted_col + 1  # Shift values to avoid log(0)
    
    # Visualization
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    log_col = np.log(adjusted_col)
    
    sns.histplot(df[col], ax=ax[0], kde=True)
    ax[0].set_title(f'Original {col}')

    sns.histplot(log_col, ax=ax[1], kde=True)
    ax[1].set_title(f'Log Transformed {col}')

    zscore_log_col = np.abs((log_col - log_col.mean()) / log_col.std())

    sns.histplot(zscore_log_col, kde=True, ax=ax[2])
    ax[2].set_title(f'Z-score Normalized Log {col}')

    sns.boxplot(x=log_col, ax=ax[3])
    ax[3].set_title(f'Boxplot of Log {col}')

    plt.tight_layout()
    plt.show()

    return log_col, zscore_log_col

# %%
def calculate_outlier_bounds_and_cap(df, values):
    # Calculate Q1, Q3, and IQR
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1

    # Calculate IQR boundaries
    max_iqr = Q3 + 1.5 * IQR
    min_iqr = Q1 - 1.5 * IQR

    # Calculate the 5th and 95th percentiles
    q95 = values.quantile(0.95)
    q05 = values.quantile(0.05)

    # Print minimum and maximum values
    print(f'Min value in : {values.min()}')
    print(f'Max value in : {values.max()}')
    
    # Cap the values based on IQR boundaries
    capped = np.where(values > max_iqr, max_iqr, np.where(values < min_iqr, min_iqr, values))

    return capped, min_iqr, max_iqr, q05, q95

# %%
def cap_z_scores(df, col, threshold=3):
    df_capped = df.copy()  # Create a copy of the dataframe
    
    # Calculate Z-scores for the column, keeping NaN values intact
    col_zscore = (df[col] - df[col].mean()) / df[col].std()
    
    # Apply capping based on the Z-score threshold
    df_capped[col] = np.where(col_zscore > threshold, 
                              df[col].mean() + threshold * df[col].std(),
                              np.where(col_zscore < -threshold, 
                                       df[col].mean() - threshold * df[col].std(), 
                                       df[col]))
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))



    
    sns.histplot(col_zscore, ax=ax[0], kde=True)
    ax[0].set_title(f'Original {col}')


    zscore_col = np.abs((df_capped[col] - df_capped[col].mean()) / df_capped[col].std())


    sns.histplot(zscore_col, kde=True, ax=ax[1])
    ax[1].set_title(f'Z-score Normalized after capping{col}')

    # sns.boxplot(x=df_capped[col] ,ax=ax[2])
    # ax[2].set_title(f'Boxplot of capping {col}')

    plt.tight_layout()
    plt.show()
    return df_capped



def handle_outliers(fintech_df):
    # Cap outliers for 'int_rate'
    df_capped = cap_z_scores(fintech_df, 'int_rate')
    bounds_data = {}
    # Log transformation and outlier capping for 'annual_inc'
    log_annual_inc, zscore_log_annual_inc = log_and_visualize(fintech_df, 'annual_inc')
    capped_annual_inc, min_iqr_annual, max_iqr_annual, q05_annual, q95_annual = calculate_outlier_bounds_and_cap(fintech_df, log_annual_inc)
    bounds_data['annual_inc'] = {'min': min_iqr_annual, 'max': max_iqr_annual}
    
    # Log transformation and outlier capping for 'annual_inc_joint'
    log_annual_inc_joint, zscore_log_annual_inc_joint = log_and_visualize(fintech_df, 'annual_inc_joint')
    capped_annual_inc_joint, min_iqr_joint, max_iqr_joint, q05_joint, q95_joint = calculate_outlier_bounds_and_cap(fintech_df, log_annual_inc_joint)
    bounds_data['annual_inc_joint'] = {'min': min_iqr_joint, 'max': max_iqr_joint}
    # Log transformation and outlier capping for 'tot_cur_bal'
    log_tot_cur_bal, zscore_log_tot_cur_bal = log_and_visualize(fintech_df, 'tot_cur_bal')
    capped_tot_cur_bal, min_iqr_tot, max_iqr_tot, q05_tot, q95_tot = calculate_outlier_bounds_and_cap(fintech_df, log_tot_cur_bal)
    bounds_data['tot_cur_bal'] = {'min': min_iqr_tot, 'max': max_iqr_tot}

    # Log transformation and outlier capping for 'avg_cur_bal'
    log_avg_cur_bal, zscore_log_avg_cur_bal = log_and_visualize(fintech_df, 'avg_cur_bal')
    print(log_avg_cur_bal)
    capped_avg_cur_bal, min_iqr_avg, max_iqr_avg, q05_avg, q95_avg = calculate_outlier_bounds_and_cap(fintech_df, log_avg_cur_bal)
    bounds_data['avg_cur_bal'] = {'min': min_iqr_avg, 'max': max_iqr_avg}

    # Assign capped values to the DataFrame
    df_capped["annual_inc"] = capped_annual_inc
    df_capped["annual_inc_joint"] = capped_annual_inc_joint
    df_capped["tot_cur_bal"] = capped_tot_cur_bal
    df_capped["avg_cur_bal"] = capped_avg_cur_bal

    bounds_df = pd.DataFrame(bounds_data).T 

    return df_capped ,bounds_df


def annual_inc_joint_missing(df_missing):
    df_missing['annual_inc_joint'] = df_missing['annual_inc_joint'].fillna(df_missing['annual_inc'])
    missing_count_after = df_missing['annual_inc_joint'].isnull().sum()
    print(f'Missing annual_inc_joint after imputation: {missing_count_after}')
    return df_missing


def description_missing(df_missing):
    df_copy = df_missing.copy()
    purpose_mode_description = df_copy.groupby('purpose')['description'].apply(lambda x: x.mode()[0] if not x.mode().empty else None)
    df_missing = df_missing.merge(purpose_mode_description.rename('mode_description'), on='purpose', how='left')
    df_missing['description'] = df_missing['description'].fillna(df_missing['mode_description'])
    df_missing.drop('mode_description', axis=1, inplace=True)
    df_missing.index = df_copy.index
    missing_count_after = df_missing['description'].isnull().sum()
    print(f'Missing description count after imputation: {missing_count_after}')
    return df_missing




def int_rate_missing(df_missing):
    df_copy = df_missing.copy()
    mean_int_rate_state_grade = df_copy.groupby(['state', 'grade'])['int_rate'].mean().reset_index()
    mean_int_rate_state_grade.rename(columns={'int_rate': 'mean_int_rate_state_grade'}, inplace=True)
    df_missing = df_missing.merge(mean_int_rate_state_grade, on=['state', 'grade'], how='left')

    df_missing['int_rate'] = df_missing['int_rate'].fillna(df_missing['mean_int_rate_state_grade'])

    mean_int_rate_state = df_copy.groupby('state')['int_rate'].mean().reset_index()
    mean_int_rate_state.rename(columns={'int_rate': 'mean_int_rate_state'}, inplace=True)

    df_missing = df_missing.merge(mean_int_rate_state, on='state', how='left')

    df_missing['int_rate'] = df_missing['int_rate'].fillna(df_missing['mean_int_rate_state'])

    df_missing.drop(['mean_int_rate_state_grade', 'mean_int_rate_state'], axis=1, inplace=True)

    df_missing.index = df_copy.index

    missing_count_after = df_missing['int_rate'].isnull().sum()
    print(f'Missing interest rates after imputation: {missing_count_after}')
    print(df_missing[['state', 'grade', 'int_rate']].head(10))
    return df_missing


def emp_length_title_missing(df_missing):
    # Categorize 'annual_inc' directly in df_missing
    df_copy = df_missing.copy()
    df_missing['annual_inc_cat'] = pd.qcut(df_missing['annual_inc'], q=3, labels=['Low', 'Medium', 'High'])

    # Compute mode values for each category of 'annual_inc_cat'
    mode_values = df_missing.groupby('annual_inc_cat').agg({
        'emp_length': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        'emp_title': lambda x: x.mode().iloc[0] if not x.mode().empty else None
    }).reset_index()

    # Merge mode values back to df_missing
    df_missing = df_missing.merge(
        mode_values,
        on='annual_inc_cat',
        how='left',
        suffixes=('', '_mode')
    )

    # Impute missing values
    df_missing['emp_length'] = df_missing['emp_length'].fillna(df_missing['emp_length_mode'])
    df_missing['emp_title'] = df_missing['emp_title'].fillna(df_missing['emp_title_mode'])

    # Drop temporary columns
    df_missing.drop(columns=['emp_length_mode', 'emp_title_mode'], inplace=True)

    # Report remaining missing values for verification
    missing_emp_length = df_missing['emp_length'].isnull().sum()
    missing_emp_title = df_missing['emp_title'].isnull().sum()
    print(f'Missing emp_length after imputation: {missing_emp_length}')
    print(f'Missing emp_title after imputation: {missing_emp_title}')

    # Optional: Check the result for the first few rows
    print(df_missing[['annual_inc', 'annual_inc_cat', 'emp_length', 'emp_title']].head(10))
    df_missing=emp_length_map(df_missing)
    df_missing.index = df_copy.index
    return df_missing


def save_consolidated_imputation_statistics(df):
    # Mode for 'description' based on 'purpose'
    description_mode = df['description'].mode()[0] if not df['description'].mode().empty else None
    
    # Mean of 'int_rate'
    int_rate_mean = df['int_rate'].mean()
    
    # Mode of 'emp_length' and 'emp_title' (overall, not grouped by anything)
    emp_length_mode = df['emp_length'].mode()[0] if not df['emp_length'].mode().empty else None
    emp_title_mode = df['emp_title'].mode()[0] if not df['emp_title'].mode().empty else None

    # Create a consolidated dictionary
    consolidated_data = {
        'description_mode': description_mode,
        'int_rate_mean': int_rate_mean,
        'emp_length_mode': emp_length_mode,
        'emp_title_mode': emp_title_mode
    }

    # Return as a DataFrame for easy viewing
    consolidated_df = pd.DataFrame([consolidated_data])
    
    return consolidated_df





def handle_missing(df_capped):
    df_missing=df_capped.copy()
    df_missing=description_missing(df_missing)

    df_missing=annual_inc_joint_missing(df_missing)
    df_missing=int_rate_missing(df_missing)
    df_missing=emp_length_title_missing(df_missing)
    missing_lookup=save_consolidated_imputation_statistics(df_missing)
    return df_missing,missing_lookup





def month_no(df_missing):
    df_feature_engineering=df_missing.copy()
    df_feature_engineering["issue_date"]=df_feature_engineering["issue_date"].astype("datetime64[s]")
    df_feature_engineering["month_no"]=df_feature_engineering["issue_date"].dt.month
    df_feature_engineering["month_no"].dtype
    df_feature_engineering.head()
    return df_feature_engineering


def  salary_can_cover(df_feature_engineering,fintech_df):
    df_feature_engineering["salary_can_cover"] = np.where(
    fintech_df["type"] == "joint",
    fintech_df["annual_inc_joint"] >= fintech_df["loan_amount"],
    fintech_df["annual_inc"] >= fintech_df["loan_amount"]) 
    df_feature_engineering["salary_can_cover"].dtype
    df_feature_engineering.head()
    return df_feature_engineering

# %%
def grade_to_no(df_feature_engineering):
    conditions = [
    (df_feature_engineering["grade"] >= 1) & (df_feature_engineering["grade"] <= 5),
    (df_feature_engineering["grade"] >= 6) & (df_feature_engineering["grade"] <= 10),
    (df_feature_engineering["grade"] >= 11) & (df_feature_engineering["grade"] <= 15),
    (df_feature_engineering["grade"] >= 16) & (df_feature_engineering["grade"] <= 20),
    (df_feature_engineering["grade"] >= 21) & (df_feature_engineering["grade"] <= 25),
    (df_feature_engineering["grade"] >= 26) & (df_feature_engineering["grade"] <= 30),
    (df_feature_engineering["grade"] >= 31) & (df_feature_engineering["grade"] <= 35)]
    values = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df_feature_engineering["letter_grade"] = np.select(conditions, values, default='A')
    df_feature_engineering.head()
    return df_feature_engineering

# %%
def month_ins(df_missing,fintech_df,df_feature_engineering):
    r=(df_missing["int_rate"]/12)
    r_plus1=r+1
    no_of_months=np.where(fintech_df["term"]=="36 months",36,60)
    # df_feature_engineering["monthly_installement"]=fintech_df["loan_amount"]*(np.power(r_plus1,no_of_months)*r)/float(np.power(r_plus1,no_of_months)-1)
    df_feature_engineering["monthly_installement"]= fintech_df["loan_amount"]* ((np.power(r_plus1.astype(float), no_of_months.astype(float)) * r.astype(float)) / (np.power(r_plus1.astype(float), no_of_months.astype(float)) - 1))
   
   
    return df_feature_engineering

# %%
def add_col(fintech_df,df):
   df= month_no(df)
   df=salary_can_cover(df,fintech_df)
   df=grade_to_no(df)
   df=month_ins(df,fintech_df,df)
   return df


def encode_home_ownership(df_feature_engineering):
    df_encoded=df_feature_engineering.copy()
    encoder_home = OneHotEncoder(sparse_output=False)
    encoded = encoder_home.fit_transform(df_encoded[['home_ownership']])
    encoded_df = pd.DataFrame(encoded, columns=encoder_home.get_feature_names_out(['home_ownership']),index=df_encoded.index)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    df_encoded.head()
    home_ownership_lookup = pd.DataFrame({
    'Column': 'home_ownership',
    'Original': encoder_home.categories_[0],
    'Encoded': encoder_home.get_feature_names_out(['home_ownership']).tolist()
})
    home_ownership_lookup
    return df_encoded,home_ownership_lookup

# %%
def encode_verification(df_encoded):
    encoder_verification_status = OneHotEncoder(sparse_output=False)
    encoded = encoder_verification_status.fit_transform(df_encoded[['verification_status']])
    encoded_df = pd.DataFrame(encoded, columns=encoder_verification_status.get_feature_names_out(['verification_status']),index=df_encoded.index)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    df_encoded.head()
    verification_status_lookup = pd.DataFrame({
    'Column': "verification_status",
    'Original': encoder_verification_status.categories_[0],
    'Encoded': encoder_verification_status.get_feature_names_out(['verification_status']).tolist()
})
    verification_status_lookup
    
    return df_encoded,verification_status_lookup


def encode_addr_state(df_encoded):
    df_encoded["addr_state"].nunique()
    encoder_addr_state = LabelEncoder()
    df_encoded['addr_state_encoded'] = encoder_addr_state.fit_transform(df_encoded['addr_state'])
    df_encoded[['addr_state', 'addr_state_encoded']].sample(5)
    addr_state_lookup = pd.DataFrame({
    'Column': 'addr_state',
    'Original': encoder_addr_state.classes_,
    'Encoded': range(len(encoder_addr_state.classes_))
})
    addr_state_lookup.head()
    return df_encoded,addr_state_lookup




def encode_state(df_encoded):
    df_encoded["state"].nunique()
    encoder_state = LabelEncoder()
    df_encoded['state_encoded'] = encoder_state.fit_transform(df_encoded['state'])
    df_encoded[['state', 'state_encoded']].sample(5)
    state_lookup = pd.DataFrame({
    'Column': 'state',
    'Original': encoder_state.classes_,
    'Encoded': range(len(encoder_state.classes_))})
    state_lookup.head()
    return df_encoded,state_lookup


def encode_loan(df_encoded):
    df_encoded["loan_status"].nunique()
    encoder_loan_status = OneHotEncoder(sparse_output=False)
    encoded = encoder_loan_status.fit_transform(df_encoded[['loan_status']])
    encoded_df = pd.DataFrame(encoded, columns=encoder_loan_status.get_feature_names_out(['loan_status']),index=df_encoded.index)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    df_encoded.head()
    loan_status_lookup = pd.DataFrame({
    'Column': 'loan_status',
    'Original': encoder_loan_status.categories_[0],
    'Encoded': encoder_loan_status.get_feature_names_out(['loan_status']).tolist()
})
    loan_status_lookup

    return df_encoded,loan_status_lookup

def encode_type(df_encoded):
    df_encoded["type"].nunique()
    encoder_type = OneHotEncoder(sparse_output=False)
    encoded = encoder_type.fit_transform(df_encoded[['type']])
    encoded_df = pd.DataFrame(encoded, columns=encoder_type.get_feature_names_out(['type']),index=df_encoded.index)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    df_encoded.head()
    type_lookup = pd.DataFrame({
    'Column': "type",
    'Original': encoder_type.categories_[0],
    'Encoded':encoder_type.get_feature_names_out(['type']).tolist()
})
    type_lookup
    return df_encoded,type_lookup


def encode_letter_grade(df_encoded):
    df_encoded["letter_grade"].nunique()
    encoder_letter_grade = LabelEncoder()
    df_encoded['letter_grade_encoded'] = encoder_letter_grade.fit_transform(df_encoded['letter_grade'])
    df_encoded[['letter_grade', 'letter_grade_encoded']].sample(5)
    letter_grade_lookup = pd.DataFrame({
    'Column': 'letter_grade',
    'Original': encoder_letter_grade.classes_,
    'Encoded': range(len(encoder_letter_grade.classes_))
})
    letter_grade_lookup
    return df_encoded,letter_grade_lookup


def encode_purpose(df_encoded):
    df_encoded["purpose"].nunique()
    df_encoded["purpose"].nunique()
    encoderpurpose = LabelEncoder()
    df_encoded['purpose_encoded'] = encoderpurpose.fit_transform(df_encoded['purpose'])
    df_encoded[['purpose', 'purpose_encoded']].sample(5)
    purpose_lookup = pd.DataFrame({
    'Column': 'purpose',
    'Original': encoderpurpose.classes_,
    'Encoded': range(len(encoderpurpose.classes_))
})
    purpose_lookup
    return df_encoded,purpose_lookup


def encoding(df):
    df,home_ownership_lookup=encode_home_ownership(df)
    df,verification_status_lookup=encode_verification(df)
    df,addr_state_lookup=encode_addr_state(df)
    df,state_lookup=encode_state(df)
    df,loan_status_lookup=encode_loan(df)
    df,type_lookup=encode_type(df)
    df,letter_grade_lookup= encode_letter_grade(df)
    df,purpose_lookup=encode_purpose(df)
    df['pymnt_plan_encoded'] = df['pymnt_plan'].apply(lambda x: 1 if x else 0)
    df["salary_can_cover_encoded"] = df['salary_can_cover'].apply(lambda x: 1 if x else 0)

    emp_length_mapping = {
    '< 1 year': 0,  
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10    
}


    emp_length_and_term_lookup = pd.DataFrame({
    'Column': 'emp_length',
    'Original': list(emp_length_mapping.keys()) + ['36 months', '60 months'],
    'Encoded': [emp_length_mapping[key] for key in emp_length_mapping.keys()] + [36, 60]})
   
    payment_plan_mapping = {
    True: 1,
    False: 0
}
    rows = []
    for original, encoded in payment_plan_mapping.items():
        rows.append({
        'Column': 'pymnt_plan',
        'Original': original,
        'Encoded': encoded
    })
    pymnt_plan_lookup = pd.DataFrame(rows)
    salary_can_cover_mapping = {
    True: 1,
    False: 0
    }
    rows = []
    for original, encoded in salary_can_cover_mapping.items():
        rows.append({
        'Column': "salary_can_cover",
        'Original': original,
        'Encoded': encoded
    })
    salary_can_cover_lookup = pd.DataFrame(rows)
    lookup_df = pd.concat([purpose_lookup, type_lookup,verification_status_lookup,state_lookup,addr_state_lookup,loan_status_lookup,letter_grade_lookup,
                       home_ownership_lookup,emp_length_and_term_lookup,pymnt_plan_lookup,salary_can_cover_lookup], ignore_index=True)
    return df,lookup_df




def normalize_loan(df_encoded):
    # Initialize the scaler
    scaler = MinMaxScaler()
    
    # Min-Max Normalization
    df_encoded['loan_amount_normalized'] = scaler.fit_transform(df_encoded[['loan_amount']])
    
    # Log Transformation
    df_encoded['loan_amount_normalize_log'] = np.log(df_encoded['loan_amount'] + 1)
    
    # Plotting the histograms
    plt.figure(figsize=(12, 6))
    
    # Min-Max Normalization Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(x=df_encoded["loan_amount_normalized"], kde=True)
    plt.title('Min-Max Normalized Loan Amount')
    plt.xlabel('Min-Max Value')
    
    # Log Transformation Histogram
    plt.subplot(1, 2, 2)
    sns.histplot(x=df_encoded["loan_amount_normalize_log"], kde=True)
    plt.title('Log-Transformed Loan Amount')
    plt.xlabel('Log Value')
    
    plt.tight_layout()
    plt.show()
    
    return df_encoded

# %%
def normalize_fund(df_encoded):
    scaler = MinMaxScaler() 
    df_encoded['funded_amount_normalized'] = scaler.fit_transform(df_encoded[['funded_amount']]) 
    df_encoded[['funded_amount', 'funded_amount_normalized']]
    df_encoded['funded_amount_logged'] = np.log(df_encoded['funded_amount']+1)
    return df_encoded


def normalize_monthly_installement(df_encoded):
    scaler = MinMaxScaler() 
    df_encoded["monthly_installement_normalized"] = scaler.fit_transform(df_encoded[['monthly_installement']]) 
    df_encoded[['monthly_installement', 'monthly_installement_normalized']]
    df_encoded['monthly_installement_logged'] = np.log(df_encoded['monthly_installement']+1)
    return df_encoded


def normalization(df):
   df= normalize_loan(df)
   df=normalize_fund(df)
   df=normalize_monthly_installement(df)
   return df





def removeuneeded_col(df_bonus):
	df_bonus = df_bonus.drop(columns=["emp_length",	"home_ownership",	"verification_status","addr_state",	"loan_status",	"loan_amount",	"state"	,"funded_amount",
                                  	"term",	"grade"		,"pymnt_plan",	"type","purpose","monthly_installement_logged","funded_amount_logged",	"loan_amount_normalize_log","salary_can_cover"])
	return df_bonus
	


def handle_message_outliers(message,bounds_df):

    # Cap each relevant field in the message using the np.where approach
    for column in bounds_df.index:
        if column in message and not pd.isnull(message[column]):  # Ensure column is in message and not NaN
            min_iqr = bounds_df.loc[column, 'min']
            max_iqr = bounds_df.loc[column, 'max']
            # Apply the bounds using np.where to cap the values
            message[column] = np.where(message[column] > max_iqr, max_iqr,
                                       np.where(message[column] < min_iqr, min_iqr, message[column]))
    return message

def missingMessage(df, imputation_stats):
    # Apply handle_message_missing row by row (each row is a pandas Series)
    df_missing = df.apply(lambda row: handle_message_missing(row, imputation_stats), axis=1)
    return df_missing

def handle_message_missing(message, imputation_stats):
    print(message)
    if isinstance(message, pd.Series):
        # Description Imputation (use the saved mode for 'description')
        if pd.isnull(message['description']):
            mode_description = imputation_stats['description_mode'].iloc[0]  # Access the saved mode directly
            message['description'] = mode_description

        # annual_inc_joint Imputation
        if pd.isnull(message['annual_inc_joint']) and pd.notnull(message['annual_inc']):
            message['annual_inc_joint'] = message['annual_inc']

        # int_rate Imputation (use the saved mean for 'int_rate')
        if pd.isnull(message['int_rate']):
            mean_int_rate = imputation_stats['int_rate_mean'].iloc[0]  # Access the saved mean directly
            message['int_rate'] = mean_int_rate

        # emp_length and emp_title Imputation (use the saved mode for 'emp_length' and 'emp_title')
        if pd.isnull(message['emp_length']):
            emp_length_mode = imputation_stats['emp_length_mode'].iloc[0]  # Access the saved mode directly
            message['emp_length'] = emp_length_mode

        if pd.isnull(message['emp_title']):
            emp_title_mode = imputation_stats['emp_title_mode'].iloc[0]  # Access the saved mode directly
            message['emp_title'] = emp_title_mode

    else:
        raise ValueError("Expected message to be a pandas Series, but received a different type.")

    return message


def encode_columns(df):
    # Create the lookup tables for each column
    purpose_lookup = {
        'car': 0, 'credit_card': 1, 'debt_consolidation': 2, 'home_improvement': 3,
        'house': 4, 'major_purchase': 5, 'medical': 6, 'moving': 7, 'other': 8,
        'renewable_energy': 9, 'small_business': 10, 'vacation': 11, 'wedding': 12
    }

    type_lookup = {
        'direct_pay': 'type_direct_pay', 'individual': 'type_individual', 'joint': 'type_joint'
    }

    verification_status_lookup = {
        'not verified': 'verification_status_not verified',
        'source verified': 'verification_status_source verified',
        'verified': 'verification_status_verified'
    }

    state_lookup = { 
        'AK': 0, 'AL': 1, 'AR': 2, 'AZ': 3, 'CA': 4, 'CO': 5, 'CT': 6, 'DC': 7, 'DE': 8,
        'FL': 9, 'GA': 10, 'HI': 11, 'ID': 12, 'IL': 13, 'IN': 14, 'KS': 15, 'KY': 16, 'LA': 17,
        'MA': 18, 'MD': 19, 'ME': 20, 'MI': 21, 'MN': 22, 'MO': 23, 'MS': 24, 'MT': 25, 'NC': 26,
        'ND': 27, 'NE': 28, 'NH': 29, 'NJ': 30, 'NM': 31, 'NV': 32, 'NY': 33, 'OH': 34, 'OK': 35,
        'OR': 36, 'PA': 37, 'RI': 38, 'SC': 39, 'SD': 40, 'TN': 41, 'TX': 42, 'UT': 43, 'VA': 44,
        'VT': 45, 'WA': 46, 'WI': 47, 'WV': 48, 'WY': 49
    }

    loan_status_lookup = {
        'charged off': 'loan_status_charged off', 'current': 'loan_status_current', 
        'default': 'loan_status_default', 'fully paid': 'loan_status_fully paid', 
        'in grace period': 'loan_status_in grace period', 
        'late (16-30 days)': 'loan_status_late (16-30 days)', 
        'late (31-120 days)': 'loan_status_late (31-120 days)'
    }

    letter_grade_lookup = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6
    }

    home_ownership_lookup = {
        'any': 'home_ownership_any', 'mortgage': 'home_ownership_mortgage', 
        'own': 'home_ownership_own', 'rent': 'home_ownership_rent'
    }

    emp_length_lookup = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5,
        '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10,
        '36 months': 36, '60 months': 60
    }

    pymnt_plan_lookup = {'TRUE': 1, 'FALSE': 0}
    salary_can_cover_lookup = {'TRUE': 1, 'FALSE': 0}

    # Apply the encoding using the lookup dictionaries
    df['purpose_encoded'] = df['purpose'].map(purpose_lookup)
    
    # Correct the use of the map function here:
    df['type_direct_pay'] = df['type'].map(lambda x: 1 if x == 'direct_pay' else 0)
    df['type_individual'] = df['type'].map(lambda x: 1 if x == 'individual' else 0)
    df['type_joint'] = df['type'].map(lambda x: 1 if x == 'joint' else 0)
    
    df['verification_status_not verified'] = df['verification_status'].map(lambda x: 1 if x == 'not verified' else 0)
    df['verification_status_source verified'] = df['verification_status'].map(lambda x: 1 if x == 'source verified' else 0)
    df['verification_status_verified'] = df['verification_status'].map(lambda x: 1 if x == 'verified' else 0)
    
    df['state_encoded'] = df['state'].map(state_lookup)
    df['addr_state_encoded'] = df['addr_state'].map(state_lookup)
    
    df['loan_status_charged off'] = df['loan_status'].map(lambda x: 1 if x == 'charged off' else 0)
    df['loan_status_current'] = df['loan_status'].map(lambda x: 1 if x == 'current' else 0)
    df['loan_status_default'] = df['loan_status'].map(lambda x: 1 if x == 'default' else 0)
    df['loan_status_fully paid'] = df['loan_status'].map(lambda x: 1 if x == 'fully paid' else 0)
    df['loan_status_in grace period'] = df['loan_status'].map(lambda x: 1 if x == 'in grace period' else 0)
    df['loan_status_late (16-30 days)'] = df['loan_status'].map(lambda x: 1 if x == 'late (16-30 days)' else 0)
    df['loan_status_late (31-120 days)'] = df['loan_status'].map(lambda x: 1 if x == 'late (31-120 days)' else 0)
    
    df['letter_grade_encoded'] = df['letter_grade'].map(letter_grade_lookup)
    
    df['home_ownership_any'] = df['home_ownership'].map(lambda x: 1 if x == 'any' else 0)
    df['home_ownership_mortgage'] = df['home_ownership'].map(lambda x: 1 if x == 'mortgage' else 0)
    df['home_ownership_own'] = df['home_ownership'].map(lambda x: 1 if x == 'own' else 0)
    df['home_ownership_rent'] = df['home_ownership'].map(lambda x: 1 if x == 'rent' else 0)
    
    df['emp_length_no'] = df['emp_length'].map(emp_length_lookup)
    
    df['pymnt_plan_encoded'] = df['pymnt_plan'].apply(lambda x: 1 if x else 0)
    df["salary_can_cover_encoded"] = df['salary_can_cover'].apply(lambda x: 1 if x else 0)
    
    return df


def clean(df):
    cleanedColoumns_df=tidy_col_and_set_index(df)
    consistent_df=inconsistent_handle(cleanedColoumns_df)
    # outliers_Removed_df, bounds_df=handle_outliers(consistent_df)
   
    df_missing,missing_lookup=handle_missing(consistent_df)
    columns_to_keep = [
    'customer_id', 
    'emp_title', 
    'emp_length', 
    'home_ownership', 
    'annual_inc', 
    'annual_inc_joint', 
    'verification_status', 
    'zip_code', 
    'addr_state', 
    'avg_cur_bal', 
    'tot_cur_bal', 
    'loan_status', 
    'loan_amount', 
    'state', 
    'funded_amount', 
    'term', 
    'int_rate', 
    'grade', 
    'issue_date', 
    'pymnt_plan', 
    'type', 
    'purpose', 
    'description',
    'term_no',
    'emp_length_no'
]


    df_missing = df_missing.loc[:, columns_to_keep]

    new_col=add_col(df,df_missing)
   
    # encoded,lookup_df=encoding(new_col)
    # normalized_df=normalization(encoded)
    # final_df=removeuneeded_col(normalized_df)
    return new_col

def encode_and_normalize(new_col):
    outliers_Removed_df, bounds_df=handle_outliers(new_col)
    normalized_df=normalization(outliers_Removed_df)
    
    encoded,lookup=encoding(normalized_df)

    return encoded

def cleanMessage(df,bounds_df,imputation_stats,lookup_df):
        cleanedColoumns_df=tidy_col_and_set_index(df)
        consistent_df=inconsistent_handle(cleanedColoumns_df)
        outliers_Removed_df =handle_message_outliers(consistent_df,bounds_df)
        df_missing=missingMessage(outliers_Removed_df,imputation_stats)
        columns_to_keep = [
    'customer_id', 
    'emp_title', 
    'emp_length', 
    'home_ownership', 
    'annual_inc', 
    'annual_inc_joint', 
    'verification_status', 
    'zip_code', 
    'addr_state', 
    'avg_cur_bal', 
    'tot_cur_bal', 
    'loan_status', 
    'loan_amount', 
    'state', 
    'funded_amount', 
    'term', 
    'int_rate', 
    'grade', 
    'issue_date', 
    'pymnt_plan', 
    'type', 
    'purpose', 
    'description',
    'term_no',
    'emp_length_no'
]
        df_missing = df_missing.loc[:, columns_to_keep]
        new_col=add_col(df,df_missing)
        encoded=encode_columns(new_col)
        normalized_df=normalization(encoded)
        final_df=removeuneeded_col(normalized_df)
        print("finalizeeeeee")
        print(final_df.head())
        return final_df


def extract_clean(filename):
    df = pd.read_csv(filename)
    df = clean(df)
    df.to_csv('/opt/airflow/data/fintech_clean.csv',index=True)
    print('loaded after cleaning succesfully')

def transform(filename):
    df = pd.read_csv(filename)
    df =encode_and_normalize(df)
    try:
        df.to_csv('/opt/airflow/data/fintech_transformed.csv',index=False, mode='x')
        print('loaded after cleaning succesfully')
    except FileExistsError:
        print('file already exists')

def load_to_csv(df,filename):
    df.to_csv(filename,index=False)
    print('loaded succesfully')
    
def load_to_postgres(filename): 
    df = pd.read_csv(filename)
    engine = create_engine('postgresql://root:root@pgdatabase:5432/fintech_etl')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'fintech_db',con = engine,if_exists='replace')

