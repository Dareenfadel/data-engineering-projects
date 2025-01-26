print('> Starting...')
import pandas as pd
from consumer import consumer
from db import save_new_record, save_to_db
import os
from cleaning import clean, cleanMessage
from run_producer import start_producer, stop_container

def consume():

    for message in consumer:
        print(f'Message received: {message.value}')

        if message.value == 'EOF':
            break
        else:
            df = pd.DataFrame([message.value])
            bounds_df = pd.read_csv("data/outliers_boundries.csv") 
            imputation_stats = pd.read_csv("data/missing_lookup.csv") 
            lookup_df = pd.read_csv("data/lookup_fintech_data_MET_2_52-21362.csv") 
            cleanedMessage=cleanMessage(df,bounds_df,imputation_stats,lookup_df)
            save_new_record(cleanedMessage)
            









def main():
    print('Loading.......')
    data_path = "data/fintech_data_MET_2_52-21362_clean.csv"
    data_path2 = "data/lookup_fintech_data_MET_2_52-21362.csv"
    data_path3="data/outliers_boundries.csv"
    data_path4="data/missing_lookup.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path) 
        lookup=pd.read_csv(data_path2)
    else:
        df = pd.read_csv("data/fintech_data_30_52_21362.csv")  
        df,lookup,bounds_df,missing_lookup = clean(df)
        print("data after clean")
        print( df.head())

        df.to_csv(data_path, index=False)
        lookup.to_csv(data_path2, index=False)
        bounds_df.to_csv(data_path3,index=False)
        missing_lookup.to_csv(data_path4,index=False)

    save_to_db(df,lookup)
    print('> Done!')
    id = "52_21362" 


    producer_id = start_producer(id, topic_name='fintech', kafka_url = 'kafka:9092')
    print(f"Producer started with ID: {producer_id}")
   

    consume()


    
    stop_container(producer_id)
    print("Producer container stopped.")


# if __name__ == '__main__':
print('In main')
main()