from sqlalchemy import create_engine

engine = create_engine('postgresql://root:root@pgdatabase:5432/testdb')

def save_to_db(cleaned,lookup):
    if(engine.connect()):
        print('Connected to Database')
        try:
            print('Writing cleaned dataset to database')
            cleaned.to_sql('cleaned_db', con=engine, if_exists='fail')
            print('Done writing to database')
        except ValueError as vx:
            print('Cleaned Table already exists.')
        except Exception as ex:
            print(ex)
        try:
            print('Writing lookup  to database')
            lookup.to_sql('look_up', con=engine, if_exists='fail')
            print('Done writing lookup  to database')
        except ValueError as vx:
            print('lookup Table already exists.')
        except Exception as ex:
            print(ex)    
    else:
        print('Failed to connect to Database')

def save_new_record( new_record_df):
    if(engine.connect()):
        print('Connected to Database')
        try:
            # Save new record to the specified table
            print(f'Adding new record to cleaned_db table')
            print(new_record_df)
            new_record_df.to_sql('cleaned_db', con=engine, if_exists='append')  # Append new record
            print(f'Done adding new record to cleaned_db table')
        except Exception as ex:
            print(f"Error occurred while inserting new record into cleaned_db: {ex}")
    else:
        print('Failed to connect to Database')