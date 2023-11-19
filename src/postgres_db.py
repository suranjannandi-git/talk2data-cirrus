import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime
from psycopg2.extras import RealDictCursor
import json
from prettytable import PrettyTable

import logging

logger = logging.getLogger(__name__)
from boxsdk import Client, OAuth2

DB_ALIAS = None

load_dotenv()  # take environment variables from .env file
talk2data_config_db_host_name = os.getenv('TALK2DATA_CONFIG_DB_HOST_NAME')
talk2data_config_db_user_id = os.getenv('TALK2DATA_CONFIG_DB_USER_ID')
talk2data_config_db_password = os.getenv('TALK2DATA_CONFIG_DB_PASSWORD')
talk2data_config_db_port_no = os.getenv('TALK2DATA_CONFIG_DB_PORT_NO')
talk2data_config_db_database_name = os.getenv('TALK2DATA_CONFIG_DB_DATABASE_NAME')
talk2data_db_host_name = os.getenv('TALK2DATA_DB_HOST_NAME')
talk2data_db_user_id = os.getenv('TALK2DATA_DB_USER_ID')
talk2data_db_password = os.getenv('TALK2DATA_DB_PASSWORD')
talk2data_db_port_no = os.getenv('TALK2DATA_DB_PORT_NO')
talk2data_db_database_name = os.getenv('TALK2DATA_DB_DATABASE_NAME')
sql_result_download_directory = os.getenv('TALK2DATA_SQL_RESULT_DOWNLOAD_PATH')
box_sql_result_folder_id = os.getenv('BOX_SQL_RESULT_FOLDER_ID')
box_api_token = os.getenv('BOX_API_TOKEN')


def save_df_to_excel(df):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f'query_results_{timestamp}.xlsx'
    df.to_excel(f'{sql_result_download_directory}/{filename}', index=False)

    # upload the file to box
    auth = OAuth2(client_id='', client_secret='', access_token=box_api_token)
    boxClient = Client(auth)
    box_file = boxClient.folder(box_sql_result_folder_id).upload(f'{sql_result_download_directory}/{filename}')
    box_file_url = f'https://ibm.ent.box.com/file/{box_file.id}'
    return box_file_url


def change_column_name(input_str):
    words = input_str.split('_')
    pascal_case_words = [word.capitalize() for word in words]
    return ' '.join(pascal_case_words)


def execute_sql_query(sql_query):
    try:
        connection_uri = f"postgresql://{talk2data_db_user_id}:{talk2data_db_password}@{talk2data_db_host_name}:{talk2data_db_port_no}/{talk2data_db_database_name}"
        connection = psycopg2.connect(connection_uri)
        cursor = connection.cursor()
        logger.info(f'Executing SQL query: {sql_query}')
        cursor.execute(sql_query)
        result = cursor.fetchall()
        logger.info(f'Executed SQL query: {sql_query}')
        row_count = cursor.rowcount
        col_names = [query_result[0] for query_result in cursor.description]
        result_df = pd.DataFrame(result, columns=col_names)
        result_df.columns = [change_column_name(col) for col in result_df.columns]
        result_url = save_df_to_excel(result_df)  # save the result to excel file

        # Check the number of rows and columns
        num_rows, num_columns = result_df.shape
        # Close the cursor and connection when done
        cursor.close()
        connection.close()
    except Exception as e:
        return {"sql_generation_status": "INVALID", "sql_query": str(sql_query),
                "nl_response": f"failed to execute query error: {e}", "result": "ERROR while executing query.",
                "st_result": "Error while executing SQL query.", "sql_result_url": "Error while executing SQL query."}

    if row_count == 0:
        return {"sql_generation_status": "VALID", "sql_query": str(sql_query),
                "nl_response": "No record found for your query.", "result": "No record found for your query.",
                "st_result": "No record found for your query.", "sql_result_url": "No record found for your query."}
    elif num_rows >= 10 or num_columns >=5:
        return {"sql_generation_status": "VALID", "sql_query": str(sql_query), "nl_response": str(result), "result": "Too many records to display here. Please check the report link.",
                "st_result": result_df.to_html(index=False), "sql_result_url": result_url}
    else:
        return {"sql_generation_status": "VALID", "sql_query": str(sql_query), "nl_response": str(result),
                "result": ConvertDFtoPrettyTable(df=result_df),
                "st_result": result_df.to_html(index=False), "sql_result_url": result_url}


def ConvertDFtoPrettyTable(df: pd.DataFrame):
    # Create a PrettyTable object
    pt = PrettyTable()

    # Set the field names to the column names of the DataFrame
    pt.field_names = df.columns.tolist()

    # Loop through each row in the DataFrame and add it to the PrettyTable
    for index, row in df.iterrows():
        pt.add_row(row)

    table = pt.get_string()
    json_blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"```{table}```"
            }
        }
    ]

    return json_blocks


def get_table_description_from_db(table_name):
    connection_uri = f"postgresql://{talk2data_config_db_user_id}:{talk2data_config_db_password}@{talk2data_config_db_host_name}:{talk2data_config_db_port_no}/{talk2data_config_db_database_name}"
    conn = psycopg2.connect(connection_uri)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        sql_string = f"SELECT * FROM public.talk2data_table_description WHERE table_name = '{table_name}'"
        cur.execute(sql_string)
        rows = cur.fetchall()

    # Group columns into a single dictionary
    result = {"table_name": table_name, "columns": []}
    for row in rows:
        column = {
            "column_name": row["column_name"],
            "column_description": row["column_description"],
            "column_datatype": row["column_datatype"],
            "column_is_null": row["column_is_null"],
        }
        result["columns"].append(column)

    return result


def list_table_descriptions():
    global DB_ALIAS

    # Check if the cache is valid
    if DB_ALIAS is not None and os.getenv('INVALIDATE_CACHE', 'false').lower() != 'true':
        return DB_ALIAS

    connection_uri = f"postgresql://{talk2data_config_db_user_id}:{talk2data_config_db_password}@{talk2data_config_db_host_name}:{talk2data_config_db_port_no}/{talk2data_config_db_database_name}"
    conn = psycopg2.connect(connection_uri)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Fetch data from talk2data_db_alias
        cur.execute("SELECT * FROM public.talk2data_db_alias")
        db_alias_data = cur.fetchall()

        # Fetch data from talk2data_table_description
        cur.execute("SELECT * FROM public.talk2data_table_description")
        table_description_data = cur.fetchall()

        # Initialize the list to store the data
    db_alias = []

    # Iterate over the db_alias_data
    for row in db_alias_data:
        # Create a dictionary for each row
        row_dict = dict(row)

        # Get the corresponding table descriptions
        table_descriptions = [desc_row for desc_row in table_description_data if
                              desc_row['db_alias'] == row['db_alias'] and desc_row['table_name'] == row['table_name']]

        # Add the table descriptions to the row dictionary
        row_dict['table_descriptions'] = [{key: val for key, val in desc_row.items() if
                                           key in ['column_name', 'column_description', 'column_is_null',
                                                   'column_datatype']} for desc_row in table_descriptions]

        # Add the row dictionary to the db_alias list
        db_alias.append(row_dict)

    # Store the fetched data in the global variable
    DB_ALIAS = db_alias

    # Set the INVALIDATE_CACHE environment variable back to 'false'
    os.environ['INVALIDATE_CACHE'] = 'false'

    return db_alias


def update_table_description_in_db(table_name, table_update):
    connection_uri = f"postgresql://{talk2data_config_db_user_id}:{talk2data_config_db_password}@{talk2data_config_db_host_name}:{talk2data_config_db_port_no}/{talk2data_config_db_database_name}"
    conn = psycopg2.connect(connection_uri)
    with conn.cursor() as cur:
        # Update table description in talk2data_db_alias table
        cur.execute("""  
            UPDATE public.talk2data_db_alias SET table_description = %s, vector_db_collection_name = %s, sql_tips = %s WHERE table_name = %s  
        """, (
        table_update.table_description, table_update.vector_db_collection_name, table_update.sql_tips, table_name))

        # Insert or update each column in talk2data_table_description table
        for column in table_update.columns:
            cur.execute("""  
                INSERT INTO public.talk2data_table_description (table_name, column_name, column_datatype, column_is_null, column_description)  
                VALUES (%s, %s, %s, %s, %s)  
                ON CONFLICT (table_name, column_name) DO UPDATE  
                SET column_datatype = %s, column_is_null = %s, column_description = %s  
            """, (
                table_name, column.column_name, column.column_datatype, column.column_is_null,
                column.column_description,
                column.column_datatype, column.column_is_null, column.column_description))

            # Commit the changes
        conn.commit()

    # Set the INVALIDATE_CACHE environment variable back to 'true'
    os.environ['INVALIDATE_CACHE'] = 'true'        

    return {"message": "Table description updated successfully"}


def insert_log_entry(w3_id, db_alias, question, model_name, model_provider, hint, prompt):
    connection_uri = f"postgresql://{talk2data_config_db_user_id}:{talk2data_config_db_password}@{talk2data_config_db_host_name}:{talk2data_config_db_port_no}/{talk2data_config_db_database_name}"
    conn = psycopg2.connect(connection_uri)
    current_datetime = datetime.now()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO talk2data_logs (w3_id, db_alias, question, model_name, model_provider, hint, prompt, submitted_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING question_id
        """, (w3_id, db_alias, question, model_name, model_provider, hint, prompt, current_datetime))

        question_id = cur.fetchone()[0]
    conn.commit()
    return question_id


def update_log_entry(question_id, response='', sql_generation_status='', sql_query='',
                     sql_result_url='', input_token_count=0, generated_token_count=0):
    connection_uri = f"postgresql://{talk2data_config_db_user_id}:{talk2data_config_db_password}@{talk2data_config_db_host_name}:{talk2data_config_db_port_no}/{talk2data_config_db_database_name}"
    conn = psycopg2.connect(connection_uri)
    current_datetime = datetime.now()
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE talk2data_logs
            SET sql_query = %s, responded_at = %s, input_token_count = %s, generated_token_count = %s, sql_generation_status = %s, sql_result_url = %s
            WHERE question_id = %s
        """, (sql_query, current_datetime, input_token_count,
              generated_token_count, sql_generation_status, sql_result_url, question_id))

    conn.commit()


def search_log(db_alias=None, model_name=None, model_provider=None, w3_id=None, from_time=None, to_time=None,
               sql_generation_status=None, input_token_count=None, generated_token_count=None):
    connection_uri = f"postgresql://{talk2data_config_db_user_id}:{talk2data_config_db_password}@{talk2data_config_db_host_name}:{talk2data_config_db_port_no}/{talk2data_config_db_database_name}"
    conn = psycopg2.connect(connection_uri)
    with conn.cursor() as cur:
        query = """  
            SELECT db_alias, question, sql_query, model_name, model_provider, w3_id, sql_result_url,  
                   submitted_at, hint, prompt, responded_at, sql_generation_status, input_token_count,   
                   generated_token_count, question_id  
            FROM talk2data_logs  
            WHERE 1=1  
        """
        params = []

        if db_alias:
            query += " AND db_alias = %s"
            params.append(db_alias)

        if model_name:
            query += " AND model_name = %s"
            params.append(model_name)

        if model_provider:
            query += " AND model_provider = %s"
            params.append(model_provider)

        if w3_id:
            query += " AND w3_id = %s"
            params.append(w3_id)

        if from_time and to_time:
            query += " AND submitted_at BETWEEN %s AND %s"
            params.append(from_time)
            params.append(to_time)

        if sql_generation_status:
            query += " AND sql_generation_status = %s"
            params.append(sql_generation_status)

        if input_token_count is not None:
            query += " AND input_token_count = %s"
            params.append(input_token_count)

        if generated_token_count is not None:
            query += " AND generated_token_count = %s"
            params.append(generated_token_count)

        cur.execute(query, params)

        records = cur.fetchall()
        conn.close()

        return records
