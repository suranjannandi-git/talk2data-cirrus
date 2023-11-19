from src.chroma import query_record
from src.postgres_db import *

def get_hint(db_alias, question, num_results):
    matching_records = query_record(query_texts=question, num_results=num_results, db_alias=db_alias)
    # Filter out records with distance less than 1
    # filtered_records = [record for record in matching_records if record['distance'] < 1]
    # construct the hints from the matching records (or filtered records)
    hint = ''
    for record in matching_records:
        hint += f"question: {record['question']}\nsql_query: {record['sql_query']}\n<end of sql>\n\n"
    return hint


def get_prompt(db_alias):
    # Get the list_table_descriptions data
    db_alias_data = list_table_descriptions()

    # Find the matching db_alias
    matching_db_alias = next((item for item in db_alias_data if item["db_alias"] == db_alias), None)

    if matching_db_alias is None:
        raise ValueError(f"No db_alias found for db_alias: {db_alias}")

    # Get the table_name and table_description from the matching_db_alias
    table_name = matching_db_alias["table_name"]
    table_description = matching_db_alias["table_description"]
    sql_tips = matching_db_alias["sql_tips"]
    llm_instruction = matching_db_alias["llm_instruction"]

    # Get the columns from the matching_db_alias
    columns = matching_db_alias["table_descriptions"]

    strPrompt = f""" 
{llm_instruction} {table_description}
CREATE TABLE {table_name} (
"""
    for column in columns:
        column_null = "NULL" if column["column_is_null"] else "NOT NULL"
        strPrompt += f"    {column['column_name']} {column['column_datatype']} {column_null},\n"

    strPrompt = strPrompt.rstrip(",\n") + "\n);\n\n"

    strPrompt += "Each column in the table is described below, which must be taken into consideration while generating the SQL query for the given question.\n"

    for column in columns:
        strPrompt += f"{column['column_name']}: {column['column_description']}\n"

    strPrompt += f"\n{sql_tips}\n"
    return strPrompt

