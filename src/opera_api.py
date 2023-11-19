from datetime import datetime
import json
import requests
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi.security import HTTPBearer
from fastapi import FastAPI, Depends, HTTPException, status, APIRouter, Body
import uvicorn
from typing import List, Optional
from dotenv import load_dotenv
import os
import logging
import csv
from boxsdk import Client, OAuth2
from src.chroma import *
from src.postgres_db import *
from src.build_prompt import *


# Create routers for different API groups
golden_records_router = APIRouter()
authentication_router = APIRouter()
question_router = APIRouter()
table_descriptions_router = APIRouter()
logs_router = APIRouter()

load_dotenv()
opera_api_token = os.getenv('OPERA_API_TOKEN')
wx_api_key = os.getenv('WX_API_KEY')
wx_api_url = os.getenv('WX_API_URL')
wx_project_id = os.getenv('WX_PROJECT_ID')
bam_api_key = os.getenv('BAM_API_KEY')
bam_api_url = os.getenv('BAM_API_URL')
opera_users_box_file_id = os.getenv('OPERA_USERS_BOX_FILE_ID')
box_api_token = os.getenv('BOX_API_TOKEN')
opera_users_file_name = os.getenv('OPERA_USERS_FILE_NAME')
sql_result_download_directory = os.getenv('POSTGRES_SQL_RESULT_DOWNLOAD_PATH')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

app = FastAPI()
bearer_scheme = HTTPBearer()

async def authenticate_token(token: str = Depends(bearer_scheme)):
    if token.credentials != opera_api_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid credentials"
        )
    return token


class LogInput(BaseModel):
    db_alias: Optional[str] = None
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None
    model_name: Optional[str] = None
    model_provider: Optional[str] = None
    w3_id: Optional[str] = None
    sql_generation_status: Optional[str] = None
    input_token_count: Optional[int] = None
    generated_token_count: Optional[int] = None


class LogOutput(BaseModel):
    db_alias: Optional[str] = None
    question: Optional[str] = None
    sql_query: Optional[str] = None
    model_name: Optional[str] = None
    model_provider: Optional[str] = None
    w3_id: Optional[str] = None
    sql_result_url: Optional[str] = None
    submitted_at: Optional[datetime] = None
    hint: Optional[str] = None
    prompt: Optional[str] = None
    responded_at: Optional[datetime] = None
    sql_generation_status: Optional[str] = None
    response: Optional[str] = None
    input_token_count: Optional[int] = None
    generated_token_count: Optional[int] = None
    question_id: Optional[int] = None


class Question(BaseModel):
    db_alias: str = Field(..., description="DB Alias against which the question is to be answered")
    question: str = Field(..., description="The question to answer")
    w3_id: str = Field(..., description="w3 id of the user")

class Record(BaseModel):
    db_alias: str = Field(..., description="DB Alias against which the golden record is to be added")
    question: str = Field(..., description="The question to be added in the golden record")
    sql_query: str = Field(..., description="The SQL query to be added in the golden record")

class GoldenRecords(BaseModel):
    records: List[Record]

class QueryGoldenRecord(BaseModel):
    db_alias: str = Field(..., description="DB Alias against which the question is to be queried")
    question: str = Field(..., description="The question which is to be queried in the golden records")
    result_count: int = Field(..., description="The no.of matches which are to be returned")

class Column(BaseModel):
    column_name: str
    column_description: str
    column_datatype: str
    column_is_null: bool

class TableDescription(BaseModel):
    table_description: str
    vector_db_collection_name:str
    sql_tips:str
    columns: List[Column]

@golden_records_router.post("/api/v1/golden-records")
async def add_golden_records(records: GoldenRecords, token: str = Depends(authenticate_token)):
    """
    Add Golden Records (Question and SQL Query Pair in the collection. Question gets embedded and SQL Query is saved as ids in collection.
    """
    for record in records.records:
        add_record(id=record.sql_query, document=record.question, db_alias=record.db_alias)
    return {"message": "Records added successfully"}

@golden_records_router.post("/api/v1/golden-record")
async def get_golden_record(query: QueryGoldenRecord, token: str = Depends(authenticate_token)):
    """
    Query Golden Records (SQL Queries) in the collection for the given question
    """
    return query_record(query_texts=query.question, num_results=query.result_count, db_alias=query.db_alias)
    # return [{'sql_query': result['id'], 'score': result['score']} for result in results]

@golden_records_router.get("/api/v1/golden-records/{num_records}")
async def list_all_golden_records(db_alias: str, num_records: int, token: str = Depends(authenticate_token)):
    """
    Get all Golden Records - Question and SQL Queries pair from Chroma DB
    """
    response = []
    results = get_all_records(db_alias=db_alias, nCount=num_records)
    ids = results.get('ids', [])
    documents = results.get('documents', [])
    for i in range(len(ids)):
        response.append({
            'question': documents[i],
            'sql_query': ids[i],
            'db_alias': db_alias
        })
    return response

@golden_records_router.delete("/api/v1/golden-record/{sql_query}")
async def delete_golden_record(db_alias: str, sql_query: str, token: str = Depends(authenticate_token)):
    """
    Delete the Golden Records by passing the SQL Query
    """
    delete_record(db_alias=db_alias, id=sql_query)
    return {"message": "Record deleted successfully"}

@question_router.post("/api/v1/invalidate-cache")
async def invalidate_cache(token: str = Depends(authenticate_token)):
    """
    Invalidate cache.
    """
    # Set the INVALIDATE_CACHE environment variable back to 'true'
    os.environ['INVALIDATE_CACHE'] = 'true'    

@question_router.post("/api/v1/question")
async def answer_question(question: Question, token: str = Depends(authenticate_token)):
    """
    Answer a question by generating a SQL query and executing it.
    """
    # Get the list_table_descriptions data
    db_alias_data = list_table_descriptions()

    # Find the matching db_alias
    matching_db_alias = next((item for item in db_alias_data if item["db_alias"] == question.db_alias), None)

    if matching_db_alias is None:
        raise HTTPException(status_code=400, detail="Invalid db_alias")

    # Get the model_name and model_provider from the matching_db_alias
    model_name = matching_db_alias["llm_name"]
    model_provider = matching_db_alias["llm_provider"]
    num_hints = matching_db_alias["llm_num_hints"]

    hint = get_hint(db_alias=question.db_alias, question=question.question, num_results=num_hints)

    prompt = f"{get_prompt(question.db_alias)}\n{hint}question: {question.question}\nsql_query:"
    logger.info(f'Prompt: {prompt}')
    question_id = insert_log_entry(question=question.question, hint=hint, prompt=prompt, model_name=model_name, model_provider=model_provider, w3_id=question.w3_id, db_alias=question.db_alias)

    if model_provider == 'wx':
        from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
        from ibm_watson_machine_learning.foundation_models import Model
        from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
        parameters = {
            GenParams.DECODING_METHOD: matching_db_alias["llm_decoding_method"],
            GenParams.MAX_NEW_TOKENS: matching_db_alias["llm_max_new_tokens"],
            GenParams.MIN_NEW_TOKENS: matching_db_alias["llm_min_new_tokens"],
            GenParams.STOP_SEQUENCES: [matching_db_alias["llm_stop_sequences"]],
            GenParams.TEMPERATURE: matching_db_alias["llm_temperature"],
            GenParams.TOP_K: matching_db_alias["llm_top_k"],
            GenParams.TOP_P: matching_db_alias["llm_top_p"],
            GenParams.REPETITION_PENALTY: matching_db_alias["llm_repetition_penalty"],
            GenParams.RANDOM_SEED: matching_db_alias["llm_random_seed"]
        }
        credentials = {"url": wx_api_url,"apikey": wx_api_key}
        try:
            model = Model(model_id=getattr(ModelTypes, model_name), credentials=credentials, params=parameters, project_id=wx_project_id)
            generated_response = model.generate(prompt=prompt)
        except Exception as e:
            logger.exception(f"Failed to make request to Watson Machine Learning")
            raise HTTPException(status_code=400, detail=f"Failed to make request to watsonx.ai. Original error: {str(e)}") from e
    elif model_provider == 'bam':
        headers = {'Authorization': f"Bearer {bam_api_key}"}
        json_data = {
            "model_id": model_name,
            "inputs": [prompt],
            "parameters": {
                "decoding_method": matching_db_alias["llm_decoding_method"],
                "max_new_tokens": matching_db_alias["llm_max_new_tokens"],
                "min_new_tokens": matching_db_alias["llm_min_new_tokens"],
                "stop_sequences": [matching_db_alias["llm_stop_sequences"]],
                "temperature": matching_db_alias["llm_temperature"],
                "top_k": matching_db_alias["llm_top_k"],
                "top_p": matching_db_alias["llm_top_p"],
                "typical_p": 1,
            }
        }
        response = requests.post(bam_api_url, json=json_data, headers=headers)
        generated_response = response.json()
    else:
        raise HTTPException(status_code=400, detail="Invalid model provider")


    print('CP 1')
    logger.info(f'Response: {json.dumps(generated_response, indent=2)}')
    sql_query = generated_response["results"][0]["generated_text"]
    sql_query = sql_query.replace('<end of sql>', '').strip()
    logger.info(f'SQL Query: {sql_query}')

    sql_result = execute_sql_query(sql_query)
    logger.info(f'SQL Result: {sql_result}')

    update_log_entry(question_id=question_id, sql_query=sql_query, response=generated_response,
                     input_token_count=generated_response['results'][0]['input_token_count'],
                     generated_token_count=generated_response['results'][0]['generated_token_count'],
                     sql_result_url=sql_result['sql_result_url'], sql_generation_status=sql_result['sql_generation_status'])
    final_response = {"generated_response": generated_response, "sql_result": sql_result}
    return final_response


@authentication_router.post("/api/v1/authenticate-talk2opera")
async def authenticate_talk2opera(w3id: str, token: str = Depends(authenticate_token)):
    """
    Authenticate Opera user by checking if the w3 id exists in a CSV file in the folder
    """
    auth = OAuth2(client_id='', client_secret='', access_token=box_api_token)
    boxClient = Client(auth)
    output_file = open(opera_users_file_name, 'wb')
    boxClient.file(opera_users_box_file_id).download_to(output_file)
    output_file.close()
    with open('opera_users.csv', mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            if w3id in lines:
                return {"authentication": True}
        return {"authentication": False}

@table_descriptions_router.get("/api/v1/table-descriptions")
async def list_all_table_descriptions(token: str = Depends(authenticate_token)):
    """
    List all table descriptions
    """
    return list_table_descriptions()

@table_descriptions_router.get("/api/v1/table-description/{table_name}")
async def get_table_description(table_name: str, token: str = Depends(authenticate_token)):
    """
    Get table description
    """
    return get_table_description_from_db(table_name)

@table_descriptions_router.patch("/api/v1/table-descriptions/{table_name}")
async def update_table_description(table_name: str, table_update: TableDescription = Body(...), token: str = Depends(authenticate_token)):
    """
    Update table description for the given table name
    """
    return update_table_description_in_db(table_name,table_update)


@logs_router.post("/api/v1/log", response_model=List[LogOutput])
async def get_logs(log_input: LogInput, token: str = Depends(authenticate_token)):
    """
    Get logs from talk2opera_logs table in DB for the given input
    """
    records = search_log(db_alias=log_input.db_alias, model_name=log_input.model_name, model_provider=log_input.model_provider, w3_id=log_input.w3_id,
                         from_time=log_input.from_time, to_time=log_input.to_time, sql_generation_status=log_input.sql_generation_status,
                         input_token_count=log_input.input_token_count, generated_token_count=log_input.generated_token_count)
    return [LogOutput(db_alias=r[0], question=r[1], sql_query=r[2], model_name=r[3], model_provider=r[4], w3_id=r[5],
                      sql_result_url=r[6], nl_response=r[7], submitted_at=r[8], hint=r[9], prompt=r[10], responded_at=r[11],
                      sql_generation_status=r[12], response=r[13], input_token_count=r[14], generated_token_count=r[15],
                      question_id=r[16]) for r in records]

# Include the routers in FastAPI application
app.include_router(golden_records_router, tags=["Golden Records"])
app.include_router(table_descriptions_router, tags=["Table Descriptions"])
app.include_router(question_router, tags=["Question"])
app.include_router(logs_router, tags=["Logs"])
app.include_router(authentication_router, tags=["Authentication"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
