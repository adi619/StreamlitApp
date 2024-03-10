import os
from dotenv import load_dotenv

load_dotenv()

class Config:

    confluence_url = os.getenv('confluence_url')
    confluence_username = os.getenv('confluence_username')
    confluence_api_key = os.getenv('confluence_api_key')
    aws_region_name = os.getenv('aws_region_name')
    aws_access_key_id = os.getenv('aws_access_key_id')
    aws_secret_access_key = os.getenv('aws_secret_access_key')
    aws_llm_model_id="anthropic.claude-v2"
    llm_temperature = 0
    llm_max_tokens_to_sample = 2048
