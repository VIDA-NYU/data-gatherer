import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

PORTKEY_GATEWAY_URL: str = os.getenv("PORTKEY_GATEWAY_URL")
PORTKEY_API_KEY: str = os.getenv("PORTKEY_API_KEY")
PORTKEY_ROUTE: str = os.getenv("PORTKEY_ROUTE")