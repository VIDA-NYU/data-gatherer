import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

PORTKEY_GATEWAY_URL: str = os.getenv("PORTKEY_GATEWAY_URL")
PORTKEY_API_KEY: str = os.getenv("PORTKEY_API_KEY")
PORTKEY_ROUTE: str = os.getenv("PORTKEY_ROUTE")
PORTKEY_CONFIG: str = os.getenv("PORTKEY_CONFIG")
OLLAMA_CLIENT: str = os.getenv("OLLAMA_CLIENT")
ELSEVIER_KEY: str = os.getenv("ELSEVIER_KEY")
MOZ_LOG: str = os.getenv("MOZ_LOG")
MOZ_LOG_FILE: str = os.getenv("MOZ_LOG_FILE")
GPT_API_KEY: str = os.getenv("GPT_API_KEY")
GEMINI_KEY: str = os.getenv("GEMINI_KEY")
CLAUDE_KEY: str = os.getenv("CLAUDE_KEY")
DATA_GATHERER_USER_NAME: str = os.getenv("DATA_GATHERER_USER_NAME")
CACHE_BASE_DIR: str = os.getenv("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "data_gatherer/")
GH_TOKEN: str = os.getenv("GH_TOKEN")
GITHUB_USERNAME: str = os.getenv("GITHUB_USERNAME")
NCBI_API_KEY: str = os.getenv("NCBI_API_KEY")