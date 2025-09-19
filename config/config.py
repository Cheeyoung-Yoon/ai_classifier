# config/langgraph_config.py
import os
from pathlib import Path
from dotenv import load_dotenv

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent

_loaded_from = None
for p in [
    PROJECT_ROOT / "env/.env.local",
    PROJECT_ROOT / "env/.env",
    CONFIG_DIR / "env/.env.local",
    CONFIG_DIR / "env/.env",
]:
    if p.is_file():
        load_dotenv(p, override=False)
        _loaded_from = str(p)
        break

class Settings:
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")  # ✅ 오타 수정
    
    # 프로젝트 디렉토리 관리 설정
    SAVE_STATE_LOG: bool = os.getenv("SAVE_STATE_LOG", "true").lower() == "true"
    SAVE_TEMP_DATA: bool = os.getenv("SAVE_TEMP_DATA", "true").lower() == "true"
    PROJECT_DATA_BASE_DIR: str = os.getenv("PROJECT_DATA_BASE_DIR", str(PROJECT_ROOT))
    
    # Batch processing settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

    def __init__(self):
        # Remove eager API key validation - now handled lazily by nodes that need it
        pass
    
    def get_openai_api_key(self) -> str:
        """
        Lazy API key retrieval with validation.
        Call this method from nodes that actually need OpenAI API access.
        """
        if not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not loaded. Put it in .env(.local) at project root or config/, "
                "or export it in your shell."
            )
        return self.OPENAI_API_KEY

settings = Settings()
where_loaded = _loaded_from  # for debugging

# Silent operation - no API key status messages
# if bool(settings.OPENAI_API_KEY):
#     logger.debug("API Key loaded successfully.")
# else:
#     logger.warning("API Key not found. Check your .env files or environment variables.")