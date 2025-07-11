from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Fotogrametria aplicada"
    DEBUG: bool = False
    IMAGE_STORAGE_PATH: str = "./resources/output/images"

    class Config:
        env_file = ".env"