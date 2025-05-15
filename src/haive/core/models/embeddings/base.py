import os
from typing import Any, Dict, Optional

import torch
from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import BaseModel, Field, SecretStr, field_validator

from haive.core.config.constants import EMBEDDINGS_CACHE_DIR
from haive.core.models.embeddings.provider_types import EmbeddingProvider

load_dotenv(".env")


class SecureConfigMixin:
    @field_validator("api_key", mode="after")
    @classmethod
    def resolve_api_key(cls, v, values):
        if v and v.get_secret_value().strip():
            return v

        provider = values.get("provider")
        if not provider:
            return SecretStr("")

        env_map = {
            "azure": "AZURE_OPENAI_API_KEY",
            "huggingface": "HUGGING_FACE_API_KEY",
        }

        env_key = env_map.get(provider.value.lower())
        if env_key:
            key = os.getenv(env_key, "")
            if key.strip():
                return SecretStr(key)
        return SecretStr("")


class BaseEmbeddingConfig(BaseModel, SecureConfigMixin):
    provider: EmbeddingProvider
    model: str
    api_key: SecretStr = Field(default=SecretStr(""))

    def instantiate(self, **kwargs) -> Any:
        raise NotImplementedError


class AzureEmbeddingConfig(BaseEmbeddingConfig):
    provider: EmbeddingProvider = EmbeddingProvider.AZURE

    api_version: str = Field(
        default_factory=lambda: os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
        )
    )
    api_base: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_BASE", "")
    )
    api_type: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_TYPE", "azure")
    )

    def instantiate(self, **kwargs) -> AzureOpenAIEmbeddings:
        return AzureOpenAIEmbeddings(
            model=self.model,
            api_key=self.get_api_key(),
            api_version=self.api_version,
            azure_endpoint=self.api_base,
            **kwargs,
        )

    def get_api_key(self) -> str:
        return self.api_key.get_secret_value()


class HuggingFaceEmbeddingConfig(BaseEmbeddingConfig):
    provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE

    model_kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    )
    model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    encode_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)
    query_encode_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)
    multi_process: bool = False
    cache_folder: Optional[str] = Field(default=str(EMBEDDINGS_CACHE_DIR))
    show_progress: bool = False
    use_cache: bool = True

    def instantiate(self, **kwargs) -> HuggingFaceEmbeddings:
        try:
            embedder = HuggingFaceEmbeddings(
                model_name=self.model,
                model_kwargs=self.model_kwargs,
                encode_kwargs=self.encode_kwargs,
                multi_process=self.multi_process,
                cache_folder=self.cache_folder,
                show_progress=self.show_progress,
                **kwargs,
            )

            if self.use_cache:
                store = LocalFileStore(self.cache_folder)
                return CacheBackedEmbeddings.from_bytes_store(
                    embedder,
                    document_embedding_cache=store,
                    query_embedding_cache=True,
                    namespace=self.model,
                )
            return embedder

        except Exception:
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                embedder = HuggingFaceEmbeddings(
                    model_name=self.model,
                    model_kwargs=self.model_kwargs,
                    encode_kwargs=self.encode_kwargs,
                    multi_process=self.multi_process,
                    cache_folder=self.cache_folder,
                    show_progress=self.show_progress,
                    **kwargs,
                )
                if self.use_cache:
                    store = LocalFileStore(self.cache_folder)
                    return CacheBackedEmbeddings.from_bytes_store(
                        embedder,
                        document_embedding_cache=store,
                        query_embedding_cache=True,
                        namespace=self.model,
                    )
                return embedder
            except Exception as e:
                print(f"Error instantiating HuggingFaceEmbeddings: {e}")
                raise e


# Factory function
def create_embeddings(config: BaseEmbeddingConfig) -> Any:
    return config.instantiate()
