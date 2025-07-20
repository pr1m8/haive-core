from __future__ import annotations

import logging
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    """Enum of source types for document loading."""

    # Web-based sources
    URL = "url"
    WEB = "web"
    SITEMAP = "sitemap"
    RECURSIVE_URL = "recursive_url"
    READTHEDOCS = "readthedocs"
    RSS = "rss"
    WIKIPEDIA = "wikipedia"
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    BROWSERLESS = "browserless"
    BROWSERBASE = "browserbase"
    BRAVE_SEARCH = "brave_search"
    COLLEGE_CONFIDENTIAL = "college_confidential"
    HACKER_NEWS = "hn"
    NEWS_URL = "news_url"
    FIRECRAWL = "firecrawl"
    SCRAPFLY = "scrapfly"
    SCRAPINGANT = "scrapingant"

    # File-based sources
    FILE = "file"
    DIRECTORY = "directory"
    PDF = "pdf"
    PDF_PYMUPDF = "pdf_pymupdfloader"
    PDF_PDFMINER = "pdf_pdfminerloader"
    PDF_PDFPLUMBER = "pdf_pdfplumberloader"
    PDF_UNSTRUCTURED = "pdf_unstructuredpdfloader"
    PDF_PYPDFIUM2 = "pdf_pypdfium2loader"
    CSV = "csv"
    DOCX = "docx"
    EXCEL = "excel"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    XML = "xml"
    SRT = "srt"
    RTF = "rtf"
    EPUB = "epub"
    RST = "rst"
    ODT = "odt"
    TOML = "toml"
    PYTHON = "python"
    NOTEBOOK = "notebook"
    BIBTEX = "bibtex"
    VSDX = "vsdx"
    MHTML = "mhtml"
    CHM = "chm"
    PPTX = "pptx"
    PPT = "ppt"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    GIF = "gif"
    SVG = "svg"
    TIFF = "tiff"
    WEBP = "webp"

    # Repositories
    GITHUB = "github"
    GITHUB_ISSUES = "github_issues"
    GIT = "git"
    GITBOOK = "gitbook"
    CONFLUENCE = "confluence"
    NOTION = "notion"
    NOTION_DIRECTORY = "notion_directory"
    TRELLO = "trello"
    AIRTABLE = "airtable"
    JOPLIN = "joplin"
    OBSIDIAN = "obsidian"
    DOCUSAURUS = "docusaurus"
    QUIP = "quip"
    ROAM = "roam"
    LAKEFS = "lakefs"

    # Media sources
    IMAGE = "image"
    IMAGE_CAPTION = "image_caption"
    AUDIO = "audio"
    YOUTUBE = "youtube"
    YOUTUBE_AUDIO = "youtube_audio"
    IMDB = "imdb"
    IFIXIT = "ifixit"
    FIGMA = "figma"
    BILIBILI = "bilibili"

    # Database sources
    DATABASE = "database"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    ASTRADB = "astradb"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    DUCKDB = "duckdb"
    ATHENA = "athena"
    CASSANDRA = "cassandra"
    COUCHBASE = "couchbase"
    ORACLE = "oracle"
    ORACLE_ADB = "oracle_adb"
    ORACLE_AI = "oracle_ai"
    POSTGRESQL = "postgresql"
    TIDB = "tidb"
    ROCKSET = "rockset"
    MAXCOMPUTE = "maxcompute"
    FAUNA = "fauna"
    KINETICA = "kinetica"
    SURREALDB = "surrealdb"
    GLUE_CATALOG = "glue_catalog"
    DATAFRAME = "dataframe"
    POLARS_DATAFRAME = "polars_dataframe"
    PYSPARK_DATAFRAME = "pyspark_dataframe"
    GEODATAFRAME = "geodataframe"
    XORBITS = "xorbits"
    CUBE_SEMANTIC = "cube_semantic"

    # Social and platforms
    EMAIL = "email"
    OUTLOOK = "outlook_message"
    SLACK = "slack"
    TELEGRAM = "telegram"
    TELEGRAM_CHAT_API = "telegram_chat_api"
    DISCORD = "discord"
    FACEBOOK = "facebook"
    WHATSAPP = "whatsapp"
    TWITTER = "twitter"
    MASTODON = "mastodon"
    REDDIT = "reddit"
    CHATGPT = "chatgpt"
    EVERNOTE = "evernote"
    ETHERSCAN = "etherscan"
    SPREADLY = "spreadly"
    STRIPE = "stripe"
    IUGU = "iugu"
    MODERN_TREASURY = "modern_treasury"
    DATADOG_LOGS = "datadog_logs"
    BLOCKCHAIN = "blockchain"
    MINTBASE = "mintbase"
    LARKSUITE_DOC = "larksuite_doc"
    PSYCHIC = "psychic"

    # Cloud storage
    AZURE_BLOB = "azure_blob"
    S3 = "s3"
    GCS = "gcs"
    GOOGLE_DRIVE = "google_drive"
    ONEDRIVE = "onedrive"
    ONEDRIVE_FILE = "onedrive_file"
    DROPBOX = "dropbox"
    SHAREPOINT = "sharepoint"
    OBS_DIRECTORY = "obs_directory"
    OBS_FILE = "obs_file"
    TENCENT_COS_DIRECTORY = "tencent_cos_directory"
    TENCENT_COS_FILE = "tencent_cos_file"
    BAIDU_BOS_DIRECTORY = "baidu_bos_directory"
    BAIDU_BOS_FILE = "baidu_bos_file"

    # Education
    BLACKBOARD = "blackboard"
    ONENOTE = "onenote"

    # Literature
    GUTENBERG = "gutenberg"
    MEDIAWIKI = "mediawiki"
    AZLYRICS = "azlyrics"

    # Research and AI
    PUBMED = "pubmed"
    ARXIV = "arxiv"
    HUGGINGFACE_MODEL = "huggingface_model"
    HUGGINGFACE_DATASET = "huggingface_dataset"
    TENSORFLOW_DATASET = "tensorflow_dataset"
    ARCGIS = "arcgis"
    OPEN_CITY_DATA = "open_city_data"
    NEEDLE = "needle"
    DOCUGAMI = "docugami"
    LLM_SHERPA = "llm_sherpa"

    # Other
    WEATHER = "weather"
    API = "api"
    UNSTRUCTURED = "unstructured"
    UNSTRUCTURED_API = "unstructured_api"
    DEDOC = "dedoc"
    DEDOC_API = "dedoc_api"
    APIFY = "apify"
    PEBBLO = "pebblo"
    CONCURRENT = "concurrent"
    DIFFBOT = "diffbot"
    AZURE_AI_DATA = "azure_ai_data"
    AZURE_AI_DOCUMENT = "azure_ai_document"
    ACREOM = "acreom"
    SPIDER = "spider"
    CONLLU = "conllu"
    MERGED_DATA = "merged_data"
    GOOGLE_SPEECH_TO_TEXT = "google_speech_to_text"
    ASSEMBLYAI_AUDIO = "assemblyai_audio"
    TO_MARKDOWN = "to_markdown"
    YUQUE = "yuque"


# Base Source Classes

__all__ = [
    "ACREOM",
    "AIRTABLE",
    "API",
    "APIFY",
    "ARCGIS",
    "ARXIV",
    "ASSEMBLYAI_AUDIO",
    "ASTRADB",
    "ATHENA",
    "AUDIO",
    "AZLYRICS",
    "AZURE_AI_DATA",
    "AZURE_AI_DOCUMENT",
    "AZURE_BLOB",
    "BAIDU_BOS_DIRECTORY",
    "BAIDU_BOS_FILE",
    "BIBTEX",
    "BIGQUERY",
    "BILIBILI",
    "BLACKBOARD",
    "BLOCKCHAIN",
    "BRAVE_SEARCH",
    "BROWSERBASE",
    "BROWSERLESS",
    "CASSANDRA",
    "CHATGPT",
    "CHM",
    "COLLEGE_CONFIDENTIAL",
    "CONCURRENT",
    "CONFLUENCE",
    "CONLLU",
    "COUCHBASE",
    "CSV",
    "CUBE_SEMANTIC",
    "DATABASE",
    "DATADOG_LOGS",
    "DATAFRAME",
    "DEDOC",
    "DEDOC_API",
    "DIFFBOT",
    "DIRECTORY",
    "DISCORD",
    "DOCUGAMI",
    "DOCUSAURUS",
    "DOCX",
    "DROPBOX",
    "DUCKDB",
    "EMAIL",
    "EPUB",
    "ETHERSCAN",
    "EVERNOTE",
    "EXCEL",
    "FACEBOOK",
    "FAUNA",
    "FIGMA",
    "FILE",
    "FIRECRAWL",
    "GCS",
    "GEODATAFRAME",
    "GIF",
    "GIT",
    "GITBOOK",
    "GITHUB",
    "GITHUB_ISSUES",
    "GLUE_CATALOG",
    "GOOGLE_DRIVE",
    "GOOGLE_SPEECH_TO_TEXT",
    "GUTENBERG",
    "HACKER_NEWS",
    "HTML",
    "HUGGINGFACE_DATASET",
    "HUGGINGFACE_MODEL",
    "IFIXIT",
    "IMAGE",
    "IMAGE_CAPTION",
    "IMDB",
    "IUGU",
    "JOPLIN",
    "JPEG",
    "JPG",
    "JSON",
    "KINETICA",
    "LAKEFS",
    "LARKSUITE_DOC",
    "LLM_SHERPA",
    "MARKDOWN",
    "MASTODON",
    "MAXCOMPUTE",
    "MEDIAWIKI",
    "MERGED_DATA",
    "MHTML",
    "MINTBASE",
    "MODERN_TREASURY",
    "MONGODB",
    "NEEDLE",
    "NEWS_URL",
    "NOTEBOOK",
    "NOTION",
    "NOTION_DIRECTORY",
    "OBSIDIAN",
    "OBS_DIRECTORY",
    "OBS_FILE",
    "ODT",
    "ONEDRIVE",
    "ONEDRIVE_FILE",
    "ONENOTE",
    "OPEN_CITY_DATA",
    "ORACLE",
    "ORACLE_ADB",
    "ORACLE_AI",
    "OUTLOOK",
    "PDF",
    "PDF_PDFMINER",
    "PDF_PDFPLUMBER",
    "PDF_PYMUPDF",
    "PDF_PYPDFIUM2",
    "PDF_UNSTRUCTURED",
    "PEBBLO",
    "PLAYWRIGHT",
    "PNG",
    "POLARS_DATAFRAME",
    "POSTGRESQL",
    "PPT",
    "PPTX",
    "PSYCHIC",
    "PUBMED",
    "PYSPARK_DATAFRAME",
    "PYTHON",
    "QUIP",
    "READTHEDOCS",
    "RECURSIVE_URL",
    "REDDIT",
    "ROAM",
    "ROCKSET",
    "RSS",
    "RST",
    "RTF",
    "S3",
    "SCRAPFLY",
    "SCRAPINGANT",
    "SELENIUM",
    "SHAREPOINT",
    "SITEMAP",
    "SLACK",
    "SNOWFLAKE",
    "SPIDER",
    "SPREADLY",
    "SQLITE",
    "SRT",
    "STRIPE",
    "SURREALDB",
    "SVG",
    "TELEGRAM",
    "TELEGRAM_CHAT_API",
    "TENCENT_COS_DIRECTORY",
    "TENCENT_COS_FILE",
    "TENSORFLOW_DATASET",
    "TIDB",
    "TIFF",
    "TOML",
    "TO_MARKDOWN",
    "TRELLO",
    "TWITTER",
    "UNSTRUCTURED",
    "UNSTRUCTURED_API",
    "URL",
    "VSDX",
    "WEATHER",
    "WEB",
    "WEBP",
    "WHATSAPP",
    "WIKIPEDIA",
    "XML",
    "XORBITS",
    "YOUTUBE",
    "YOUTUBE_AUDIO",
    "YUQUE",
    "SourceType",
]
