from pathlib import Path

# Automatically resolve root relative to this file (should be inside `src/haive_core/config`)
CONFIG_DIR = Path(__file__).resolve().parent
HAIVE_CORE_DIR = CONFIG_DIR.parent
SRC_DIR = HAIVE_CORE_DIR.parent
ROOT_DIR = SRC_DIR.parent.parent  # one more up for the polyrepo root (haive/)

# Important shared folders
RESOURCES_DIR = ROOT_DIR / "resources"
CACHE_DIR = ROOT_DIR / "lc_cache"
GRAPH_IMAGES_DIR = RESOURCES_DIR / "graph_images"
EMBEDDINGS_CACHE_DIR = RESOURCES_DIR / "embeddings_cache"

# Optional project-specific folders
PROJECTS_DIR = SRC_DIR / "projects"
DOCUMENTS_DIR = SRC_DIR / "documents"
VECTORSTORE_DIR = SRC_DIR / "vectorstore"
AGENTS_DIR = SRC_DIR / "haive_core" / "agents"

# Create necessary dirs if not already present
for path in [
    RESOURCES_DIR,
    CACHE_DIR,
    GRAPH_IMAGES_DIR,
    EMBEDDINGS_CACHE_DIR,
    PROJECTS_DIR,
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    AGENTS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Expose constants
__all__ = [
    "AGENTS_DIR",
    "CACHE_DIR",
    "DOCUMENTS_DIR",
    "EMBEDDINGS_CACHE_DIR",
    "GRAPH_IMAGES_DIR",
    "PROJECTS_DIR",
    "RESOURCES_DIR",
    "ROOT_DIR",
    "SRC_DIR",
    "VECTORSTORE_DIR",
]
