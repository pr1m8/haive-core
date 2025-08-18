#!/bin/bash
# Enhanced documentation build script for haive-core

echo "🚀 Building enhanced haive-core documentation..."
echo "================================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Clean previous build
echo -e "${BLUE}📧 Cleaning previous build...${NC}"
rm -rf docs/build/
rm -rf docs/source/autoapi/

# Build documentation
echo -e "${PURPLE}🔨 Building documentation with Sphinx...${NC}"
poetry run sphinx-build -b html docs/source docs/build

# Check if build was successful
if [ $? -eq 0 ]; then
	echo -e "${GREEN}✅ Documentation built successfully!${NC}"
	echo ""
	echo -e "${PURPLE}🎨 Features included:${NC}"
	echo "  - 🎨 Purple/violet Furo theme (light & dark modes)"
	echo "  - 📇 Sphinx-design cards with emojis"
	echo "  - 🔧 Enhanced AutoAPI with better organization"
	echo "  - 💫 Interactive tooltips with sphinx-tippy"
	echo "  - 📊 Mermaid diagrams support"
	echo "  - 🔍 36+ Sphinx extensions integrated"
	echo ""
	echo -e "${GREEN}📂 Documentation location:${NC}"
	echo "  file://$PWD/docs/build/index.html"
	echo ""
	echo -e "${BLUE}🌐 To serve locally:${NC}"
	echo "  cd docs/build && python -m http.server 8000"
	echo "  Then visit: http://localhost:8000"
else
	echo -e "${RED}❌ Documentation build failed!${NC}"
	exit 1
fi
