#!/bin/bash
# Build documentation with enhanced logging for haive-core
# This script captures all output for debugging Sphinx builds

set -e # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(dirname "$SCRIPT_DIR")"
PACKAGE_ROOT="$(dirname "$DOCS_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$PACKAGE_ROOT")")")"

echo -e "${GREEN}📚 Building haive-core documentation with enhanced logging${NC}"
echo -e "${YELLOW}Working directory: $DOCS_DIR${NC}"

# Create logs directory if it doesn't exist
mkdir -p "$DOCS_DIR/logs/enhanced"

# Set timestamp for this build
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
CONSOLE_LOG="/tmp/haive-core-sphinx-build-${TIMESTAMP}.log"
ENHANCED_LOG="$DOCS_DIR/logs/enhanced/sphinx-build-${TIMESTAMP}.log"

echo -e "${YELLOW}📝 Console output will be saved to: $CONSOLE_LOG${NC}"
echo -e "${YELLOW}📊 Enhanced log will be saved to: $ENHANCED_LOG${NC}"

# Change to docs directory
cd "$DOCS_DIR"

# Clean previous build (optional, comment out to keep)
echo -e "${GREEN}🧹 Cleaning previous build...${NC}"
rm -rf build/html

# Run the build with enhanced logging
echo -e "${GREEN}🔨 Starting Sphinx build...${NC}"

# Check if enhanced logger exists
ENHANCED_LOGGER="$PROJECT_ROOT/haive/tools/pydevelop-docs/scripts/enhanced_build_logger.py"
if [ -f "$ENHANCED_LOGGER" ]; then
	echo -e "${GREEN}✅ Using enhanced build logger${NC}"
	poetry run python "$ENHANCED_LOGGER" \
		--command "poetry run sphinx-build -b html source build/html -v" \
		--output logs/enhanced \
		2>&1 | tee "$CONSOLE_LOG"
else
	echo -e "${YELLOW}⚠️  Enhanced logger not found, using standard sphinx-build${NC}"
	poetry run sphinx-build -b html source build/html -v 2>&1 | tee "$CONSOLE_LOG"
fi

# Check if build succeeded
if [ $? -eq 0 ]; then
	echo -e "${GREEN}✅ Build completed successfully!${NC}"

	# Count HTML files generated
	HTML_COUNT=$(find build/html -name "*.html" 2>/dev/null | wc -l)
	echo -e "${GREEN}📄 Generated $HTML_COUNT HTML files${NC}"

	# Check for index.html
	if [ -f "build/html/index.html" ]; then
		echo -e "${GREEN}✅ index.html generated successfully${NC}"
		echo -e "${YELLOW}🌐 View docs at: file://$DOCS_DIR/build/html/index.html${NC}"
	else
		echo -e "${RED}❌ index.html not found${NC}"
	fi
else
	echo -e "${RED}❌ Build failed! Check logs for details${NC}"
	echo -e "${YELLOW}📋 Console log: $CONSOLE_LOG${NC}"
	echo -e "${YELLOW}📊 Enhanced log: $ENHANCED_LOG${NC}"

	# Show last 20 lines of error
	echo -e "${RED}Last 20 lines of output:${NC}"
	tail -20 "$CONSOLE_LOG"

	exit 1
fi

# Summary
echo -e "\n${GREEN}📊 Build Summary:${NC}"
echo -e "  Console log: $CONSOLE_LOG"
echo -e "  Enhanced log: $ENHANCED_LOG"
echo -e "  HTML output: $DOCS_DIR/build/html/"

# Optional: Run analysis if available
ANALYZER="$PROJECT_ROOT/haive/tools/pydevelop-docs/scripts/analyze_build_log.py"
if [ -f "$ANALYZER" ] && [ -f "$ENHANCED_LOG" ]; then
	echo -e "\n${GREEN}🔍 Running build analysis...${NC}"
	poetry run python "$ANALYZER" "$ENHANCED_LOG" || true
fi

echo -e "\n${GREEN}✨ Done!${NC}"
