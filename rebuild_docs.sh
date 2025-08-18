#!/bin/bash
# Script to clean and rebuild haive-core documentation

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}🧹 Cleaning old documentation build...${NC}"
echo "Removing docs/build/"
rm -rf docs/build/
echo "Removing docs/source/autoapi/"
rm -rf docs/source/autoapi/
echo -e "${GREEN}✓ Clean complete${NC}\n"

echo -e "${YELLOW}🔨 Building documentation...${NC}"
poetry run sphinx-build -b html docs/source docs/build/html

# Check if build was successful
if [ $? -eq 0 ]; then
	echo -e "\n${GREEN}✓ Documentation built successfully!${NC}"

	# Check if port 8005 is already in use
	if lsof -Pi :8005 -sTCP:LISTEN -t >/dev/null; then
		echo -e "\n${YELLOW}⚠️  Port 8005 is already in use${NC}"
		echo 'Kill the existing server with: kill $(lsof -t -i:8005)'
		echo "Or use a different port with: python -m http.server 8006 --directory docs/build/html"
	else
		echo -e "\n${GREEN}📡 Starting documentation server on port 8005...${NC}"
		echo "Documentation will be available at: http://localhost:8005"
		echo "Press Ctrl+C to stop the server"
		echo ""
		python -m http.server 8005 --directory docs/build/html
	fi
else
	echo -e "\n${RED}✗ Documentation build failed!${NC}"
	echo "Check the error messages above for details."
	exit 1
fi
