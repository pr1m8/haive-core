#!/bin/bash
# Script to clean and rebuild haive-core documentation (background server version)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Log file for background server
LOG_FILE="/tmp/haive-core-docs-server.log"
PID_FILE="/tmp/haive-core-docs-server.pid"

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

	# Kill any existing server on port 8005
	if lsof -Pi :8005 -sTCP:LISTEN -t >/dev/null; then
		echo -e "\n${YELLOW}⚠️  Killing existing server on port 8005...${NC}"
		kill $(lsof -t -i:8005) 2>/dev/null
		sleep 1
	fi

	# Start server in background
	echo -e "\n${GREEN}📡 Starting documentation server in background on port 8005...${NC}"
	nohup python -m http.server 8005 --directory docs/build/html >"$LOG_FILE" 2>&1 &
	echo $! >"$PID_FILE"

	sleep 1

	# Check if server started successfully
	if lsof -Pi :8005 -sTCP:LISTEN -t >/dev/null; then
		echo -e "${GREEN}✓ Server started successfully!${NC}"
		echo ""
		echo "Documentation available at: http://localhost:8005"
		echo "Server PID: $(cat $PID_FILE)"
		echo "Server log: $LOG_FILE"
		echo ""
		echo "To stop the server: kill \$(cat $PID_FILE)"
		echo "To view logs: tail -f $LOG_FILE"

		# Open in browser
		echo -e "\n${GREEN}🌐 Opening documentation in browser...${NC}"
		xdg-open "http://localhost:8005" 2>/dev/null || echo "Please open http://localhost:8005 in your browser"
	else
		echo -e "${RED}✗ Failed to start server!${NC}"
		echo "Check the log file: $LOG_FILE"
		exit 1
	fi
else
	echo -e "\n${RED}✗ Documentation build failed!${NC}"
	echo "Check the error messages above for details."
	exit 1
fi
