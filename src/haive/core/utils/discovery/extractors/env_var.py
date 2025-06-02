"""
Environment variable extraction utilities.

Extracts potential environment variables from source code.
"""

import logging
import os
import re
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class EnvironmentVariableExtractor:
    """Extracts environment variables from source code."""

    def extract(self, source_code: Optional[str]) -> Set[str]:
        """
        Extract potential environment variables from source code.

        Args:
            source_code: Source code to extract from, or None

        Returns:
            Set of environment variable names
        """
        if not source_code:
            return set()

        # Patterns for environment variables
        patterns = [
            r'os\.environ\.get\(["\']([A-Za-z0-9_]+)["\']',  # os.environ.get('VAR_NAME')
            r'os\.getenv\(["\']([A-Za-z0-9_]+)["\']',  # os.getenv('VAR_NAME')
            r'os\.environ\[["\']([A-Za-z0-9_]+)["\']',  # os.environ['VAR_NAME']
            r'getenv\(["\']([A-Za-z0-9_]+)["\']',  # getenv('VAR_NAME')
            r'ENV\[["\']([A-Za-z0-9_]+)["\']',  # ENV['VAR_NAME']
            r'env\.["\']([A-Za-z0-9_]+)["\']',  # env.'VAR_NAME'
            r'config\[["\']([A-Za-z0-9_]+)["\']',  # config['VAR_NAME']
            r'dotenv\.get\(["\']([A-Za-z0-9_]+)["\']',  # dotenv.get('VAR_NAME')
            r"\.env\.([A-Z][A-Z0-9_]+)",  # .env.API_KEY
            r'["\']([A-Z][A-Z0-9_]+_(?:KEY|TOKEN|SECRET|PASSWORD|ID|URL|URI|ENDPOINT|CREDENTIALS))["\']',  # 'API_KEY'
            r'Field\(.*?default=.*?os\.environ\.get\(["\']([A-Za-z0-9_]+)["\']',  # Field(default=os.environ.get('VAR'))
        ]

        env_vars = set()
        for pattern in patterns:
            try:
                matches = re.findall(pattern, source_code)
                env_vars.update(matches)
            except Exception as e:
                logger.warning(f"Error matching pattern {pattern}: {e}")

        return env_vars

    def check_availability(self, env_vars: Set[str]) -> Dict[str, bool]:
        """
        Check which environment variables are available.

        Args:
            env_vars: Set of environment variable names to check

        Returns:
            Dictionary mapping variable names to availability
        """
        return {var: var in os.environ for var in env_vars}

    def get_available_vars(self, env_vars: Set[str]) -> Set[str]:
        """
        Get subset of environment variables that are available.

        Args:
            env_vars: Set of environment variable names to check

        Returns:
            Set of available variable names
        """
        return {var for var in env_vars if var in os.environ}

    def get_missing_vars(self, env_vars: Set[str]) -> Set[str]:
        """
        Get subset of environment variables that are missing.

        Args:
            env_vars: Set of environment variable names to check

        Returns:
            Set of missing variable names
        """
        return {var for var in env_vars if var not in os.environ}

    def extract_description(self, source_code: Optional[str]) -> Dict[str, str]:
        """
        Extract descriptions for environment variables from docstrings or comments.

        Args:
            source_code: Source code to extract from, or None

        Returns:
            Dictionary mapping variable names to descriptions
        """
        if not source_code:
            return {}

        # Extract variables first
        env_vars = self.extract(source_code)

        # Look for descriptions in comments near variable usage
        descriptions = {}

        for var in env_vars:
            # Look for common docstring/comment patterns
            patterns = [
                rf"{var}.*?#\s*(.*?)$",  # VAR_NAME # Description
                rf"['\"]?{var}['\"]?.*?description.*?['\"]([^'\"]+)['\"]",  # "VAR_NAME", description="Description"
                rf"# {var}: (.*?)$",  # # VAR_NAME: Description
            ]

            for pattern in patterns:
                try:
                    matches = re.findall(pattern, source_code, re.MULTILINE)
                    if matches:
                        descriptions[var] = matches[0].strip()
                        break
                except Exception:
                    pass

        return descriptions
