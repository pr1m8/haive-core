#!/usr/bin/env python3
"""
Test script to verify AutoAPI namespace fix configuration
This validates the settings before running the full build
"""

import sys
from pathlib import Path

# Add the project to Python path
project_root = Path(__file__).parent
packages_dir = project_root / "packages"
sys.path.insert(0, str(packages_dir / "haive-core/src"))

print("🧪 AutoAPI Namespace Fix Configuration Test")
print("=" * 50)

# Test 1: Verify package structure
print("\n1. 📁 Package Structure Check:")
haive_init = packages_dir / "haive-core/src/haive/__init__.py"
core_init = packages_dir / "haive-core/src/haive/core/__init__.py"

if haive_init.exists():
    print(f"✅ Found namespace package: {haive_init}")
else:
    print(f"❌ Missing namespace package: {haive_init}")

if core_init.exists():
    print(f"✅ Found core package: {core_init}")
else:
    print(f"❌ Missing core package: {core_init}")

# Test 2: Test import paths
print("\n2. 🐍 Import Path Test:")
try:
    import haive
    print(f"✅ Successfully imported haive: {haive.__path__}")
except ImportError as e:
    print(f"❌ Failed to import haive: {e}")

try:
    import haive.core
    print("✅ Successfully imported haive.core")
except ImportError as e:
    print(f"❌ Failed to import haive.core: {e}")

# Test 3: AutoAPI directory configuration
print("\n3. ⚙️  AutoAPI Configuration Test:")
autoapi_dirs_correct = [str(packages_dir / "haive-core/src/haive")]
autoapi_dirs_incorrect = [str(packages_dir / "haive-core/src")]

print(f"✅ Correct autoapi_dirs: {autoapi_dirs_correct}")
print(f"❌ Incorrect autoapi_dirs: {autoapi_dirs_incorrect}")

# Test 4: Module discovery simulation
print("\n4. 🔍 Module Discovery Simulation:")
haive_dir = packages_dir / "haive-core/src/haive"
if haive_dir.exists():
    print(f"📂 Contents of {haive_dir}:")
    for item in sorted(haive_dir.iterdir()):
        if item.is_dir():
            print(f"   📁 {item.name}/")
            # Check for core subdirectories
            if item.name == "core":
                core_dir = item
                print("     📂 Contents of core/:")
                for core_item in sorted(core_dir.iterdir())[:5]:  # Limit output
                    if core_item.is_dir():
                        print(f"       📁 {core_item.name}/")

# Test 5: Provider module test
print("\n5. 🔧 Provider Module Test:")
try:
    from haive.core.models.llm.providers import get_provider, list_providers
    providers = list_providers()
    print(f"✅ LLM providers accessible: {len(providers)} found")
    print(f"   Sample providers: {providers[:3]}")
except ImportError as e:
    print(f"❌ Provider import failed: {e}")

try:
    from haive.core.engine.embedding.providers import OpenAIEmbeddingConfig
    print("✅ Embedding providers accessible: OpenAIEmbeddingConfig")
except ImportError as e:
    print(f"❌ Embedding provider import failed: {e}")

# Test 6: Namespace package test
print("\n6. 🏷️  Namespace Package Test:")
import pkgutil

for importer, modname, ispkg in pkgutil.walk_packages(
    path=[str(packages_dir / "haive-core/src/haive")],
    prefix="haive.",
    onerror=lambda x: None
):
    if ispkg and modname.count(".") <= 2:  # Limit depth
        print(f"   📦 {modname}")

print("\n🎯 Configuration Summary:")
print("=" * 30)
print(f"AutoAPI dirs should point to: {packages_dir / 'haive-core/src/haive'}")
print(f"sys.path should include: {packages_dir / 'haive-core/src'}")
print("autoapi_python_use_implicit_namespaces should be: True")
print("Expected module discovery: haive.core.*, not core.*")

print("\n✅ Configuration test complete! Run the build script to test documentation generation.")
