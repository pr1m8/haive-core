"""Prompt template library for reusable prompts.

This module provides a library of reusable prompt templates with versioning
and composition support, extracted from scattered prompt management.
"""

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate, PromptTemplate

from haive.core.contracts.prompt_config import PromptContract, PromptVariable


class PromptTemplate(BaseModel):
    """Versioned prompt template.
    
    Attributes:
        name: Template identifier.
        version: Template version.
        template: The actual prompt template.
        contract: Template contract.
        tags: Categorization tags.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        usage_count: Number of times used.
        parent_version: Parent version if forked.
    """
    
    name: str = Field(..., description="Template identifier")
    version: str = Field(default="1.0.0", description="Template version")
    template: BasePromptTemplate = Field(..., description="Prompt template")
    contract: PromptContract = Field(..., description="Template contract")
    tags: Set[str] = Field(default_factory=set, description="Categorization tags")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Update timestamp")
    usage_count: int = Field(default=0, description="Usage count")
    parent_version: Optional[str] = Field(default=None, description="Parent version")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class PromptCategory(BaseModel):
    """Category of prompt templates.
    
    Attributes:
        name: Category name.
        description: Category description.
        templates: Templates in this category.
        subcategories: Nested categories.
    """
    
    name: str = Field(..., description="Category name")
    description: str = Field(..., description="Category description")
    templates: List[str] = Field(default_factory=list, description="Template names")
    subcategories: List[str] = Field(default_factory=list, description="Subcategory names")


class PromptLibrary(BaseModel):
    """Library of reusable prompt templates.
    
    Provides:
    - Template storage with versioning
    - Category-based organization
    - Template composition
    - Usage tracking
    - Template evolution
    
    Attributes:
        templates: Templates by name and version.
        categories: Template categories.
        tag_index: Templates indexed by tag.
        latest_versions: Latest version of each template.
        composition_rules: Rules for template composition.
    
    Examples:
        Add a template:
            >>> library = PromptLibrary()
            >>> library.add_template(
            ...     name="analysis",
            ...     template=analysis_prompt,
            ...     contract=analysis_contract
            ... )
        
        Get latest version:
            >>> prompt = library.get_latest("analysis")
    """
    
    templates: Dict[str, PromptTemplate] = Field(
        default_factory=dict,
        description="Templates by name:version"
    )
    categories: Dict[str, PromptCategory] = Field(
        default_factory=dict,
        description="Template categories"
    )
    tag_index: Dict[str, Set[str]] = Field(
        default_factory=dict,
        description="Templates indexed by tag"
    )
    latest_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Latest version of each template"
    )
    composition_rules: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Template composition rules"
    )
    
    def add_template(
        self,
        name: str,
        template: BasePromptTemplate,
        contract: PromptContract,
        version: str = "1.0.0",
        tags: Optional[Set[str]] = None,
        category: Optional[str] = None
    ) -> "PromptLibrary":
        """Add a template to the library.
        
        Args:
            name: Template name.
            template: Prompt template.
            contract: Template contract.
            version: Template version.
            tags: Template tags.
            category: Template category.
            
        Returns:
            Self for chaining.
        """
        from datetime import datetime
        
        # Create template record
        template_key = f"{name}:{version}"
        prompt_template = PromptTemplate(
            name=name,
            version=version,
            template=template,
            contract=contract,
            tags=tags or set(),
            created_at=datetime.now().isoformat()
        )
        
        # Store template
        self.templates[template_key] = prompt_template
        
        # Update latest version
        if name not in self.latest_versions or self._compare_versions(version, self.latest_versions[name]) > 0:
            self.latest_versions[name] = version
        
        # Update tag index
        for tag in tags or []:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(template_key)
        
        # Add to category
        if category:
            if category not in self.categories:
                self.categories[category] = PromptCategory(
                    name=category,
                    description=f"Category for {category} templates"
                )
            self.categories[category].templates.append(template_key)
        
        return self
    
    def get_template(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[BasePromptTemplate]:
        """Get a template by name and version.
        
        Args:
            name: Template name.
            version: Template version (latest if not specified).
            
        Returns:
            Template or None.
        """
        if not version:
            version = self.latest_versions.get(name)
            if not version:
                return None
        
        template_key = f"{name}:{version}"
        template_record = self.templates.get(template_key)
        
        if template_record:
            template_record.usage_count += 1
            return template_record.template
        
        return None
    
    def get_latest(self, name: str) -> Optional[BasePromptTemplate]:
        """Get latest version of a template.
        
        Args:
            name: Template name.
            
        Returns:
            Latest template or None.
        """
        return self.get_template(name)
    
    def fork_template(
        self,
        name: str,
        new_name: str,
        new_version: str = "1.0.0",
        modifications: Optional[Dict[str, Any]] = None
    ) -> "PromptLibrary":
        """Fork a template to create a new version.
        
        Args:
            name: Original template name.
            new_name: New template name.
            new_version: New version.
            modifications: Modifications to apply.
            
        Returns:
            Self for chaining.
        """
        # Get original template
        original = self.get_template(name)
        if not original:
            raise ValueError(f"Template '{name}' not found")
        
        original_key = f"{name}:{self.latest_versions[name]}"
        original_record = self.templates[original_key]
        
        # Create forked template
        forked_template = original
        if modifications:
            # Apply modifications (simplified for now)
            if isinstance(original, ChatPromptTemplate) and "messages" in modifications:
                forked_template = ChatPromptTemplate.from_messages(modifications["messages"])
        
        # Add forked template
        self.add_template(
            name=new_name,
            template=forked_template,
            contract=original_record.contract,
            version=new_version,
            tags=original_record.tags,
        )
        
        # Set parent version
        new_key = f"{new_name}:{new_version}"
        self.templates[new_key].parent_version = original_key
        
        return self
    
    def compose_templates(
        self,
        template_names: List[str],
        composed_name: str,
        mode: str = "sequential"
    ) -> BasePromptTemplate:
        """Compose multiple templates into one.
        
        Args:
            template_names: Templates to compose.
            composed_name: Name for composed template.
            mode: Composition mode.
            
        Returns:
            Composed template.
        """
        templates = []
        contracts = []
        
        for name in template_names:
            template = self.get_template(name)
            if template:
                templates.append(template)
                
                # Get contract
                template_key = f"{name}:{self.latest_versions[name]}"
                if template_key in self.templates:
                    contracts.append(self.templates[template_key].contract)
        
        if not templates:
            raise ValueError("No valid templates found")
        
        # Compose templates
        if mode == "sequential":
            # Concatenate templates
            if all(isinstance(t, ChatPromptTemplate) for t in templates):
                messages = []
                for template in templates:
                    messages.extend(template.messages)
                composed = ChatPromptTemplate.from_messages(messages)
            else:
                # String concatenation for other types
                combined = "\n\n".join(str(t.template) for t in templates)
                composed = PromptTemplate.from_template(combined)
        else:
            # Other composition modes can be added
            composed = templates[0]
        
        # Merge contracts
        merged_contract = self._merge_contracts(contracts)
        
        # Add composed template
        self.add_template(
            name=composed_name,
            template=composed,
            contract=merged_contract
        )
        
        # Record composition rule
        self.composition_rules[composed_name] = template_names
        
        return composed
    
    def find_by_tag(self, tag: str) -> List[str]:
        """Find templates by tag.
        
        Args:
            tag: Tag to search for.
            
        Returns:
            List of template keys.
        """
        return list(self.tag_index.get(tag, set()))
    
    def find_by_category(self, category: str) -> List[str]:
        """Find templates by category.
        
        Args:
            category: Category name.
            
        Returns:
            List of template keys.
        """
        if category in self.categories:
            return self.categories[category].templates
        return []
    
    def get_evolution_history(self, name: str) -> List[str]:
        """Get evolution history of a template.
        
        Args:
            name: Template name.
            
        Returns:
            List of versions in order.
        """
        versions = []
        for key in self.templates:
            if key.startswith(f"{name}:"):
                version = key.split(":")[1]
                versions.append(version)
        
        # Sort versions
        versions.sort(key=lambda v: tuple(map(int, v.split("."))))
        return versions
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics.
        
        Returns:
            Usage count by template.
        """
        stats = {}
        for key, template in self.templates.items():
            stats[key] = template.usage_count
        return stats
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings.
        
        Args:
            v1: First version.
            v2: Second version.
            
        Returns:
            1 if v1 > v2, -1 if v1 < v2, 0 if equal.
        """
        v1_parts = tuple(map(int, v1.split(".")))
        v2_parts = tuple(map(int, v2.split(".")))
        
        if v1_parts > v2_parts:
            return 1
        elif v1_parts < v2_parts:
            return -1
        return 0
    
    def _merge_contracts(self, contracts: List[PromptContract]) -> PromptContract:
        """Merge multiple contracts.
        
        Args:
            contracts: Contracts to merge.
            
        Returns:
            Merged contract.
        """
        if not contracts:
            raise ValueError("No contracts to merge")
        
        # Start with first contract
        merged = contracts[0].model_copy()
        
        # Merge variables from all contracts
        all_variables = {}
        for contract in contracts:
            for var in contract.variables:
                if var.name not in all_variables:
                    all_variables[var.name] = var
        
        merged.variables = list(all_variables.values())
        
        # Merge constraints
        all_constraints = set()
        for contract in contracts:
            all_constraints.update(contract.constraints)
        merged.constraints = list(all_constraints)
        
        # Update name and description
        merged.name = f"composed_{len(contracts)}_templates"
        merged.description = f"Composed from {len(contracts)} templates"
        
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Library as dictionary.
        """
        return {
            "templates": {
                key: {
                    "name": template.name,
                    "version": template.version,
                    "tags": list(template.tags),
                    "usage_count": template.usage_count,
                    "parent_version": template.parent_version
                }
                for key, template in self.templates.items()
            },
            "categories": {
                name: category.model_dump()
                for name, category in self.categories.items()
            },
            "latest_versions": self.latest_versions,
            "composition_rules": self.composition_rules,
            "usage_stats": self.get_usage_stats()
        }