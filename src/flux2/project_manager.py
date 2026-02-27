"""
Project Manager: Session-based projects with version history, collections, and sharing.
Handles generation log, favorites, albums, version rollback, and expiring share links.
"""

import json
import uuid
import secrets
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from enum import Enum


class ProjectStatus(Enum):
    """Project status."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    TRASH = "trash"


class CollectionType(Enum):
    """Collection types."""
    ALBUM = "album"
    SERIES = "series"
    FAVORITES = "favorites"
    CUSTOM = "custom"


@dataclass
class GenerationSnapshot:
    """Single generation record within a project."""
    id: str
    timestamp: datetime
    prompt: str
    model: str
    seed: int
    guidance: float
    num_steps: int
    image_path: Path
    metadata: Dict = field(default_factory=dict)
    version: int = 1
    tags: List[str] = field(default_factory=list)
    starred: bool = False
    notes: str = ""
    parent_version_id: Optional[str] = None  # For version history


@dataclass
class GenerationVersion:
    """Version of a generation (for history/rollback)."""
    version_id: str
    generation_id: str
    timestamp: datetime
    snapshot: GenerationSnapshot
    change_description: str = "Initial generation"


@dataclass
class Collection:
    """Collection/album of generations."""
    id: str
    name: str
    type: CollectionType
    created: datetime
    description: str = ""
    generation_ids: List[str] = field(default_factory=list)
    cover_image_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    public: bool = False


@dataclass
class SharingLink:
    """Expiring sharing link for projects."""
    link_id: str
    project_id: str
    created: datetime
    expires: datetime
    access_token: str
    max_views: Optional[int] = None
    views_count: int = 0
    includes_metadata: bool = True
    includes_prompts: bool = False
    password_protected: bool = False


@dataclass
class Project:
    """Session-based project containing generations and metadata."""
    id: str
    name: str
    created: datetime
    modified: datetime
    owner_id: str = ""
    status: ProjectStatus = ProjectStatus.ACTIVE
    description: str = ""
    
    # Contents
    generation_ids: List[str] = field(default_factory=list)
    collections: Dict[str, Collection] = field(default_factory=dict)
    favorites: Set[str] = field(default_factory=set)
    
    # Metadata
    generation_log: Dict[str, GenerationSnapshot] = field(default_factory=dict)
    version_history: Dict[str, List[GenerationVersion]] = field(default_factory=dict)
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Sharing
    sharing_links: Dict[str, SharingLink] = field(default_factory=dict)
    
    # Statistics
    total_generations: int = 0
    total_starred: int = 0


class ProjectManager:
    """Manager for projects with version control and sharing."""
    
    def __init__(self, projects_dir: Path = Path("projects")):
        self.projects_dir = projects_dir
        self.projects_dir.mkdir(exist_ok=True)
        
        self.projects: Dict[str, Project] = {}
        self.current_project: Optional[Project] = None
        
        self._load_projects()
    
    def create_project(
        self,
        name: str,
        description: str = "",
        owner_id: str = "default"
    ) -> Project:
        """Create new project."""
        project = Project(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            owner_id=owner_id,
            created=datetime.now(),
            modified=datetime.now()
        )
        
        self.projects[project.id] = project
        self.current_project = project
        
        self._save_project(project)
        
        return project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        return self.projects.get(project_id)
    
    def list_projects(self, status: Optional[ProjectStatus] = None) -> List[Project]:
        """List projects, optionally filtered by status."""
        projects = list(self.projects.values())
        
        if status:
            projects = [p for p in projects if p.status == status]
        
        return sorted(projects, key=lambda p: p.modified, reverse=True)
    
    def rename_project(self, project_id: str, new_name: str) -> bool:
        """Rename project."""
        project = self.get_project(project_id)
        if not project:
            return False
        
        project.name = new_name
        project.modified = datetime.now()
        self._save_project(project)
        return True
    
    def delete_project(self, project_id: str, permanent: bool = False) -> bool:
        """Delete or trash project."""
        project = self.get_project(project_id)
        if not project:
            return False
        
        if permanent:
            del self.projects[project_id]
            project_file = self.projects_dir / f"{project_id}.json"
            if project_file.exists():
                project_file.unlink()
        else:
            project.status = ProjectStatus.TRASH
            project.modified = datetime.now()
            self._save_project(project)
        
        if self.current_project and self.current_project.id == project_id:
            self.current_project = None
        
        return True
    
    # Generation management
    
    def add_generation(
        self,
        project_id: str,
        generation: GenerationSnapshot
    ) -> bool:
        """Add generation to project."""
        project = self.get_project(project_id)
        if not project:
            return False
        
        project.generation_log[generation.id] = generation
        project.generation_ids.append(generation.id)
        project.total_generations += 1
        project.modified = datetime.now()
        
        # Initialize version history
        if generation.id not in project.version_history:
            version = GenerationVersion(
                version_id=f"v1_{generation.id}",
                generation_id=generation.id,
                timestamp=generation.timestamp,
                snapshot=generation,
                change_description="Initial generation"
            )
            project.version_history[generation.id] = [version]
        
        self._save_project(project)
        return True
    
    def get_generation(self, project_id: str, generation_id: str) -> Optional[GenerationSnapshot]:
        """Get generation from project."""
        project = self.get_project(project_id)
        if not project:
            return None
        
        return project.generation_log.get(generation_id)
    
    def remove_generation(self, project_id: str, generation_id: str) -> bool:
        """Remove generation from project."""
        project = self.get_project(project_id)
        if not project or generation_id not in project.generation_ids:
            return False
        
        project.generation_ids.remove(generation_id)
        del project.generation_log[generation_id]
        project.total_generations -= 1
        
        if generation_id in project.favorites:
            project.favorites.remove(generation_id)
            project.total_starred -= 1
        
        project.modified = datetime.now()
        self._save_project(project)
        return True
    
    # Favorites and starring
    
    def star_generation(self, project_id: str, generation_id: str) -> bool:
        """Mark generation as favorite."""
        project = self.get_project(project_id)
        if not project or generation_id not in project.generation_ids:
            return False
        
        if generation_id not in project.favorites:
            project.favorites.add(generation_id)
            project.total_starred += 1
            project.modified = datetime.now()
            self._save_project(project)
        
        return True
    
    def unstar_generation(self, project_id: str, generation_id: str) -> bool:
        """Remove favorite marking."""
        project = self.get_project(project_id)
        if not project:
            return False
        
        if generation_id in project.favorites:
            project.favorites.remove(generation_id)
            project.total_starred -= 1
            project.modified = datetime.now()
            self._save_project(project)
        
        return True
    
    def get_starred_generations(self, project_id: str) -> List[GenerationSnapshot]:
        """Get all starred generations."""
        project = self.get_project(project_id)
        if not project:
            return []
        
        return [
            project.generation_log[gen_id]
            for gen_id in project.favorites
            if gen_id in project.generation_log
        ]
    
    # Version history and rollback
    
    def get_version_history(self, project_id: str, generation_id: str) -> List[GenerationVersion]:
        """Get version history for generation."""
        project = self.get_project(project_id)
        if not project or generation_id not in project.version_history:
            return []
        
        return project.version_history[generation_id]
    
    def rollback_to_version(
        self,
        project_id: str,
        generation_id: str,
        version_id: str
    ) -> bool:
        """Rollback to previous version."""
        project = self.get_project(project_id)
        if not project:
            return False
        
        history = project.version_history.get(generation_id, [])
        version = next((v for v in history if v.version_id == version_id), None)
        
        if not version:
            return False
        
        # Create new current version pointing to rolled-back snapshot
        current = GenerationVersion(
            version_id=f"{version_id}_restored",
            generation_id=generation_id,
            timestamp=datetime.now(),
            snapshot=version.snapshot,
            change_description=f"Restored from {version_id}"
        )
        
        project.version_history[generation_id].append(current)
        project.generation_log[generation_id] = current.snapshot
        project.modified = datetime.now()
        
        self._save_project(project)
        return True
    
    def create_new_version(
        self,
        project_id: str,
        generation_id: str,
        new_snapshot: GenerationSnapshot,
        description: str = ""
    ) -> bool:
        """Create new version from modification."""
        project = self.get_project(project_id)
        if not project or generation_id not in project.generation_ids:
            return False
        
        if generation_id not in project.version_history:
            project.version_history[generation_id] = []
        
        version_num = len(project.version_history[generation_id]) + 1
        version = GenerationVersion(
            version_id=f"v{version_num}_{generation_id}",
            generation_id=generation_id,
            timestamp=datetime.now(),
            snapshot=new_snapshot,
            change_description=description or f"Version {version_num}"
        )
        
        project.version_history[generation_id].append(version)
        project.generation_log[generation_id] = new_snapshot
        project.modified = datetime.now()
        
        self._save_project(project)
        return True
    
    # Collections and albums
    
    def create_collection(
        self,
        project_id: str,
        name: str,
        collection_type: CollectionType = CollectionType.CUSTOM,
        description: str = ""
    ) -> Optional[Collection]:
        """Create collection/album."""
        project = self.get_project(project_id)
        if not project:
            return None
        
        collection = Collection(
            id=str(uuid.uuid4())[:8],
            name=name,
            type=collection_type,
            created=datetime.now(),
            description=description
        )
        
        project.collections[collection.id] = collection
        project.modified = datetime.now()
        self._save_project(project)
        
        return collection
    
    def add_to_collection(
        self,
        project_id: str,
        collection_id: str,
        generation_id: str
    ) -> bool:
        """Add generation to collection."""
        project = self.get_project(project_id)
        if not project or collection_id not in project.collections:
            return False
        
        collection = project.collections[collection_id]
        
        if generation_id not in collection.generation_ids:
            collection.generation_ids.append(generation_id)
            project.modified = datetime.now()
            self._save_project(project)
        
        return True
    
    def remove_from_collection(
        self,
        project_id: str,
        collection_id: str,
        generation_id: str
    ) -> bool:
        """Remove generation from collection."""
        project = self.get_project(project_id)
        if not project or collection_id not in project.collections:
            return False
        
        collection = project.collections[collection_id]
        
        if generation_id in collection.generation_ids:
            collection.generation_ids.remove(generation_id)
            project.modified = datetime.now()
            self._save_project(project)
        
        return True
    
    def get_collection(self, project_id: str, collection_id: str) -> Optional[Collection]:
        """Get collection."""
        project = self.get_project(project_id)
        if not project:
            return None
        
        return project.collections.get(collection_id)
    
    def list_collections(self, project_id: str) -> List[Collection]:
        """List all collections in project."""
        project = self.get_project(project_id)
        if not project:
            return []
        
        return list(project.collections.values())
    
    def delete_collection(self, project_id: str, collection_id: str) -> bool:
        """Delete collection (not generations)."""
        project = self.get_project(project_id)
        if not project or collection_id not in project.collections:
            return False
        
        del project.collections[collection_id]
        project.modified = datetime.now()
        self._save_project(project)
        return True
    
    # Sharing links
    
    def create_share_link(
        self,
        project_id: str,
        expires_in_days: int = 30,
        max_views: Optional[int] = None,
        includes_metadata: bool = True,
        includes_prompts: bool = False,
        password: Optional[str] = None
    ) -> Optional[SharingLink]:
        """Create expiring share link."""
        project = self.get_project(project_id)
        if not project:
            return None
        
        share_link = SharingLink(
            link_id=str(uuid.uuid4())[:12],
            project_id=project_id,
            created=datetime.now(),
            expires=datetime.now() + timedelta(days=expires_in_days),
            access_token=secrets.token_urlsafe(32),
            max_views=max_views,
            includes_metadata=includes_metadata,
            includes_prompts=includes_prompts,
            password_protected=password is not None
        )
        
        project.sharing_links[share_link.link_id] = share_link
        project.modified = datetime.now()
        self._save_project(project)
        
        return share_link
    
    def get_share_link(self, project_id: str, link_id: str) -> Optional[SharingLink]:
        """Get share link details."""
        project = self.get_project(project_id)
        if not project:
            return None
        
        share_link = project.sharing_links.get(link_id)
        
        # Check expiration
        if share_link and share_link.expires <= datetime.now():
            return None
        
        # Check view limit
        if share_link and share_link.max_views and share_link.views_count >= share_link.max_views:
            return None
        
        return share_link
    
    def access_share_link(self, project_id: str, link_id: str) -> bool:
        """Record access to share link."""
        share_link = self.get_share_link(project_id, link_id)
        if not share_link:
            return False
        
        share_link.views_count += 1
        
        project = self.get_project(project_id)
        if project:
            project.modified = datetime.now()
            self._save_project(project)
        
        return True
    
    def revoke_share_link(self, project_id: str, link_id: str) -> bool:
        """Revoke share link immediately."""
        project = self.get_project(project_id)
        if not project or link_id not in project.sharing_links:
            return False
        
        del project.sharing_links[link_id]
        project.modified = datetime.now()
        self._save_project(project)
        return True
    
    def list_share_links(self, project_id: str, active_only: bool = True) -> List[SharingLink]:
        """List share links."""
        project = self.get_project(project_id)
        if not project:
            return []
        
        links = list(project.sharing_links.values())
        
        if active_only:
            links = [l for l in links if l.expires > datetime.now()]
        
        return links
    
    # Persistence
    
    def _save_project(self, project: Project) -> None:
        """Save project to disk."""
        try:
            project_file = self.projects_dir / f"{project.id}.json"
            
            # Convert to JSON-serializable format
            project_data = {
                "id": project.id,
                "name": project.name,
                "created": project.created.isoformat(),
                "modified": project.modified.isoformat(),
                "owner_id": project.owner_id,
                "status": project.status.value,
                "description": project.description,
                "generation_ids": project.generation_ids,
                "collections": {
                    cid: asdict(c) | {"created": c.created.isoformat()}
                    for cid, c in project.collections.items()
                },
                "favorites": list(project.favorites),
                "notes": project.notes,
                "tags": project.tags,
                "total_generations": project.total_generations,
                "total_starred": project.total_starred,
                "sharing_links": {
                    lid: asdict(l) | {
                        "created": l.created.isoformat(),
                        "expires": l.expires.isoformat()
                    }
                    for lid, l in project.sharing_links.items()
                }
                # Generation log and version history would be persisted separately
                # to avoid JSON serialization complexity
            }
            
            with open(project_file, 'w') as f:
                json.dump(project_data, f, indent=2)
        
        except Exception as e:
            print(f"Error saving project: {e}")
    
    def _load_projects(self) -> None:
        """Load projects from disk."""
        try:
            for project_file in self.projects_dir.glob("*.json"):
                try:
                    with open(project_file, 'r') as f:
                        project_data = json.load(f)
                    
                    # Reconstruct project
                    project = Project(
                        id=project_data.get("id"),
                        name=project_data.get("name", "Untitled"),
                        created=datetime.fromisoformat(project_data.get("created", datetime.now().isoformat())),
                        modified=datetime.fromisoformat(project_data.get("modified", datetime.now().isoformat())),
                        owner_id=project_data.get("owner_id", "default"),
                        status=ProjectStatus(project_data.get("status", "active")),
                        description=project_data.get("description", ""),
                        generation_ids=project_data.get("generation_ids", []),
                        favorites=set(project_data.get("favorites", [])),
                        notes=project_data.get("notes", ""),
                        tags=project_data.get("tags", []),
                        total_generations=project_data.get("total_generations", 0),
                        total_starred=project_data.get("total_starred", 0)
                    )
                    
                    self.projects[project.id] = project
                
                except Exception as e:
                    print(f"Error loading project {project_file}: {e}")
        
        except Exception as e:
            print(f"Error loading projects: {e}")


def get_project_manager() -> ProjectManager:
    """Get singleton ProjectManager instance."""
    if not hasattr(get_project_manager, '_instance'):
        get_project_manager._instance = ProjectManager()
    return get_project_manager._instance
