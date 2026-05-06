import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import hashlib
import shutil

from utils import FAISSVectorStore, VietnameseEmbedder, process_pdf, load_embeddings_with_cache
from crawler import crawl_url
from langchain_core.documents import Document
from runtime_paths import BASE_DIR, RESOURCE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_DIR = RESOURCE_DIR / "pdfs"
HTML_DIR = RESOURCE_DIR / "html"
CONFIG_FILE = RESOURCE_DIR / "config.json"
SESSION_DIR = RESOURCE_DIR / "sessions"
USER_DIR = RESOURCE_DIR / "users"

# Ensure directories exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(USER_DIR, exist_ok=True)
if not CONFIG_FILE.exists():
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"urls": []}, f)


def _safe_session_id(session_id: Optional[str]) -> Optional[str]:
    if not session_id:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", str(session_id).strip())
    return cleaned or None


def _safe_user_id(user_id: Optional[str]) -> Optional[str]:
    if not user_id:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", str(user_id).strip())
    return cleaned or None


def _resolve_scope(session_id: Optional[str] = None, user_id: Optional[str] = None) -> Tuple[str, Optional[str], Optional[str]]:
    safe_user = _safe_user_id(user_id)
    safe_session = _safe_session_id(session_id)
    if safe_user:
        return "user", safe_user, safe_session
    if safe_session:
        return "session", None, safe_session
    return "global", None, None

class ResourceLoader:
    def __init__(self, vector_store: FAISSVectorStore = None):
        self.vector_store = vector_store
        self.loaded_resources: Set[str] = set()  # global scope for backward compatibility
        self.loaded_resources_by_session: Dict[str, Set[str]] = {}
        self.loaded_resources_by_user: Dict[str, Set[str]] = {}

    def set_vector_store(self, store: FAISSVectorStore):
        self.vector_store = store

    def reset_loaded_state(self):
        self.loaded_resources = set()
        self.loaded_resources_by_session = {}
        self.loaded_resources_by_user = {}

    def _scope_dirs(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Tuple[Path, Path, Path]:
        scope, safe_user, safe_session = _resolve_scope(session_id=session_id, user_id=user_id)
        if scope == "global":
            return PDF_DIR, HTML_DIR, CONFIG_FILE
        root = USER_DIR / safe_user if scope == "user" and safe_user else SESSION_DIR / safe_session
        pdf_dir = root / "pdfs"
        html_dir = root / "html"
        config_file = root / "config.json"
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(html_dir, exist_ok=True)
        if not config_file.exists():
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump({"urls": []}, f, ensure_ascii=False)
        return pdf_dir, html_dir, config_file

    def _scope_prefix(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        scope, safe_user, safe_session = _resolve_scope(session_id=session_id, user_id=user_id)
        if scope == "user" and safe_user:
            return f"user::{safe_user}"
        if scope == "session" and safe_session:
            return f"session::{safe_session}"
        return "global"

    def _resource_id(self, name: str, session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        prefix = self._scope_prefix(session_id=session_id, user_id=user_id)
        if prefix == "global":
            return name
        return f"{prefix}::{name}"

    def _resource_ui_id(self, name: str, session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        scope, safe_user, safe_session = _resolve_scope(session_id=session_id, user_id=user_id)
        if scope == "user" and safe_user:
            return f"user::{safe_user}::{name}"
        if scope == "session" and safe_session:
            return f"session::{safe_session}::{name}"
        return f"global::{name}"

    def _parse_resource_ui_id(
        self,
        resource_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str], str]:
        rid = str(resource_id or "").strip()
        if rid.startswith("global::"):
            return None, None, rid[len("global::"):]
        if rid.startswith("user::"):
            remainder = rid[len("user::"):]
            if "::" in remainder:
                parsed_user, name = remainder.split("::", 1)
                return _safe_user_id(parsed_user), None, name
        if rid.startswith("session::"):
            remainder = rid[len("session::"):]
            if "::" in remainder:
                parsed_session, name = remainder.split("::", 1)
                return None, _safe_session_id(parsed_session), name
        scope, safe_user, safe_session = _resolve_scope(session_id=session_id, user_id=user_id)
        if scope == "user":
            return safe_user, None, rid
        if scope == "session":
            return None, safe_session, rid
        return None, None, rid

    def get_loaded_resource_ids(
        self,
        session_id: Optional[str] = None,
        include_global: bool = True,
        user_id: Optional[str] = None,
    ) -> Set[str]:
        result: Set[str] = set()
        if include_global:
            result.update(self.loaded_resources)
        scope, safe_user, safe_session = _resolve_scope(session_id=session_id, user_id=user_id)
        if scope == "user" and safe_user:
            result.update(self.loaded_resources_by_user.get(safe_user, set()))
        elif scope == "session" and safe_session:
            result.update(self.loaded_resources_by_session.get(safe_session, set()))
        return result

    def list_scope_resource_ids(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Set[str]:
        """
        Return all resource ids currently present on disk/config for a scope.
        """
        safe_session = _safe_session_id(session_id)
        safe_user = _safe_user_id(user_id)
        pdf_dir, html_dir, _ = self._scope_dirs(session_id=safe_session, user_id=safe_user)

        ids: Set[str] = set()
        for pdf_file in pdf_dir.glob("*.pdf"):
            ids.add(self._resource_id(pdf_file.name, session_id=safe_session, user_id=safe_user))
        for html_file in list(html_dir.glob("*.html")) + list(html_dir.glob("*.htm")):
            ids.add(self._resource_id(html_file.name, session_id=safe_session, user_id=safe_user))

        config = self._load_config(session_id=safe_session, user_id=safe_user)
        for url_entry in config.get("urls", []):
            url = str((url_entry or {}).get("url") or "").strip()
            if not url:
                continue
            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
            pseudo_file = f"url_{url_hash}"
            ids.add(self._resource_id(pseudo_file, session_id=safe_session, user_id=safe_user))
        return ids

    def mark_scope_loaded(
        self,
        resource_ids: Set[str],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """
        Mark resources as loaded for the given scope. Useful when a vector snapshot
        already contains those resources and we want to avoid re-ingesting.
        """
        if not resource_ids:
            return
        scope_type, safe_user, safe_session = _resolve_scope(session_id=session_id, user_id=user_id)
        if scope_type == "user" and safe_user:
            loaded_set = self.loaded_resources_by_user.setdefault(safe_user, set())
        elif scope_type == "session" and safe_session:
            loaded_set = self.loaded_resources_by_session.setdefault(safe_session, set())
        else:
            loaded_set = self.loaded_resources
        loaded_set.update(resource_ids)

    def get_scope_signature(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        """
        Compute a deterministic signature for resources within a scope.
        Uses file names + size + mtime_ns and configured URL list.
        """
        safe_session = _safe_session_id(session_id)
        safe_user = _safe_user_id(user_id)
        pdf_dir, html_dir, _ = self._scope_dirs(session_id=safe_session, user_id=safe_user)

        signature_parts: List[str] = []
        for pdf_file in sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.name.lower()):
            stat = pdf_file.stat()
            signature_parts.append(f"pdf|{pdf_file.name}|{stat.st_size}|{getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9))}")
        for html_file in sorted(
            list(html_dir.glob("*.html")) + list(html_dir.glob("*.htm")),
            key=lambda p: p.name.lower(),
        ):
            stat = html_file.stat()
            signature_parts.append(
                f"html|{html_file.name}|{stat.st_size}|{getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9))}"
            )

        config = self._load_config(session_id=safe_session, user_id=safe_user)
        urls = sorted(
            str((entry or {}).get("url") or "").strip()
            for entry in config.get("urls", [])
            if str((entry or {}).get("url") or "").strip()
        )
        for url in urls:
            signature_parts.append(f"url|{url}")

        raw = "\n".join(signature_parts)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _load_config(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict:
        _, _, config_file = self._scope_dirs(session_id=session_id, user_id=user_id)
        if not config_file.exists():
            return {"urls": []}
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"urls": []}

    def _save_config(self, config: Dict, session_id: Optional[str] = None, user_id: Optional[str] = None):
        _, _, config_file = self._scope_dirs(session_id=session_id, user_id=user_id)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def load_resources(self, session_id: Optional[str] = None, user_id: Optional[str] = None):
        """
        Scan resources and ingest into vector store.
        - session_id=None: load global resources (legacy behavior)
        - session_id=<id>: load only that session's resources
        """
        if not self.vector_store:
            logger.warning("[ResourceLoader] Vector store not set. Skipping load.")
            return

        scope_type, safe_user, safe_session = _resolve_scope(session_id=session_id, user_id=user_id)
        pdf_dir, html_dir, _ = self._scope_dirs(session_id=safe_session, user_id=safe_user)
        owner_id = safe_user if scope_type == "user" else safe_session
        scope = "global" if scope_type == "global" else f"{scope_type}:{owner_id}"
        logger.info("[ResourceLoader] strict loading resources for %s...", scope)
        added_any = False

        if scope_type == "user" and safe_user:
            loaded_set = self.loaded_resources_by_user.setdefault(safe_user, set())
        elif safe_session:
            loaded_set = self.loaded_resources_by_session.setdefault(safe_session, set())
        else:
            loaded_set = self.loaded_resources

        # 1. Load PDFs
        for pdf_file in pdf_dir.glob("*.pdf"):
            file_id = self._resource_id(pdf_file.name, session_id=safe_session, user_id=safe_user)
            if file_id in loaded_set:
                continue

            try:
                logger.info("[ResourceLoader] Ingesting PDF: %s (%s)", pdf_file.name, scope)
                embedder = self.vector_store.embedder
                docs = process_pdf(str(pdf_file))
                for d in docs:
                    d.metadata["is_global_resource"] = scope_type == "global"
                    d.metadata["resource_scope"] = scope_type
                    d.metadata["owner_type"] = scope_type
                    d.metadata["owner_id"] = owner_id
                    d.metadata["session_id"] = safe_session if scope_type == "session" else None
                    d.metadata["user_id"] = safe_user if scope_type == "user" else None
                    d.metadata["file_id"] = file_id
                    d.metadata["file_name"] = pdf_file.name

                embeddings = load_embeddings_with_cache(str(pdf_file), embedder, docs)
                self.vector_store.add_documents_with_embeddings(docs, embeddings, rebuild_index=False)
                if docs:
                    added_any = True
                loaded_set.add(file_id)
            except Exception as e:
                logger.error("[ResourceLoader] Failed to load PDF %s: %s", pdf_file.name, e)

        # 2. Load HTMLs
        for html_file in list(html_dir.glob("*.html")) + list(html_dir.glob("*.htm")):
            file_id = self._resource_id(html_file.name, session_id=safe_session, user_id=safe_user)
            if file_id in loaded_set:
                continue

            try:
                logger.info("[ResourceLoader] Ingesting HTML: %s (%s)", html_file.name, scope)
                docs = crawl_url(str(html_file))
                for d in docs:
                    d.metadata["is_global_resource"] = scope_type == "global"
                    d.metadata["resource_scope"] = scope_type
                    d.metadata["owner_type"] = scope_type
                    d.metadata["owner_id"] = owner_id
                    d.metadata["session_id"] = safe_session if scope_type == "session" else None
                    d.metadata["user_id"] = safe_user if scope_type == "user" else None
                    d.metadata["file_id"] = file_id
                    d.metadata["file_name"] = html_file.name

                if docs:
                    embedder = self.vector_store.embedder
                    embeddings = load_embeddings_with_cache(str(html_file), embedder, docs)
                    self.vector_store.add_documents_with_embeddings(docs, embeddings, rebuild_index=False)
                    added_any = True
                    loaded_set.add(file_id)
            except Exception as e:
                logger.error("[ResourceLoader] Failed to load HTML %s: %s", html_file.name, e)

        # 3. Load URLs
        config = self._load_config(session_id=safe_session, user_id=safe_user)
        urls = config.get("urls", [])
        for url_entry in urls:
            url = url_entry.get("url")
            if not url:
                continue

            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
            pseudo_file = f"url_{url_hash}"
            pseudo_id = self._resource_id(pseudo_file, session_id=safe_session, user_id=safe_user)
            if pseudo_id in loaded_set:
                continue

            try:
                logger.info("[ResourceLoader] Crawling URL: %s (%s)", url, scope)
                docs = crawl_url(url)
                for d in docs:
                    d.metadata["is_global_resource"] = scope_type == "global"
                    d.metadata["resource_scope"] = scope_type
                    d.metadata["owner_type"] = scope_type
                    d.metadata["owner_id"] = owner_id
                    d.metadata["session_id"] = safe_session if scope_type == "session" else None
                    d.metadata["user_id"] = safe_user if scope_type == "user" else None
                    d.metadata["file_id"] = pseudo_id
                    d.metadata["file_name"] = url

                if docs:
                    self.vector_store.add_documents(docs, rebuild_index=False)
                    added_any = True
                    loaded_set.add(pseudo_id)
            except Exception as e:
                logger.error("[ResourceLoader] Failed to crawl %s: %s", url, e)

        if added_any:
            try:
                if hasattr(self.vector_store, "rebuild_index"):
                    self.vector_store.rebuild_index()
                elif hasattr(self.vector_store, "_rebuild_index"):
                    self.vector_store._rebuild_index()
            except Exception as e:
                logger.error("[ResourceLoader] Failed to rebuild vector index for %s: %s", scope, e)

    def add_pdf(self, file_path: str, original_filename: str, session_id: Optional[str] = None, user_id: Optional[str] = None):
        """
        Moves a temp file to resource dir and ingests it.
        """
        scope_type, safe_user, safe_session = _resolve_scope(session_id=session_id, user_id=user_id)
        pdf_dir, _, _ = self._scope_dirs(session_id=safe_session, user_id=safe_user)
        target_path = pdf_dir / original_filename
        # Avoid overwrite or handle versioning? For now overwrite.
        shutil.copy(file_path, target_path)
        
        # Trigger load single
        self.load_resources(session_id=safe_session, user_id=safe_user)

    def add_html(self, file_path: str, original_filename: str, session_id: Optional[str] = None, user_id: Optional[str] = None):
        """
        Moves a temp file to resource html dir and ingests it.
        """
        scope_type, safe_user, safe_session = _resolve_scope(session_id=session_id, user_id=user_id)
        _, html_dir, _ = self._scope_dirs(session_id=safe_session, user_id=safe_user)
        target_path = html_dir / original_filename
        shutil.copy(file_path, target_path)
        
        # Trigger load single
        self.load_resources(session_id=safe_session, user_id=safe_user)

    def add_url(self, url: str, session_id: Optional[str] = None, user_id: Optional[str] = None):
        """
        Adds URL to config and ingests.
        """
        scope_type, safe_user, safe_session = _resolve_scope(session_id=session_id, user_id=user_id)
        config = self._load_config(session_id=safe_session, user_id=safe_user)
        # Check duplicate
        if any(u["url"] == url for u in config["urls"]):
            logger.info(f"URL already exists: {url}")
            # If it exists but not loaded (e.g. restart), we still want to load it?
            # 'load_resources' handles unloaded ones. 
            # If we call add_url, we imply we want to add to config AND load.
            # If already in config, proceed to try loading just in case.
        else:
            config["urls"].append({"url": url, "added_at": str(logging.Formatter().converter())})
            self._save_config(config, session_id=safe_session, user_id=safe_user)
        
        # Trigger load single
        self.load_resources(session_id=safe_session, user_id=safe_user)

    def get_resources(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict]:
        """
        Returns list of available resources.
        """
        scope_type, safe_user, safe_session = _resolve_scope(session_id=session_id, user_id=user_id)
        scope_specs: List[Tuple[Optional[str], Optional[str], str]] = [(None, None, "global")]
        if scope_type == "user" and safe_user:
            scope_specs.append((safe_user, None, "user"))
        elif scope_type == "session" and safe_session:
            scope_specs.append((None, safe_session, "session"))

        res = []
        for scope_user, scope_session, scope_name in scope_specs:
            pdf_dir, html_dir, _ = self._scope_dirs(session_id=scope_session, user_id=scope_user)

            for p in pdf_dir.glob("*.pdf"):
                res.append(
                    {
                        "type": "pdf",
                        "name": p.name,
                        "id": self._resource_ui_id(p.name, session_id=scope_session, user_id=scope_user),
                        "scope": scope_name,
                        "owner_type": scope_name,
                        "owner_id": scope_user or scope_session,
                    }
                )

            for h in list(html_dir.glob("*.html")) + list(html_dir.glob("*.htm")):
                res.append(
                    {
                        "type": "html",
                        "name": h.name,
                        "id": self._resource_ui_id(h.name, session_id=scope_session, user_id=scope_user),
                        "scope": scope_name,
                        "owner_type": scope_name,
                        "owner_id": scope_user or scope_session,
                    }
                )

            config = self._load_config(session_id=scope_session, user_id=scope_user)
            for u in config.get("urls", []):
                url_value = u["url"]
                url_hash = f"url_{hashlib.md5(url_value.encode('utf-8')).hexdigest()}"
                res.append(
                    {
                        "type": "url",
                        "name": url_value,
                        "id": self._resource_ui_id(url_hash, session_id=scope_session, user_id=scope_user),
                        "scope": scope_name,
                        "owner_type": scope_name,
                        "owner_id": scope_user or scope_session,
                    }
                )

        return res

    def delete_resource(self, resource_id: str, session_id: Optional[str] = None, user_id: Optional[str] = None):
        """
        Deletes a resource (PDF or URL) and removes it from config/disk.
        """
        safe_user, safe_session, normalized_resource_id = self._parse_resource_ui_id(
            resource_id,
            session_id=session_id,
            user_id=user_id,
        )
        scope_type, _, _ = _resolve_scope(session_id=safe_session, user_id=safe_user)
        pdf_dir, html_dir, _ = self._scope_dirs(session_id=safe_session, user_id=safe_user)
        owner_id = safe_user if scope_type == "user" else safe_session
        scope_label = f"{scope_type}:{owner_id}" if owner_id else "global"
        logger.info(f"[ResourceLoader] Deleting resource: {normalized_resource_id} (%s)", scope_label)
        
        # 1. Check if PDF
        pdf_path = pdf_dir / normalized_resource_id
        if pdf_path.exists() and normalized_resource_id.endswith(".pdf"):
            try:
                os.remove(pdf_path)
                logger.info("[ResourceLoader] Removed PDF file: %s (%s)", normalized_resource_id, scope_label)
                
                # REMOVE CACHE FILES
                from utils import CACHE_DIR
                cache_file = CACHE_DIR / f"{normalized_resource_id}.pkl"
                cache_meta = CACHE_DIR / f"{normalized_resource_id}_metadata.pkl"
                # Also embeddings
                emb_cache = CACHE_DIR / f"{normalized_resource_id}_embeddings.npy"
                emb_meta = CACHE_DIR / f"{normalized_resource_id}_embeddings_meta.json"
                
                for f in [cache_file, cache_meta, emb_cache, emb_meta]:
                    if f.exists():
                        try:
                             os.remove(f)
                             logger.info(f"[ResourceLoader] Removed cache file: {f.name}")
                        except Exception as ce:
                             logger.warning(f"Failed to remove cache {f.name}: {ce}")
                             
                loaded_id = self._resource_id(normalized_resource_id, session_id=safe_session, user_id=safe_user)
                if scope_type == "user" and safe_user:
                    loaded_set = self.loaded_resources_by_user.setdefault(safe_user, set())
                    loaded_set.discard(loaded_id)
                elif safe_session:
                    loaded_set = self.loaded_resources_by_session.setdefault(safe_session, set())
                    loaded_set.discard(loaded_id)
                else:
                    self.loaded_resources.discard(loaded_id)
                return True
            except Exception as e:
                logger.error(f"[ResourceLoader] Failed to delete PDF {normalized_resource_id}: {e}")
                return False

        # 2. Check if HTML
        html_path = html_dir / normalized_resource_id
        if html_path.exists() and (normalized_resource_id.endswith(".html") or normalized_resource_id.endswith(".htm")):
            try:
                os.remove(html_path)
                logger.info("[ResourceLoader] Removed HTML file: %s (%s)", normalized_resource_id, scope_label)
                loaded_id = self._resource_id(normalized_resource_id, session_id=safe_session, user_id=safe_user)
                if scope_type == "user" and safe_user:
                    loaded_set = self.loaded_resources_by_user.setdefault(safe_user, set())
                    loaded_set.discard(loaded_id)
                elif safe_session:
                    loaded_set = self.loaded_resources_by_session.setdefault(safe_session, set())
                    loaded_set.discard(loaded_id)
                else:
                    self.loaded_resources.discard(loaded_id)
                return True
            except Exception as e:
                logger.error(f"[ResourceLoader] Failed to delete HTML {normalized_resource_id}: {e}")
                return False

        # 2. Check if URL
        if normalized_resource_id.startswith("url_"):
            config = self._load_config(session_id=safe_session, user_id=safe_user)
            original_len = len(config["urls"])
            # Filter out the URL matching the hash
            config["urls"] = [
                u for u in config["urls"] 
                if f"url_{hashlib.md5(u['url'].encode('utf-8')).hexdigest()}" != normalized_resource_id
            ]
            
            if len(config["urls"]) < original_len:
                self._save_config(config, session_id=safe_session, user_id=safe_user)
                logger.info("[ResourceLoader] Removed URL from config: %s (%s)", normalized_resource_id, scope_label)
                loaded_id = self._resource_id(normalized_resource_id, session_id=safe_session, user_id=safe_user)
                if scope_type == "user" and safe_user:
                    loaded_set = self.loaded_resources_by_user.setdefault(safe_user, set())
                    loaded_set.discard(loaded_id)
                elif safe_session:
                    loaded_set = self.loaded_resources_by_session.setdefault(safe_session, set())
                    loaded_set.discard(loaded_id)
                else:
                    self.loaded_resources.discard(loaded_id)
                return True
        
        logger.warning(f"[ResourceLoader] Resource not found: {normalized_resource_id}")
        return False

# Singleton instance?
resource_loader = ResourceLoader()
