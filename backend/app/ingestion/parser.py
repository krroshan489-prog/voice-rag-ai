import os
import re
from typing import List, Dict, Any

class DocumentParser:
    """Parses PDF, TXT, Markdown, and DOCX documents into raw text with section/page metadata."""
    
    @staticmethod
    def parse_file(file_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        Parses document into pages/sections.
        Returns list of dicts: [{"text": str, "page": int, "section": str, "doc_name": str}]
        """
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == ".pdf":
            return DocumentParser._parse_pdf(file_path, filename)
        elif ext == ".docx":
            return DocumentParser._parse_docx(file_path, filename)
        elif ext in [".txt", ".md", ".markdown"]:
            return DocumentParser._parse_text(file_path, filename)
        else:
            # Fallback text reading
            return DocumentParser._parse_text(file_path, filename)

    @staticmethod
    def _parse_pdf(file_path: str, filename: str) -> List[Dict[str, Any]]:
        pages = []
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                cleaned_text = DocumentParser._clean_text(text)
                if cleaned_text:
                    pages.append({
                        "text": cleaned_text,
                        "page": page_num,
                        "section": f"Page {page_num}",
                        "doc_name": filename,
                        "source_location": f"{filename} (Page {page_num})"
                    })
        except Exception as e:
            # Fallback reading if pdf parsing library fails
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
                cleaned = DocumentParser._clean_text(raw)
                pages.append({
                    "text": cleaned,
                    "page": 1,
                    "section": "Document Content",
                    "doc_name": filename,
                    "source_location": filename
                })
        return pages

    @staticmethod
    def _parse_docx(file_path: str, filename: str) -> List[Dict[str, Any]]:
        sections = []
        try:
            import docx
            doc = docx.Document(file_path)
            current_heading = "Overview"
            current_buffer = []
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                if para.style and para.style.name.startswith("Heading"):
                    if current_buffer:
                        sections.append({
                            "text": DocumentParser._clean_text("\n".join(current_buffer)),
                            "page": 1,
                            "section": current_heading,
                            "doc_name": filename,
                            "source_location": f"{filename} ({current_heading})"
                        })
                        current_buffer = []
                    current_heading = text
                else:
                    current_buffer.append(text)
                    
            if current_buffer:
                sections.append({
                    "text": DocumentParser._clean_text("\n".join(current_buffer)),
                    "page": 1,
                    "section": current_heading,
                    "doc_name": filename,
                    "source_location": f"{filename} ({current_heading})"
                })
        except Exception:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
                sections.append({
                    "text": DocumentParser._clean_text(raw),
                    "page": 1,
                    "section": "Overview",
                    "doc_name": filename,
                    "source_location": filename
                })
        return sections

    @staticmethod
    def _parse_text(file_path: str, filename: str) -> List[Dict[str, Any]]:
        sections = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        # Parse headings (# Heading or [Section]) if present
        heading_matches = list(re.finditer(r'^(#{1,4}\s+.*|[A-Z0-9\s]{3,}\n[-=]{3,})', content, re.MULTILINE))
        
        if heading_matches:
            last_idx = 0
            last_heading = "Introduction"
            for match in heading_matches:
                start = match.start()
                if start > last_idx:
                    chunk_text = content[last_idx:start].strip()
                    if chunk_text:
                        sections.append({
                            "text": DocumentParser._clean_text(chunk_text),
                            "page": 1,
                            "section": last_heading,
                            "doc_name": filename,
                            "source_location": f"{filename} ({last_heading})"
                        })
                last_heading = match.group(0).replace("#", "").strip()
                last_idx = match.end()
                
            if last_idx < len(content):
                chunk_text = content[last_idx:].strip()
                if chunk_text:
                    sections.append({
                        "text": DocumentParser._clean_text(chunk_text),
                        "page": 1,
                        "section": last_heading,
                        "doc_name": filename,
                        "source_location": f"{filename} ({last_heading})"
                    })
        else:
            sections.append({
                "text": DocumentParser._clean_text(content),
                "page": 1,
                "section": "Overview",
                "doc_name": filename,
                "source_location": filename
            })
            
        return sections

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        # Remove multiple newlines and tab spaces
        cleaned = re.sub(r'\n{3,}', '\n\n', text)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        return cleaned.strip()
