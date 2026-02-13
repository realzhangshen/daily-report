"""
Renderer for Daily Feed reports.
Generates HTML and Markdown output from analysis results.
Includes a table of contents (TOC) and navigation features.
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from .core.types import ExtractionResult

def render_briefing(briefing: str, extractions: list[ExtractionResult], output_path: Path, title: str):
    """Render the daily briefing and extractions to an HTML file."""
    html_content = _generate_html(briefing, extractions, title)
    output_path.write_text(html_content, encoding="utf-8")

def render_html(briefing: str, extractions: list[ExtractionResult], title: str) -> str:
    """Return the daily briefing and extractions as an HTML string."""
    return _generate_html(briefing, extractions, title)

def render_markdown(briefing: str, extractions: list[ExtractionResult], title: str) -> str:
    """Return the daily briefing and extractions as a Markdown string."""
    # Implementation simplified for now
    md = f"# {title}\n\n## Briefing\n\n{briefing}\n\n"
    for ex in extractions:
        md += f"### {ex.article.title}\n\n{ex.one_line_summary}\n\n"
    return md

def _generate_html(briefing: str, extractions: list[ExtractionResult], title: str) -> str:
    # Build articles HTML
    articles_html = ""
    for ex in extractions:
        tags_html = "".join([f'<span class="tag">{tag}</span>' for tag in ex.tags])
        importance_stars = "⭐" * ex.importance
        
        articles_html += f"""
        <article class="card" id="article-{id(ex)}">
            <div class="card-header">
                <span class="category">{ex.category.upper()}</span>
                <span class="importance">{importance_stars}</span>
            </div>
            <h2 class="article-title"><a href="{ex.article.url}" target="_blank">{ex.article.title}</a></h2>
            <div class="tags">{tags_html}</div>
            <p class="summary">{ex.one_line_summary}</p>
            <div class="takeaway">
                <strong>Key Takeaway:</strong> {ex.key_takeaway}
            </div>
            <div class="meta">
                <span>Source: {ex.article.source or 'Unknown'}</span>
                <span>Type: {ex.content_type}</span>
            </div>
        </article>
        """

    # Final HTML Template
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-light: #64748b;
            --border: #e2e8f0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.6;
            margin: 0;
            display: flex;
            justify-content: center;
        }}
        .container {{
            max-width: 900px;
            width: 100%;
            padding: 2rem;
            margin-right: 250px; /* Space for TOC */
        }}
        header {{
            margin-bottom: 3rem;
            text-align: center;
        }}
        .home-btn {{
            display: inline-block;
            margin-bottom: 1rem;
            padding: 0.5rem 1rem;
            background: var(--primary);
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 500;
            transition: opacity 0.2s;
        }}
        .home-btn:hover {{ opacity: 0.9; }}
        
        h1 {{ font-size: 2.5rem; margin-top: 0.5rem; color: var(--text); }}
        
        .briefing-section {{
            background: var(--card-bg);
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid var(--border);
            margin-bottom: 3rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .briefing-content {{ font-size: 1.1rem; white-space: pre-wrap; }}

        .card {{
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border);
            margin-bottom: 1.5rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        
        .card-header {{ display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.8rem; font-weight: bold; }}
        .category {{ color: var(--primary); letter-spacing: 0.05em; }}
        
        .article-title {{ margin: 0.5rem 0; font-size: 1.5rem; }}
        .article-title a {{ color: inherit; text-decoration: none; }}
        .article-title a:hover {{ color: var(--primary); }}
        
        .tags {{ margin-bottom: 1rem; }}
        .tag {{
            display: inline-block;
            background: #eff6ff;
            color: #1e40af;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            margin-right: 0.5rem;
            border: 1px solid #dbeafe;
        }}
        
        .summary {{ color: var(--text); margin-bottom: 1rem; }}
        .takeaway {{
            background: #f1f5f9;
            padding: 1rem;
            border-radius: 8px;
            font-size: 0.95rem;
            border-left: 4px solid var(--primary);
        }}
        
        .meta {{
            margin-top: 1rem;
            display: flex;
            gap: 1rem;
            font-size: 0.8rem;
            color: var(--text-light);
        }}

        /* Table of Contents Sidebar */
        #toc-sidebar {{
            position: fixed;
            right: 2rem;
            top: 2rem;
            width: 220px;
            max-height: 80vh;
            overflow-y: auto;
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border);
            font-size: 0.9rem;
        }}
        #toc-sidebar h3 {{ margin-top: 0; font-size: 1.1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
        #toc-sidebar ul {{ list-style: none; padding: 0; margin: 0; }}
        #toc-sidebar li {{ margin: 0.8rem 0; }}
        #toc-sidebar a {{ color: var(--text-light); text-decoration: none; transition: color 0.2s; }}
        #toc-sidebar a:hover {{ color: var(--primary); }}
        #toc-sidebar .toc-h2 {{ font-weight: 600; }}
        #toc-sidebar .toc-h3 {{ padding-left: 1rem; font-size: 0.85rem; }}

        @media (max-width: 1200px) {{
            .container {{ margin-right: 0; }}
            #toc-sidebar {{ display: none; }}
        }}
        @media (max-width: 640px) {{
            .container {{ padding: 1rem; }}
            h1 {{ font-size: 1.8rem; }}
        }}
    </style>
</head>
<body>
    <div id="toc-sidebar">
        <a href="/" class="home-btn" style="width: 100%; text-align: center; box-sizing: border-box; margin-bottom: 1.5rem;">返回主页</a>
        <h3>目录</h3>
        <ul id="toc-list"></ul>
    </div>

    <div class="container">
        <header>
            <a href="/" class="home-btn">返回主页</a>
            <h1>{title}</h1>
            <p style="color: var(--text-light)">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </header>

        <section class="briefing-section">
            <h2>每日简报</h2>
            <div class="briefing-content">{briefing}</div>
        </section>

        <section class="articles-section">
            {articles_html}
        </section>

        <footer style="margin-top: 5rem; text-align: center; color: var(--text-light); padding-bottom: 3rem;">
            <p>© {datetime.now().year} Daily Feed Agent · Created by Claw Zhang 🦞</p>
        </footer>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const tocList = document.getElementById('toc-list');
            const headings = document.querySelectorAll('h2, h3');
            
            headings.forEach((heading, index) => {{
                if (heading.textContent === '目录') return;
                
                // Add ID if not present
                if (!heading.id) {{
                    heading.id = 'heading-' + index;
                }}
                
                const li = document.createElement('li');
                li.className = heading.tagName.toLowerCase() === 'h2' ? 'toc-h2' : 'toc-h3';
                
                const a = document.createElement('a');
                a.href = '#' + heading.id;
                a.textContent = heading.textContent;
                
                li.appendChild(a);
                tocList.appendChild(li);
            }});
        }});
    </script>
</body>
</html>
"""
