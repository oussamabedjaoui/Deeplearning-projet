from __future__ import annotations

from pathlib import Path
import shutil

import markdown as md
from nbconvert import HTMLExporter


ROOT = Path(__file__).resolve().parent
SITE = ROOT / 'site'


def render_markdown(source_path: Path, output_name: str, title: str) -> None:
    content = source_path.read_text(encoding='utf-8')
    body = md.markdown(content, extensions=['fenced_code', 'tables'])
    html = f'''<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #0f172a; color: #e5e7eb; }}
    main {{ max-width: 1000px; margin: 0 auto; padding: 48px 24px 80px; }}
    article {{ background: #111827; border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 20px; padding: 32px; line-height: 1.7; }}
    a {{ color: #38bdf8; }}
    pre {{ overflow-x: auto; padding: 16px; background: #020617; border-radius: 12px; }}
    code {{ font-family: Consolas, Monaco, monospace; }}
    h1, h2, h3 {{ line-height: 1.2; }}
  </style>
</head>
<body>
  <main>
    <article>
      {body}
    </article>
  </main>
</body>
</html>'''
    (SITE / output_name).write_text(html, encoding='utf-8')


def render_notebook(source_path: Path, output_name: str) -> None:
    exporter = HTMLExporter()
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_filename(str(source_path))
    (SITE / output_name).write_text(body, encoding='utf-8')


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)

    render_markdown(ROOT / 'README.md', 'README.html', 'SECOM Predictive Maintenance - README')
    render_markdown(ROOT / 'RAPPORT_PROJET.md', 'RAPPORT_PROJET.html', 'SECOM Predictive Maintenance - Rapport')
    render_notebook(ROOT / 'secom_predictive_maintenance.ipynb', 'secom_predictive_maintenance.html')

    shutil.copyfile(ROOT / 'README.md', SITE / 'README.md')
    shutil.copyfile(ROOT / 'RAPPORT_PROJET.md', SITE / 'RAPPORT_PROJET.md')

    index_html = '''<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SECOM Predictive Maintenance</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #0f172a;
      --card: #1f2937;
      --accent: #38bdf8;
      --text: #e5e7eb;
      --muted: #94a3b8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
      color: var(--text);
    }
    main {
      max-width: 960px;
      margin: 0 auto;
      padding: 64px 24px;
    }
    .hero {
      background: rgba(15, 23, 42, 0.82);
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 24px;
      padding: 40px;
      box-shadow: 0 20px 80px rgba(15, 23, 42, 0.45);
    }
    h1 { margin: 0 0 12px; font-size: clamp(2rem, 4vw, 3.5rem); }
    p { line-height: 1.6; color: var(--muted); }
    .links {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-top: 28px;
    }
    a.card {
      display: block;
      padding: 18px 20px;
      border-radius: 18px;
      text-decoration: none;
      color: var(--text);
      background: rgba(31, 41, 55, 0.9);
      border: 1px solid rgba(56, 189, 248, 0.18);
    }
    a.card strong {
      display: block;
      margin-bottom: 6px;
      color: var(--accent);
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>SECOM Predictive Maintenance</h1>
      <p>Version publiée du projet de maintenance prédictive sur le dataset SECOM. Les sources du rapport et le notebook sont disponibles ci-dessous.</p>
      <div class="links">
        <a class="card" href="secom_predictive_maintenance.html">
          <strong>Notebook rendu</strong>
          Ouvre la version HTML du notebook.
        </a>
        <a class="card" href="RAPPORT_PROJET.html">
          <strong>Rapport</strong>
          Consulte le rapport du projet en HTML.
        </a>
        <a class="card" href="README.html">
          <strong>Instructions</strong>
          Voir les informations d'exécution et le contexte.
        </a>
      </div>
    </section>
  </main>
</body>
</html>'''
    (SITE / 'index.html').write_text(index_html, encoding='utf-8')


if __name__ == '__main__':
    main()