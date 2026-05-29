import os
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from RAG import ingest_document, run_rag_pipeline

ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown"}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "uploads")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    def _is_allowed(filename: str) -> bool:
        _, ext = os.path.splitext(filename)
        return ext.lower() in ALLOWED_EXTENSIONS

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.post("/api/upload")
    def upload():
        if "file" not in request.files:
            return jsonify({"error": "Missing file field"}), 400

        file = request.files["file"]
        if not file or not file.filename:
            return jsonify({"error": "No file selected"}), 400

        original_name = secure_filename(file.filename)
        if not original_name or not _is_allowed(original_name):
            return jsonify({"error": "Only .txt, .md, and .markdown files are supported."}), 400

        base, ext = os.path.splitext(original_name)
        stored_name = f"{base}_{uuid4().hex}{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        file.save(save_path)

        try:
            chunk_count = ingest_document(save_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": "Ingestion failed", "details": str(exc)}), 500

        return jsonify(
            {
                "ingested_chunks": chunk_count,
                "filename": original_name,
                "stored_path": save_path,
            }
        )

    @app.post("/api/query")
    def query():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question")
        if not question:
            return jsonify({"error": "Missing required field: question"}), 400

        try:
            answer = run_rag_pipeline(question)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:
            return jsonify({"error": "Query failed", "details": str(exc)}), 500

        return jsonify({"answer": answer})

    return app


app = create_app()

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug)
