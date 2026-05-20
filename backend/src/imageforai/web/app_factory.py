import os

from flask import Flask, render_template
from flask_cors import CORS

from ..api.routes import analyze_bp, export_bp
from ..infra.storage.history_repository import JsonHistoryRepository
from ..infra.storage.task_repository import JsonTaskRepository
from ..services.analysis_service import AnalysisService


def create_app(test_config=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    template_folder = os.path.join(base_dir, "frontend", "templates")
    app = Flask(__name__, template_folder=template_folder)
    CORS(app)

    app.config.update(
        UPLOAD_FOLDER=os.path.join(base_dir, "uploads"),
        EXPORT_FOLDER=os.path.join(base_dir, "exports"),
        HISTORY_FILE=os.path.join(base_dir, "exports", "history.json"),
        TASK_FILE=os.path.join(base_dir, "exports", "tasks.json"),
    )

    if test_config:
        app.config.update(test_config)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["EXPORT_FOLDER"], exist_ok=True)

    if "ANALYSIS_SERVICE" not in app.config:
        ml_model_path = app.config.get("ML_MODEL_PATH") or os.environ.get("IMAGEFORAI_ML_MODEL_PATH")
        app.config["ANALYSIS_SERVICE"] = AnalysisService(ml_model_path=ml_model_path)

    if "HISTORY_REPOSITORY" not in app.config:
        app.config["HISTORY_REPOSITORY"] = JsonHistoryRepository(app.config["HISTORY_FILE"])

    if "TASK_REPOSITORY" not in app.config:
        app.config["TASK_REPOSITORY"] = JsonTaskRepository(app.config["TASK_FILE"])

    app.register_blueprint(analyze_bp)
    app.register_blueprint(export_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app
