import os

from flask import Blueprint, current_app, jsonify, request

from ..schemas.request import AnalyzeRequest
from ..schemas.response import build_detection_response, build_error_response, build_success_response
from ...services.analysis_service import AnalysisService


analyze_bp = Blueprint("analyze", __name__)


@analyze_bp.route("/api/analyze", methods=["POST"])
def analyze_image():
    mode = request.args.get("mode", "all").lower()
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    request_model = AnalyzeRequest(filename=file.filename, mode=mode)

    upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], request_model.filename)
    file.save(upload_path)
    service = current_app.config.get("ANALYSIS_SERVICE") or AnalysisService()

    try:
        result = service.analyze_domain(upload_path, request_model.mode, request_model.filename)
        payload = build_detection_response(result)

        history_repository = current_app.config.get("HISTORY_REPOSITORY")
        if history_repository is not None:
            history_repository.save(payload)

        task_repository = current_app.config.get("TASK_REPOSITORY")
        if task_repository is not None:
            task_repository.save(payload)

        return jsonify(build_success_response(payload))
    except ValueError as error:
        return jsonify(build_error_response(str(error))), 400
    except Exception as error:
        return jsonify(build_error_response(str(error), 500)), 500
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)
