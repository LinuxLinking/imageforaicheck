import os

from flask import Blueprint, current_app, jsonify, request, send_file

from ..schemas.response import build_error_response, build_success_response
from ...services.export_service import ExportService


export_bp = Blueprint("export", __name__)


@export_bp.route("/api/export", methods=["POST"])
def export_result():
    data = request.json or {}
    service = ExportService(current_app.config["EXPORT_FOLDER"])

    try:
        payload = service.export(
            data.get("filename", "result"),
            data.get("format", "md"),
            data.get("result", {}),
        )
        return jsonify(build_success_response(payload))
    except ValueError as error:
        return jsonify(build_error_response(str(error))), 400


@export_bp.route("/api/export/download/<filename>")
def download_export(filename):
    export_path = os.path.join(current_app.config["EXPORT_FOLDER"], filename)
    if os.path.exists(export_path):
        return send_file(export_path, as_attachment=True)
    return jsonify(build_error_response("File not found", 404)), 404


@export_bp.route("/api/history", methods=["GET"])
def list_history():
    repository = current_app.config.get("HISTORY_REPOSITORY")
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 20))
    if repository is None:
        return jsonify(build_success_response({"items": [], "total": 0, "page": page, "page_size": page_size}))
    payload = repository.list_page(page=page, page_size=page_size)
    return jsonify(build_success_response(payload))


@export_bp.route("/api/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    repository = current_app.config.get("TASK_REPOSITORY")
    if repository is None:
        return jsonify(build_error_response("Task not found", 3000)), 404

    item = repository.get(task_id)
    if item is None:
        return jsonify(build_error_response("Task not found", 3000)), 404
    return jsonify(build_success_response(item))
