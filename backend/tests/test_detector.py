import sys
import os
import importlib.util

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from imageforai import MetadataExtractor, PixelAnalyzer, AIDetector, ObjectDetector, StegDetector


def load_web_app_module():
    app_path = os.path.join(os.path.dirname(__file__), '..', '..', 'app.py')
    spec = importlib.util.spec_from_file_location('webapp', app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_test_images():
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'images')
    if os.path.exists(data_dir):
        return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return []

def test_metadata_extractor():
    print("Testing Metadata Extractor...")
    extractor = MetadataExtractor()
    
    test_images = get_test_images()
    
    if test_images:
        for img_path in test_images[:2]:
            print(f"\nAnalyzing: {os.path.basename(img_path)}")
            metadata = extractor.extract_all_metadata(img_path)
            print(f"  Format: {metadata.get('image_info', {}).get('format')}")
            print(f"  Size: {metadata.get('file_size')} bytes")
            print(f"  Dimensions: {metadata.get('image_info', {}).get('size')}")
            
            if 'png_chunks' in metadata:
                print(f"  PNG Chunks: {len(metadata['png_chunks'])}")
    else:
        print("  No test images found")
    
    print("✓ Metadata Extractor test completed")

def test_pixel_analyzer():
    print("\nTesting Pixel Analyzer...")
    analyzer = PixelAnalyzer()
    
    test_images = get_test_images()
    
    if test_images:
        for img_path in test_images[:2]:
            print(f"\nAnalyzing: {os.path.basename(img_path)}")
            results = analyzer.analyze_all(img_path)
            print(f"  Mean Entropy: {results['entropy']['mean_entropy']:.2f}")
            print(f"  Noise Level: {'High' if results['noise_analysis']['is_high_noise'] else 'Normal'}")
            print(f"  Blockiness: {'Detected' if results['blockiness']['is_blocky'] else 'None'}")
    else:
        print("  No test images found")
    
    print("✓ Pixel Analyzer test completed")

def test_ai_detector():
    print("\nTesting AI Detector...")
    detector = AIDetector()
    
    test_images = get_test_images()
    
    if test_images:
        for img_path in test_images[:2]:
            print(f"\nAnalyzing: {os.path.basename(img_path)}")
            results = detector.detect(img_path)
            print(f"  AI Probability: {results['ai_probability']:.2%}")
            print(f"  Confidence: {results['confidence']:.2%}")
            print(f"  Likely AI Generated: {'Yes' if results['is_likely_ai'] else 'No'}")
    else:
        print("  No test images found")
    
    print("✓ AI Detector test completed")

def test_steg_detector():
    print("\nTesting Steganography Detector...")
    detector = StegDetector()
    
    test_images = get_test_images()
    
    if test_images:
        for img_path in test_images[:2]:
            print(f"\nAnalyzing: {os.path.basename(img_path)}")
            results = detector.analyze_all(img_path)
            print(f"  Suspicious Tests: {results['total_suspicious_tests']}/4")
            print(f"  Overall Confidence: {results['overall_confidence']:.2%}")
            print(f"  Likely Stego: {'Yes' if results['is_likely_stego'] else 'No'}")
    else:
        print("  No test images found")
    
    print("✓ Steganography Detector test completed")


@pytest.fixture()
def flask_client(tmp_path):
    module = load_web_app_module()
    module.app.config['TESTING'] = True
    module.UPLOAD_FOLDER = str(tmp_path / 'uploads')
    module.EXPORT_FOLDER = str(tmp_path)
    os.makedirs(module.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(module.EXPORT_FOLDER, exist_ok=True)
    with module.app.test_client() as client:
        yield {'client': client, 'module': module}


def test_export_returns_downloadable_filename(flask_client):
    response = flask_client['client'].post(
        '/api/export',
        json={
            'filename': 'sample-image',
            'format': 'json',
            'result': {'filename': 'sample-image.png', 'ai_detection': {'ai_probability': 0.5}},
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['code'] == 0
    assert payload['data']['filename'] == 'sample-image.json'
    assert payload['data']['download_url'] == '/api/export/download/sample-image.json'


def test_analyze_supports_mode_parameter(flask_client, monkeypatch):
    from imageforai.services.analysis_service import AnalysisService

    module = flask_client['module']

    class DummyMetadataExtractor:
        def extract_all_metadata(self, _):
            return {'image_info': {'format': 'PNG'}}

    class DummyAIDetector:
        def detect(self, _, metadata_indicators=None):
            return {'ai_probability': 0.9, 'confidence': 0.8, 'is_likely_ai': True}

    class DummyPixelAnalyzer:
        def analyze_all(self, _):
            return {'entropy': {'mean_entropy': 7.5}}

    class DummyStegDetector:
        def analyze_all(self, _):
            return {'overall_confidence': 0.7, 'is_likely_stego': False}

    service = AnalysisService(
        metadata_extractor_factory=DummyMetadataExtractor,
        ai_detector_factory=DummyAIDetector,
        pixel_analyzer_factory=DummyPixelAnalyzer,
        steg_detector_factory=DummyStegDetector,
    )
    module.app.config['ANALYSIS_SERVICE'] = service

    with open(__file__, 'rb') as file:
        response = flask_client['client'].post(
            '/api/analyze?mode=ai',
            data={'file': (file, 'sample.png')},
            content_type='multipart/form-data',
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['code'] == 0
    assert 'ai_detection' in payload['data']
    assert 'metadata' not in payload['data']
    assert 'pixel_analysis' not in payload['data']
    assert payload['data']['mode'] == 'ai'


def test_service_layer_can_build_all_mode_result():
    from imageforai.services.analysis_service import AnalysisService

    class DummyMetadataExtractor:
        def extract_all_metadata(self, _):
            return {'image_info': {'format': 'PNG'}, 'png_chunks': {}}

    class DummyAIDetector:
        def detect(self, _, metadata_indicators=None):
            return {
                'ai_probability': 0.8,
                'confidence': 0.7,
                'is_likely_ai': True,
                'metadata_indicators': metadata_indicators or [],
            }

    class DummyPixelAnalyzer:
        def analyze_all(self, _):
            return {'entropy': {'mean_entropy': 7.1}}

    class DummyStegDetector:
        def analyze_all(self, _):
            return {'overall_confidence': 0.3, 'is_likely_stego': False}

    service = AnalysisService(
        metadata_extractor_factory=DummyMetadataExtractor,
        ai_detector_factory=DummyAIDetector,
        pixel_analyzer_factory=DummyPixelAnalyzer,
        steg_detector_factory=DummyStegDetector,
    )

    result = service.analyze('sample.png', 'all')

    assert result['mode'] == 'all'
    assert result['metadata']['image_info']['format'] == 'PNG'
    assert result['ai_detection']['is_likely_ai'] is True
    assert result['pixel_analysis']['entropy']['mean_entropy'] == 7.1
    assert result['steg_detection']['is_likely_stego'] is False


def test_service_layer_rejects_unsupported_mode():
    from imageforai.services.analysis_service import AnalysisService

    service = AnalysisService()

    with pytest.raises(ValueError, match='Unsupported analyze mode'):
        service.analyze('sample.png', 'unknown')


def test_web_app_factory_registers_frontend_template():
    from imageforai.web.app_factory import create_app

    app = create_app(test_config={'TESTING': True})

    assert app.template_folder.endswith(os.path.join('frontend', 'templates'))


def test_domain_detection_result_model_exposes_summary_fields():
    from imageforai.domain.models import DetectionResult, RiskSummary

    summary = RiskSummary(level='medium', ai_probability=0.72, confidence=0.81, summary='存在中等风险')
    result = DetectionResult(filename='sample.png', mode='all', risk=summary, evidence={'ai_detection': {'ai_probability': 0.72}})

    assert result.filename == 'sample.png'
    assert result.risk.level == 'medium'
    assert result.evidence['ai_detection']['ai_probability'] == 0.72


def test_domain_evidence_bundle_supports_optional_sections():
    from imageforai.domain.models import EvidenceBundle

    evidence = EvidenceBundle(ai_detection={'ai_probability': 0.5}, metadata=None, pixel_analysis={'entropy': 7.2}, steg_detection=None)

    assert evidence.ai_detection['ai_probability'] == 0.5
    assert evidence.metadata is None
    assert evidence.pixel_analysis['entropy'] == 7.2


def test_response_schema_builds_detection_payload_from_domain_model():
    from imageforai.api.schemas.response import build_detection_response
    from imageforai.domain.models import DetectionResult, EvidenceBundle, RiskSummary

    result = DetectionResult(
        filename='sample.png',
        mode='ai',
        risk=RiskSummary(level='high', ai_probability=0.88, confidence=0.77, summary='AI 风险较高', score=88.0, evidence_summary=['频域异常']),
        evidence=EvidenceBundle(ai_detection={'ai_probability': 0.88, 'is_likely_ai': True})
    )

    payload = build_detection_response(result)

    assert payload['filename'] == 'sample.png'
    assert payload['mode'] == 'ai'
    assert payload['risk']['level'] == 'high'
    assert payload['risk']['score'] == 88.0
    assert payload['risk']['evidence_summary'][0] == '频域异常'
    assert payload['ai_detection']['is_likely_ai'] is True


def test_analysis_service_builds_risk_score_and_evidence_summary():
    from imageforai.services.analysis_service import AnalysisService

    class DummyMetadataExtractor:
        def extract_all_metadata(self, _):
            return {'image_info': {'format': 'PNG'}, 'exif': {}, 'png_chunks': {}}

    class DummyAIDetector:
        def detect(self, _, metadata_indicators=None):
            return {'ai_probability': 0.82, 'confidence': 0.64, 'is_likely_ai': True}

    class DummyPixelAnalyzer:
        def analyze_all(self, _):
            return {'entropy': {'mean_entropy': 7.8}}

    class DummyStegDetector:
        def analyze_all(self, _):
            return {'overall_confidence': 0.2, 'is_likely_stego': False}

    service = AnalysisService(
        metadata_extractor_factory=DummyMetadataExtractor,
        ai_detector_factory=DummyAIDetector,
        pixel_analyzer_factory=DummyPixelAnalyzer,
        steg_detector_factory=DummyStegDetector,
    )

    result = service.analyze_domain('sample.png', 'all', 'sample.png')

    assert result.risk.level in {'low', 'medium', 'high'}
    assert result.risk.score >= 35
    assert any('视觉模型分数' in item or '元数据' in item for item in result.risk.evidence_summary)


def test_analysis_service_does_not_treat_thumbnail_or_gaincontrol_as_ai_metadata():
    from imageforai.services.analysis_service import AnalysisService

    service = AnalysisService()
    metadata = {
        'exif': {
            'Thumbnail Compression': 'JPEG',
            'EXIF GainControl': 'Normal',
            'Image Software': 'Adobe Photoshop Camera Raw 16.0.1 (Windows)',
        },
        'png_chunks': {},
    }

    indicators = service._collect_metadata_indicators(metadata)

    assert 'EXIF: Thumbnail Compression' not in indicators
    assert 'EXIF: EXIF GainControl' not in indicators
    assert indicators == []


def test_analysis_service_reduces_risk_for_document_like_image_without_ai_metadata():
    from imageforai.services.analysis_service import AnalysisService

    class DummyMetadataExtractor:
        def extract_all_metadata(self, _):
            return {
                'image_info': {'format': 'PNG', 'mode': 'RGBA', 'width': 1678, 'height': 1216},
                'exif': {},
                'png_chunks': {},
            }

    class DummyAIDetector:
        def detect(self, _, metadata_indicators=None):
            return {
                'ai_probability': 0.6,
                'confidence': 0.57,
                'is_likely_ai': False,
                'metadata_indicators': metadata_indicators or [],
                'features': {
                    'color_entropy': 0.45,
                    'noise_level': 2315.53,
                    'edge_quality': 5199.47,
                    'color_abnormality': 0.0,
                    'texture_score': 795.90,
                    'blockiness': 962.17,
                    'metadata_score': 0.0,
                },
                'normalized_features': {
                    'color_entropy': 0.0,
                    'noise_level': 1.0,
                    'edge_quality': 1.0,
                    'color_abnormality': 0.0,
                    'texture_score': 1.0,
                    'blockiness': 1.0,
                    'metadata_score': 0.0,
                },
            }

    class DummyPixelAnalyzer:
        def analyze_all(self, _):
            return {
                'entropy': {'mean_entropy': 0.45},
                'pixel_patterns': {
                    'color_diversity': 0.00012,
                    'color_correlation': {'rg': 1.0, 'rb': 1.0, 'gb': 1.0},
                },
                'noise_analysis': {'is_high_noise': True, 'noise_ratio': 0.15},
            }

    class DummyStegDetector:
        def analyze_all(self, _):
            return {'overall_confidence': 0.12, 'is_likely_stego': False}

    service = AnalysisService(
        metadata_extractor_factory=DummyMetadataExtractor,
        ai_detector_factory=DummyAIDetector,
        pixel_analyzer_factory=DummyPixelAnalyzer,
        steg_detector_factory=DummyStegDetector,
    )

    result = service.analyze_domain('diagram.png', 'all', 'diagram.png')

    assert result.risk.level in {'low', 'unknown'}
    assert result.risk.score < 40


def test_analysis_service_marks_known_generator_metadata_as_high_risk():
    from imageforai.services.analysis_service import AnalysisService

    class DummyMetadataExtractor:
        def extract_all_metadata(self, _):
            return {
                'image_info': {'format': 'PNG', 'mode': 'RGB', 'width': 1024, 'height': 1024},
                'exif': {
                    'Image Software': 'Midjourney 6.0',
                },
                'png_chunks': {
                    'tEXt': {'keyword': 'parameters', 'text': 'stable diffusion prompt test'}
                },
            }

    class DummyAIDetector:
        def detect(self, _, metadata_indicators=None):
            return {
                'ai_probability': 0.38,
                'confidence': 0.41,
                'is_likely_ai': False,
                'metadata_indicators': metadata_indicators or [],
            }

    class DummyPixelAnalyzer:
        def analyze_all(self, _):
            return {'entropy': {'mean_entropy': 6.9}, 'pixel_patterns': {'color_diversity': 0.03, 'color_correlation': {'rg': 0.6, 'rb': 0.5, 'gb': 0.55}}}

    class DummyStegDetector:
        def analyze_all(self, _):
            return {'overall_confidence': 0.15, 'is_likely_stego': False}

    service = AnalysisService(
        metadata_extractor_factory=DummyMetadataExtractor,
        ai_detector_factory=DummyAIDetector,
        pixel_analyzer_factory=DummyPixelAnalyzer,
        steg_detector_factory=DummyStegDetector,
    )

    result = service.analyze_domain('generated.png', 'all', 'generated.png')

    assert result.risk.level == 'high'
    assert result.risk.score >= 75
    assert any('元数据' in item or 'metadata' in item.lower() for item in result.risk.evidence_summary)


def test_analysis_service_keeps_camera_photo_without_ai_markers_below_high_risk():
    from imageforai.services.analysis_service import AnalysisService

    class DummyMetadataExtractor:
        def extract_all_metadata(self, _):
            return {
                'image_info': {'format': 'JPEG', 'mode': 'RGB', 'width': 4000, 'height': 3000},
                'exif': {
                    'Image Make': 'Canon',
                    'Image Model': 'EOS R6',
                    'EXIF LensModel': 'RF24-105mm',
                    'EXIF SceneType': 'Directly Photographed',
                    'Image Software': 'Adobe Photoshop Lightroom Classic 13.2',
                },
                'png_chunks': {},
            }

    class DummyAIDetector:
        def detect(self, _, metadata_indicators=None):
            return {
                'ai_probability': 0.71,
                'confidence': 0.76,
                'is_likely_ai': True,
                'metadata_indicators': metadata_indicators or [],
            }

    class DummyPixelAnalyzer:
        def analyze_all(self, _):
            return {
                'entropy': {'mean_entropy': 6.2},
                'pixel_patterns': {'color_diversity': 0.02, 'color_correlation': {'rg': 0.88, 'rb': 0.8, 'gb': 0.86}},
                'noise_analysis': {'is_high_noise': False, 'noise_ratio': 0.09},
            }

    class DummyStegDetector:
        def analyze_all(self, _):
            return {'overall_confidence': 0.1, 'is_likely_stego': False}

    service = AnalysisService(
        metadata_extractor_factory=DummyMetadataExtractor,
        ai_detector_factory=DummyAIDetector,
        pixel_analyzer_factory=DummyPixelAnalyzer,
        steg_detector_factory=DummyStegDetector,
    )

    result = service.analyze_domain('camera.jpg', 'all', 'camera.jpg')

    assert result.risk.level in {'low', 'medium'}
    assert result.risk.score < 75


def test_ai_detector_document_like_features_do_not_collapse_to_fixed_probability():
    from imageforai.modules.ai_detector import AIDetector

    detector = AIDetector()
    normalized_features = {
        'color_entropy': 0.0,
        'noise_level': 1.0,
        'edge_quality': 1.0,
        'color_abnormality': 0.0,
        'texture_score': 1.0,
        'blockiness': 1.0,
        'metadata_score': 0.0,
    }
    context = {
        'is_document_like': True,
        'has_camera_markers': False,
        'has_strong_ai_metadata': False,
        'color_diversity': 0.00012,
        'mean_entropy': 0.45,
        'avg_correlation': 1.0,
    }

    probability = detector.estimate_ai_probability(normalized_features, context)

    assert probability < 0.35
    assert probability != 0.6


def test_ai_detector_distinguishes_retouched_photo_from_direct_ai_image():
    from imageforai.modules.ai_detector import AIDetector

    detector = AIDetector()

    retouched_photo_probability = detector.estimate_ai_probability(
        {
            'color_entropy': 0.62,
            'noise_level': 0.18,
            'edge_quality': 0.74,
            'color_abnormality': 0.32,
            'texture_score': 0.58,
            'blockiness': 0.22,
            'metadata_score': 0.0,
        },
        {
            'is_document_like': False,
            'has_camera_markers': True,
            'has_strong_ai_metadata': False,
            'color_diversity': 0.018,
            'mean_entropy': 6.2,
            'avg_correlation': 0.88,
        },
    )

    direct_ai_probability = detector.estimate_ai_probability(
        {
            'color_entropy': 0.92,
            'noise_level': 0.81,
            'edge_quality': 0.86,
            'color_abnormality': 0.77,
            'texture_score': 0.91,
            'blockiness': 0.84,
            'metadata_score': 1.0,
        },
        {
            'is_document_like': False,
            'has_camera_markers': False,
            'has_strong_ai_metadata': True,
            'color_diversity': 0.029,
            'mean_entropy': 6.9,
            'avg_correlation': 0.54,
        },
    )

    assert retouched_photo_probability < 0.55
    assert direct_ai_probability > 0.75
    assert direct_ai_probability - retouched_photo_probability > 0.2


def test_analysis_service_does_not_label_high_visual_ai_probability_as_low_risk_without_counter_evidence():
    from imageforai.services.analysis_service import AnalysisService

    class DummyMetadataExtractor:
        def extract_all_metadata(self, _):
            return {
                'image_info': {'format': 'PNG', 'mode': 'RGB', 'width': 1536, 'height': 1024},
                'exif': {},
                'png_chunks': {},
            }

    class DummyAIDetector:
        def detect(self, _, metadata_indicators=None):
            return {
                'ai_probability': 0.6843,
                'confidence': 0.8571,
                'is_likely_ai': True,
                'metadata_indicators': metadata_indicators or [],
                'features': {
                    'color_entropy': 7.47,
                    'noise_level': 237.83,
                    'edge_quality': 751.87,
                    'color_abnormality': 29.80,
                    'texture_score': 442.38,
                    'blockiness': 173.64,
                    'metadata_score': 0.0,
                },
                'normalized_features': {
                    'color_entropy': 1.0,
                    'noise_level': 1.0,
                    'edge_quality': 1.0,
                    'color_abnormality': 0.49,
                    'texture_score': 1.0,
                    'blockiness': 1.0,
                    'metadata_score': 0.0,
                },
            }

    class DummyPixelAnalyzer:
        def analyze_all(self, _):
            return {
                'entropy': {'mean_entropy': 7.47},
                'pixel_patterns': {'color_diversity': 0.015, 'color_correlation': {'rg': 0.72, 'rb': 0.66, 'gb': 0.70}},
                'noise_analysis': {'is_high_noise': False, 'noise_ratio': 0.14},
            }

    class DummyStegDetector:
        def analyze_all(self, _):
            return {'overall_confidence': 0.2, 'is_likely_stego': False, 'total_suspicious_tests': 0}

    service = AnalysisService(
        metadata_extractor_factory=DummyMetadataExtractor,
        ai_detector_factory=DummyAIDetector,
        pixel_analyzer_factory=DummyPixelAnalyzer,
        steg_detector_factory=DummyStegDetector,
    )

    result = service.analyze_domain('candidate-ai.png', 'all', 'candidate-ai.png')

    assert result.risk.ai_probability >= 0.65
    assert result.risk.level in {'medium', 'high'}
    assert result.risk.score >= 45


def test_api_analyze_route_blueprint_exists():
    from imageforai.api.routes.analyze import analyze_bp

    assert analyze_bp.name == 'analyze'


def test_api_export_route_blueprint_exists():
    from imageforai.api.routes.export import export_bp

    assert export_bp.name == 'export'


def test_standard_response_wraps_payload_and_message():
    from imageforai.api.schemas.response import build_success_response

    payload = build_success_response({'filename': 'sample.png'}, message='done')

    assert payload['code'] == 0
    assert payload['message'] == 'done'
    assert payload['data']['filename'] == 'sample.png'


def test_export_service_includes_risk_explanation_in_markdown_and_text(tmp_path):
    from imageforai.services.export_service import ExportService

    service = ExportService(str(tmp_path))
    result = {
        'filename': 'sample.png',
        'risk': {
            'level': 'medium',
            'summary': '综合风险分 52.00，视觉模型分数 52.00%',
            'explanation': {
                'level_label': '中风险',
                'level_description': '存在部分可疑信号，但缺少强元数据证据。',
                'primary_reason': '视觉信号偏高，但没有明确生成器元数据。',
                'action': '建议结合原图来源、拍摄链路和更多样本复核。',
                'evidence_groups': [
                    {'title': '元数据证据', 'items': ['未发现明确生成器标识']},
                    {'title': '视觉证据', 'items': ['视觉模型分数 52.00%']},
                ],
            },
        },
    }

    markdown = service.generate_markdown(result)
    text = service.generate_text(result)

    assert '中风险' in markdown
    assert '主要原因' in markdown
    assert '建议动作' in markdown
    assert '元数据证据' in markdown
    assert '视觉证据' in markdown
    assert '中风险' in text
    assert '主要原因' in text
    assert '建议动作' in text


def test_error_response_contains_standard_error_code():
    from imageforai.api.schemas.response import build_error_response

    payload = build_error_response('bad request', 1001)

    assert payload['error'] == 'bad request'
    assert payload['code'] == 1001


def test_error_codes_expose_named_constants():
    from imageforai.api.schemas.error_codes import ErrorCode

    assert ErrorCode.BAD_REQUEST == 1001
    assert ErrorCode.NOT_FOUND == 3000


def test_analyze_request_schema_validates_mode_and_filename():
    from imageforai.api.schemas.request import AnalyzeRequest

    request_data = AnalyzeRequest(filename='sample.png', mode='ai')

    assert request_data.filename == 'sample.png'
    assert request_data.mode == 'ai'


def test_history_item_model_exposes_task_metadata():
    from imageforai.domain.models import HistoryItem, RiskSummary

    item = HistoryItem(
        task_id='task-1',
        filename='sample.png',
        mode='all',
        created_at='2026-05-20T13:00:00',
        risk=RiskSummary(level='high', ai_probability=0.8, confidence=0.7, summary='高风险')
    )

    assert item.task_id == 'task-1'
    assert item.risk.level == 'high'


def test_history_repository_saves_and_lists_records(tmp_path):
    from imageforai.infra.storage.history_repository import JsonHistoryRepository

    repository = JsonHistoryRepository(str(tmp_path / 'history.json'))
    repository.save({'filename': 'sample.png', 'mode': 'ai', 'risk': {'level': 'high'}})
    repository.save({'filename': 'sample2.png', 'mode': 'all', 'risk': {'level': 'medium'}})

    items = repository.list_all()

    assert len(items) == 2
    assert items[0]['filename'] == 'sample2.png'
    assert items[1]['risk']['level'] == 'high'


def test_history_api_returns_wrapped_records(flask_client):
    module = flask_client['module']
    history_file = os.path.join(module.app.config['EXPORT_FOLDER'], 'history.json')
    if os.path.exists(history_file):
        os.remove(history_file)

    from imageforai.infra.storage.history_repository import JsonHistoryRepository

    repository = JsonHistoryRepository(history_file)
    repository.save({'filename': 'history.png', 'mode': 'all', 'risk': {'level': 'low'}})
    module.app.config['HISTORY_REPOSITORY'] = repository

    response = flask_client['client'].get('/api/history')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['code'] == 0
    assert payload['data']['items'][0]['filename'] == 'history.png'


def test_detection_response_contains_task_id_and_timestamp():
    from imageforai.api.schemas.response import build_detection_response
    from imageforai.domain.models import DetectionResult, EvidenceBundle, RiskSummary

    result = DetectionResult(
        task_id='task-123',
        filename='sample.png',
        mode='all',
        created_at='2026-05-20T12:30:00',
        risk=RiskSummary(level='medium', ai_probability=0.52, confidence=0.61, summary='中等风险'),
        evidence=EvidenceBundle(ai_detection={'ai_probability': 0.52})
    )

    payload = build_detection_response(result)

    assert payload['task_id'] == 'task-123'
    assert payload['created_at'] == '2026-05-20T12:30:00'


def test_detection_response_contains_risk_explanation_payload():
    from imageforai.api.schemas.response import build_detection_response
    from imageforai.domain.models import DetectionResult, EvidenceBundle, RiskSummary

    result = DetectionResult(
        task_id='task-risk-explain',
        filename='sample.png',
        mode='all',
        created_at='2026-05-20T12:35:00',
        risk=RiskSummary(
            level='medium',
            ai_probability=0.52,
            confidence=0.61,
            summary='综合风险分 52.00，视觉模型分数 52.00%',
            score=52.0,
            evidence_summary=['视觉模型分数 52.00%'],
            explanation={
                'level_label': '中风险',
                'level_description': '存在部分可疑信号，但缺少强元数据证据。',
                'primary_reason': '视觉信号偏高，但没有明确生成器元数据。',
                'action': '建议结合原图来源、拍摄链路和更多样本复核。',
                'evidence_groups': [
                    {'title': '元数据证据', 'items': ['未发现明确生成器标识']},
                    {'title': '视觉证据', 'items': ['视觉模型分数 52.00%']},
                ],
            },
        ),
        evidence=EvidenceBundle(ai_detection={'ai_probability': 0.52})
    )

    payload = build_detection_response(result)

    assert payload['risk']['explanation']['level_label'] == '中风险'
    assert payload['risk']['explanation']['primary_reason']
    assert payload['risk']['explanation']['action']
    assert len(payload['risk']['explanation']['evidence_groups']) == 2


def test_detection_response_contains_probability_term_and_image_classification_tags():
    from imageforai.api.schemas.response import build_detection_response
    from imageforai.domain.models import DetectionResult, EvidenceBundle, RiskSummary

    result = DetectionResult(
        task_id='task-terms',
        filename='diagram.png',
        mode='all',
        created_at='2026-05-20T14:10:00',
        risk=RiskSummary(
            level='unknown',
            ai_probability=0.10,
            confidence=0.42,
            summary='综合风险分 0.00，更接近文档/流程图，不作为高风险依据',
            score=0.0,
            evidence_summary=['图像更接近文档/流程图/截图'],
            explanation={
                'level_label': '结果不足',
                'probability_label': 'AI风险概率',
                'probability_description': '该数值是启发式模型评分，不等同于真实后验概率。',
                'image_classification_tags': ['文档/流程图', '非相机链路'],
                'level_description': '当前证据不足，暂时无法给出稳定结论。',
                'primary_reason': '图像更接近文档、流程图或截图形态，因此对视觉高分做了降权处理。',
                'action': '建议补充更多证据后再次检测。',
                'evidence_groups': [],
            },
        ),
        evidence=EvidenceBundle(ai_detection={'ai_probability': 0.10})
    )

    payload = build_detection_response(result)

    assert payload['risk']['explanation']['probability_label'] == 'AI风险概率'
    assert '后验概率' in payload['risk']['explanation']['probability_description']
    assert '文档/流程图' in payload['risk']['explanation']['image_classification_tags']


def test_history_api_supports_page_and_page_size(flask_client):
    module = flask_client['module']
    history_file = os.path.join(module.app.config['EXPORT_FOLDER'], 'paged-history.json')
    if os.path.exists(history_file):
        os.remove(history_file)

    from imageforai.infra.storage.history_repository import JsonHistoryRepository

    repository = JsonHistoryRepository(history_file)
    repository.save({'filename': 'three.png'})
    repository.save({'filename': 'two.png'})
    repository.save({'filename': 'one.png'})
    module.app.config['HISTORY_REPOSITORY'] = repository

    response = flask_client['client'].get('/api/history?page=2&page_size=1')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['code'] == 0
    assert payload['data']['items'][0]['filename'] == 'two.png'
    assert payload['data']['total'] == 3
    assert payload['data']['page'] == 2


def test_task_repository_supports_save_and_get(tmp_path):
    from imageforai.infra.storage.task_repository import JsonTaskRepository

    repository = JsonTaskRepository(str(tmp_path / 'tasks.json'))
    repository.save({'task_id': 'task-1', 'filename': 'sample.png'})

    item = repository.get('task-1')

    assert item['filename'] == 'sample.png'


def test_task_query_api_returns_saved_task(flask_client):
    module = flask_client['module']
    task_file = os.path.join(module.app.config['EXPORT_FOLDER'], 'tasks.json')
    if os.path.exists(task_file):
        os.remove(task_file)

    from imageforai.infra.storage.task_repository import JsonTaskRepository

    repository = JsonTaskRepository(task_file)
    repository.save({'task_id': 'task-xyz', 'filename': 'query.png', 'mode': 'all'})
    module.app.config['TASK_REPOSITORY'] = repository

    response = flask_client['client'].get('/api/tasks/task-xyz')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['code'] == 0
    assert payload['data']['filename'] == 'query.png'


def test_frontend_contains_mode_buttons_and_history_panel():
    html_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as file:
        html = file.read()

    assert 'data-mode="all"' in html
    assert 'data-mode="ai"' in html
    assert 'data-mode="metadata"' in html
    assert 'data-mode="pixel"' in html
    assert 'data-mode="steg"' in html
    assert 'id="history-list"' in html
    assert 'id="toast"' in html


def test_frontend_contains_risk_explanation_panel():
    html_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as file:
        html = file.read()

    assert 'id="risk-explanation-level"' in html
    assert 'id="risk-explanation-reason"' in html
    assert 'id="risk-explanation-action"' in html
    assert 'id="risk-evidence-groups"' in html


def test_frontend_contains_risk_badge_and_level_helper_copy():
    html_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as file:
        html = file.read()

    assert 'id="risk-level-badge"' in html
    assert 'id="risk-level-helper"' in html
    assert 'badge-high' in html
    assert 'badge-medium' in html
    assert 'badge-low' in html
    assert 'badge-unknown' in html


def test_frontend_contains_probability_term_and_image_classification_tags_panel():
    html_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as file:
        html = file.read()

    assert 'AI风险概率' in html
    assert 'id="risk-probability-description"' in html
    assert 'id="image-classification-tags"' in html


def test_frontend_contains_upload_area_and_file_input():
    html_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as file:
        html = file.read()

    assert 'id="upload-area"' in html
    assert 'id="file-input"' in html
    assert '上传图片' in html


def test_frontend_contains_start_detection_button_and_flow_text():
    html_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as file:
        html = file.read()

    assert 'id="start-detect-button"' in html
    assert '开始检测' in html
    assert '选择图片后，点击开始检测' in html
    assert '正在检测，请稍候' in html


def test_frontend_contains_risk_card_and_history_loader_hook():
    html_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as file:
        html = file.read()

    assert 'id="risk-level"' in html
    assert 'id="risk-summary"' in html
    assert 'fetchHistory()' in html
    assert "data.data" in html


def test_frontend_contains_export_handlers_and_history_detail_hook():
    html_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as file:
        html = file.read()

    assert 'exportResult(' in html
    assert 'fetchTaskDetail(' in html
    assert 'risk-high' in html
    assert 'risk-medium' in html
    assert 'risk-low' in html


def test_frontend_contains_metric_cards_and_tab_switch_hook():
    html_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as file:
        html = file.read()

    assert 'metric-card' in html
    assert 'openTab(' in html
    assert 'id="metadata-cards"' in html
    assert 'id="pixel-cards"' in html
    assert 'id="steg-cards"' in html


def test_frontend_contains_evidence_summary_and_mode_based_task_switch():
    html_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as file:
        html = file.read()

    assert 'id="risk-evidence-list"' in html
    assert 'openTab(tabByMode' in html
    assert '暂无历史记录' in html
    assert '检测失败，请检查图片或稍后重试' in html


def test_frontend_tab_panels_are_not_nested_under_ai_panel():
    html_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as file:
        html = file.read()

    ai_open = html.index('<div id="ai-tab" class="tab-content active">')
    ai_close = html.index('</div>', ai_open)
    metadata_open = html.index('<div id="metadata-tab" class="tab-content">')

    assert metadata_open > ai_close

if __name__ == '__main__':
    test_metadata_extractor()
    test_pixel_analyzer()
    test_ai_detector()
    test_steg_detector()
    
    print("\n" + "="*50)
    print("All tests completed successfully!")
    print("="*50)
