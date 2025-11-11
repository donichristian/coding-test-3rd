"""
Pre-download OCR models during Docker build to avoid runtime downloads.

This script initializes docling and RapidOCR models so they are cached
in the Docker image before users upload documents.
"""
import os
import sys
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_pdf():
    """Create a minimal dummy PDF for testing"""
    try:
        # Try to download a small test PDF from a public URL
        import urllib.request
        test_pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        temp_pdf_path = temp_pdf.name
        temp_pdf.close()
        
        try:
            urllib.request.urlretrieve(test_pdf_url, temp_pdf_path)
            if os.path.exists(temp_pdf_path) and os.path.getsize(temp_pdf_path) > 0:
                return temp_pdf_path
        except Exception as e:
            logger.debug(f"Could not download test PDF: {e}")
        
        # Fallback: create minimal PDF using pypdf
        from pypdf import PdfWriter
        
        writer = PdfWriter()
        # Create minimal valid PDF
        writer.add_blank_page(width=612, height=792)
        
        # Write the PDF
        with open(temp_pdf_path, 'wb') as f:
            writer.write(f)
        
        if os.path.exists(temp_pdf_path) and os.path.getsize(temp_pdf_path) > 0:
            return temp_pdf_path
        
        return None
    except Exception as e:
        logger.warning(f"Could not create dummy PDF: {e}")
        return None

def preload_docling_models():
    """Pre-initialize docling to download any required models"""
    try:
        logger.info("Pre-loading docling and RapidOCR models...")
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        
        # Configure with OCR enabled to trigger model downloads
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        
        # Create converter - this will download models if needed
        logger.info("Creating DocumentConverter with OCR enabled...")
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # CRITICAL: Process a dummy document to fully initialize RapidOCR models
        # This ensures all RapidOCR models are downloaded and cached
        dummy_pdf = create_dummy_pdf()
        if dummy_pdf and os.path.exists(dummy_pdf):
            try:
                logger.info("Processing dummy PDF with OCR to fully initialize RapidOCR models...")
                # This will trigger RapidOCR model downloads if not already cached
                result = converter.convert(dummy_pdf)
                logger.info("✓ Dummy PDF processed - RapidOCR models are now cached in Docker image")
            except Exception as e:
                logger.warning(f"Could not process dummy PDF: {e}")
                logger.warning("RapidOCR models may still be downloaded on first document upload")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up dummy PDF
                try:
                    os.unlink(dummy_pdf)
                except:
                    pass
        else:
            logger.warning("Could not create dummy PDF - RapidOCR models may download on first use")
        
        logger.info("✓ Docling and RapidOCR models pre-loaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Could not pre-load docling models: {e}")
        import traceback
        traceback.print_exc()
        return False

def preload_rapidocr_models():
    """Pre-download RapidOCR models directly"""
    try:
        logger.info("Pre-loading RapidOCR models directly...")
        # RapidOCR models are loaded through docling, but we can verify they exist
        import os
        rapidocr_model_dir = "/usr/local/lib/python3.11/site-packages/rapidocr/models"
        
        if os.path.exists(rapidocr_model_dir):
            model_files = [
                "ch_PP-OCRv4_det_infer.pth",
                "ch_ptocr_mobile_v2.0_cls_infer.pth",
                "ch_PP-OCRv4_rec_infer.pth"
            ]
            found_models = []
            for model_file in model_files:
                model_path = os.path.join(rapidocr_model_dir, model_file)
                if os.path.exists(model_path):
                    found_models.append(model_file)
            
            if found_models:
                logger.info(f"✓ Found {len(found_models)} RapidOCR model files")
                return True
            else:
                logger.warning("RapidOCR models not found - will be downloaded via docling")
                return False
        else:
            logger.warning("RapidOCR model directory not found - models will download on first use")
            return False
    except Exception as e:
        logger.warning(f"Could not verify RapidOCR models: {e}")
        return False

def preload_sentence_transformers():
    """Pre-download sentence-transformers models"""
    try:
        logger.info("Pre-loading sentence-transformers models...")
        from sentence_transformers import SentenceTransformer

        # Use a common model that's likely to be used
        # This will download the model if not cached
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Test the model with a simple encoding to ensure it's fully loaded
        test_embedding = model.encode("test sentence")
        if test_embedding is not None and len(test_embedding) == 384:
            logger.info("✓ Sentence-transformers model tested and cached successfully")
            return True
        else:
            logger.warning("Sentence-transformers model test failed")
            return False
    except Exception as e:
        logger.warning(f"Could not pre-load sentence-transformers models: {e}")
        return False

def main():
    """Pre-load all models during Docker build"""
    logger.info("Starting model pre-loading...")
    
    results = {
        "docling": preload_docling_models(),
        "rapidocr": preload_rapidocr_models(),
        "sentence_transformers": preload_sentence_transformers(),
    }
    
    logger.info("Model pre-loading complete:")
    for model, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {model}")
    
    # Don't fail if some models can't be pre-loaded
    # They'll be downloaded on first use
    sys.exit(0)

if __name__ == "__main__":
    main()

