# app/visionsystem/image_processor.py (Updated)
from PIL import Image
import io
from typing import Union
from config.visionconfig import vision_settings
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handle image preprocessing for clinical analysis."""
    
    @staticmethod
    def preprocess_image(image_bytes: bytes, max_size: int = None) -> Image.Image:
        """
        Preprocess uploaded image for Florence-2.
        """
        max_size = max_size or vision_settings.MAX_IMAGE_SIZE
        
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Loaded image: {image.format}, mode={image.mode}, size={image.size}")
            
            # Convert to RGB (Florence-2 requires RGB)
            if image.mode != 'RGB':
                logger.info(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Resize if too large (maintain aspect ratio)
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {new_size}")
            
            # Ensure image is in proper format for transformers
            # Save to buffer and reload to ensure clean format
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            image = Image.open(buffer)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise ValueError(f"Invalid image file: {e}")
    
    @staticmethod
    def validate_image(content_type: str, size_bytes: int) -> bool:
        """Validate uploaded image meets requirements."""
        if content_type not in vision_settings.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {content_type}. Use: {vision_settings.SUPPORTED_FORMATS}")
        
        # 10MB limit
        if size_bytes > 10 * 1024 * 1024:
            raise ValueError("Image too large. Max 10MB.")
        
        return True