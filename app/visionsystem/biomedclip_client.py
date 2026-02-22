# app/visionsystem/biomedclip_client.py
"""
BiomedCLIP Client - FIXED (uses open_clip which is already installed)
open-clip-torch>=3.2.0 is in pyproject.toml - this is the REAL implementation.

BiomedCLIP is an open_clip model, NOT a standard transformers model.
Previous code used AutoModel.from_pretrained() which fails because there is no
pytorch_model.bin - weights are in open_clip format only.

Correct loading: open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-...')
"""
import logging
from PIL import Image
import torch

logger = logging.getLogger(__name__)

DENTAL_PATHOLOGY_CANDIDATES = [
    "periapical abscess",
    "dental caries",
    "periodontal bone loss",
    "normal tooth",
    "root canal treatment",
    "dental restoration",
    "impacted tooth",
    "tooth fracture",
]


class BiomedCLIPClient:
    """BiomedCLIP pathology classifier - uses open_clip (correct method for this model)"""

    _instance = None
    _model = None
    _preprocess = None
    _tokenizer = None
    _load_attempted = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _lazy_load_model(self):
        """Load BiomedCLIP via open_clip - the ONLY correct way to load this model."""
        if self._load_attempted:
            return
        self._load_attempted = True

        logger.info("🔄 Loading BiomedCLIP via open_clip (correct method)...")
        try:
            import open_clip

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            self._tokenizer = open_clip.get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            self._model.eval()
            logger.info("✅ BiomedCLIP loaded successfully via open_clip")

        except Exception as e:
            logger.error(f"❌ Failed to load BiomedCLIP: {e}")
            self._model = None
            raise

    def classify_pathology(self, image: Image.Image) -> dict:
        """
        Classify dental pathology using BiomedCLIP CLIP-style similarity scoring.
        Returns prediction, confidence, and all category scores.
        """
        self._lazy_load_model()

        if self._model is None:
            raise RuntimeError("BiomedCLIP model failed to load")

        logger.info("🔬 Running BiomedCLIP dental pathology classification...")

        try:
            # Preprocess image
            image_tensor = self._preprocess(image).unsqueeze(0)

            # Tokenize text labels with radiological context prefix
            text_inputs = [f"an X-ray showing {label}" for label in DENTAL_PATHOLOGY_CANDIDATES]
            text_tokens = self._tokenizer(text_inputs)

            with torch.no_grad():
                image_features = self._model.encode_image(image_tensor)
                text_features = self._model.encode_text(text_tokens)

                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Cosine similarity scaled by 100
                similarity = (100.0 * image_features @ text_features.T)[0]

                # Softmax probabilities
                probs = torch.nn.functional.softmax(similarity, dim=0)

            top_idx = probs.argmax().item()
            top_prob = probs[top_idx].item()

            all_scores = {
                DENTAL_PATHOLOGY_CANDIDATES[i]: round(probs[i].item(), 4)
                for i in range(len(DENTAL_PATHOLOGY_CANDIDATES))
            }

            logger.info(f"✅ BiomedCLIP: {DENTAL_PATHOLOGY_CANDIDATES[top_idx]} ({top_prob:.1%})")

            return {
                "prediction": DENTAL_PATHOLOGY_CANDIDATES[top_idx],
                "confidence": top_prob,
                "all_scores": all_scores,
            }

        except Exception as e:
            logger.error(f"❌ BiomedCLIP classification failed: {e}")
            raise

    def analyze_image(self, image: Image.Image, prompt: str = None) -> dict:
        """Backward compatibility - returns formatted text of classification."""
        result = self.classify_pathology(image)

        lines = [
            "BiomedCLIP Dental Pathology Classification",
            "=" * 45,
            f"Primary Finding: {result['prediction'].upper()}",
            f"Confidence:      {result['confidence']:.1%}",
            "",
            "All Category Scores:",
        ]
        for pathology, score in sorted(
            result["all_scores"].items(), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * int(score * 30)
            lines.append(f"  {pathology:<28} {score:.1%}  {bar}")

        return {
            "text": "\n".join(lines),
            "input_tokens": None,
            "output_tokens": None,
        }


# Global instance
biomedclip_client = BiomedCLIPClient()