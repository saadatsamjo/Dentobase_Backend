# app/cdss_engine/tooth_validator.py

"""
Tooth Number Validator - Enforce Clinical Consistency
This is a DETERMINISTIC validation layer (no AI involved)
"""
import logging
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ToothValidator:
    """
    Validate and correct tooth number inconsistencies.
    Uses anatomical rules to enforce clinical accuracy.
    """
    
    # Universal Numbering System (US/Canadian) - most common in North America
    UNIVERSAL_QUADRANTS = {
        "UR": list(range(1, 9)),    # Upper Right: 1-8
        "UL": list(range(9, 17)),   # Upper Left: 9-16
        "LL": list(range(17, 25)),  # Lower Left: 17-24
        "LR": list(range(25, 33)),  # Lower Right: 25-32
    }
    
    # FDI World Dental Federation Numbering (International)
    FDI_QUADRANTS = {
        "UR": list(range(11, 19)),  # 11-18
        "UL": list(range(21, 29)),  # 21-28
        "LL": list(range(31, 39)),  # 31-38
        "LR": list(range(41, 49)),  # 41-48
    }
    
    def validate_and_fix(
        self,
        focus_tooth: str,
        vision_teeth: List[str],
        diagnosis: str,
        image_obs: Dict
    ) -> Dict:
        """
        Main validation method - enforces consistency across all outputs.
        
        Args:
            focus_tooth: The tooth number user requested (e.g., "47")
            vision_teeth: Teeth identified by vision model (e.g., ["23", "24", "25"])
            diagnosis: The diagnosis text from LLM
            image_obs: Full image observation dict
        
        Returns:
            dict with:
            - valid: bool (True if no issues)
            - issues: List[str] (descriptions of problems found)
            - fixes: Dict (corrected values)
        """
        issues = []
        fixes = {}
        
        # Convert to int for processing
        try:
            focus_num = int(focus_tooth) if focus_tooth else None
            vision_nums = [int(t) for t in vision_teeth if t and str(t).strip()]
        except (ValueError, TypeError) as e:
            issues.append(f"Invalid tooth number format: {e}")
            return {"valid": False, "issues": issues, "fixes": fixes}
        
        # RULE 1: Vision teeth must be in same quadrant as focus tooth
        if focus_num and vision_nums:
            quadrant = self._get_quadrant(focus_num)
            valid_teeth = [t for t in vision_nums if self._get_quadrant(t) == quadrant]
            
            if len(valid_teeth) < len(vision_nums):
                removed = set(vision_nums) - set(valid_teeth)
                issues.append(
                    f"Removed teeth {removed} - not in same quadrant as focus tooth #{focus_num}"
                )
                fixes["corrected_teeth_visible"] = [str(t) for t in valid_teeth] or [focus_tooth]
                logger.warning(f"⚠️  Tooth quadrant mismatch - removed {removed}")
        
        # RULE 2: Periapical X-ray can't show more than 5 teeth
        if len(vision_nums) > 5:
            issues.append(
                f"Periapical X-ray cannot show {len(vision_nums)} teeth (maximum 5)"
            )
            # Keep teeth closest to focus tooth
            if focus_num:
                distances = [(abs(t - focus_num), t) for t in vision_nums]
                closest = sorted(distances)[:3]
                fixes["corrected_teeth_visible"] = [str(t[1]) for t in closest]
                logger.warning(f"⚠️  Too many teeth - keeping 3 closest to #{focus_num}")
            else:
                # No focus tooth - just keep first 3
                fixes["corrected_teeth_visible"] = [str(t) for t in vision_nums[:3]]
        
        # RULE 3: Diagnosis MUST mention focus tooth
        if focus_num:
            # Check if tooth number appears in diagnosis
            tooth_mentioned = (
                str(focus_num) in diagnosis or
                f"#{focus_num}" in diagnosis or
                f"tooth {focus_num}" in diagnosis.lower()
            )
            
            if not tooth_mentioned:
                issues.append(
                    f"Diagnosis missing focus tooth #{focus_num}"
                )
                # Add tooth number to diagnosis
                diagnosis_clean = diagnosis.rstrip('.')
                fixes["corrected_diagnosis"] = f"{diagnosis_clean}, tooth #{focus_num}."
                logger.warning(f"⚠️  Diagnosis missing tooth number - adding #{focus_num}")
        
        # RULE 4: Vision model's focused_tooth must match requested focus
        if focus_num and image_obs.get("structured_findings"):
            vision_focus = image_obs["structured_findings"].get("focused_tooth")
            if vision_focus:
                try:
                    vision_focus_num = int(vision_focus)
                    if vision_focus_num != focus_num:
                        issues.append(
                            f"Vision model focused on tooth #{vision_focus} instead of requested #{focus_num}"
                        )
                        fixes["corrected_focused_tooth"] = focus_tooth
                        logger.warning(f"⚠️  Vision focus mismatch: {vision_focus} vs {focus_num}")
                except (ValueError, TypeError):
                    pass
        
        # RULE 5: Validate FDI vs Universal numbering consistency
        numbering_system = self._detect_numbering_system(focus_num)
        if numbering_system:
            for tooth in vision_nums:
                tooth_system = self._detect_numbering_system(tooth)
                if tooth_system and tooth_system != numbering_system:
                    issues.append(
                        f"Mixed numbering systems detected: {numbering_system} vs {tooth_system}"
                    )
                    logger.warning(f"⚠️  Mixed numbering: {numbering_system} and {tooth_system}")
        
        # Log results
        if issues:
            logger.warning(f"⚠️  Validation found {len(issues)} issue(s)")
            for i, issue in enumerate(issues, 1):
                logger.warning(f"     {i}. {issue}")
        else:
            logger.info(f"✅ Validation passed - no issues found")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "fixes": fixes
        }
    
    def _get_quadrant(self, tooth_num: int) -> Optional[str]:
        """
        Determine which quadrant a tooth is in.
        Supports both Universal (1-32) and FDI (11-48) systems.
        """
        # Universal system (1-32)
        if 1 <= tooth_num <= 8:
            return "UR"
        if 9 <= tooth_num <= 16:
            return "UL"
        if 17 <= tooth_num <= 24:
            return "LL"
        if 25 <= tooth_num <= 32:
            return "LR"
        
        # FDI system (11-48)
        if 11 <= tooth_num <= 18:
            return "UR"
        if 21 <= tooth_num <= 28:
            return "UL"
        if 31 <= tooth_num <= 38:
            return "LL"
        if 41 <= tooth_num <= 48:
            return "LR"
        
        logger.warning(f"⚠️  Tooth #{tooth_num} not in valid range")
        return None
    
    def _detect_numbering_system(self, tooth_num: int) -> Optional[str]:
        """Detect if tooth uses Universal or FDI numbering"""
        if 1 <= tooth_num <= 32:
            return "Universal"
        if 11 <= tooth_num <= 48:
            return "FDI"
        return None
    
    def get_adjacent_teeth(self, tooth_num: int, range_size: int = 2) -> List[int]:
        """
        Get teeth within N positions of target tooth.
        Useful for periapical X-ray expectations.
        """
        system = self._detect_numbering_system(tooth_num)
        
        if system == "Universal":
            # Universal system (1-32)
            return list(range(
                max(1, tooth_num - range_size), 
                min(32, tooth_num + range_size) + 1
            ))
        elif system == "FDI":
            # FDI system - stay within quadrant
            quadrant = self._get_quadrant(tooth_num)
            if quadrant in self.FDI_QUADRANTS:
                quad_teeth = self.FDI_QUADRANTS[quadrant]
                return [t for t in quad_teeth if abs(t - tooth_num) <= range_size]
        
        return [tooth_num]
    
    def is_valid_periapical_combination(
        self, 
        teeth: List[int], 
        focus_tooth: int
    ) -> bool:
        """
        Check if a list of teeth is a valid periapical X-ray combination.
        
        Rules:
        - All teeth must be in same quadrant
        - Maximum 5 teeth
        - Must be adjacent (no gaps > 2 teeth)
        """
        if len(teeth) > 5:
            return False
        
        # Check same quadrant
        focus_quad = self._get_quadrant(focus_tooth)
        if not all(self._get_quadrant(t) == focus_quad for t in teeth):
            return False
        
        # Check adjacency
        sorted_teeth = sorted(teeth)
        for i in range(len(sorted_teeth) - 1):
            if sorted_teeth[i+1] - sorted_teeth[i] > 2:
                return False  # Gap too large
        
        return True


# Global instance
tooth_validator = ToothValidator()