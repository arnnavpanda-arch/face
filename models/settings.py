"""
Institutional Settings Model
Provides dynamic configuration for institutional identity, AI thresholds,
cooldown periods, and active camera endpoints without hardcoded strings.
"""

import time
from database import get_settings_collection
from config import Config

class SettingsModel:
    CONFIG_KEY = "institutional_config"
    _CACHE = None
    _CACHE_TIME = 0.0
    _CACHE_TTL = 30.0  # seconds

    @staticmethod
    def get_settings() -> dict:
        now = time.time()
        if SettingsModel._CACHE is not None and (now - SettingsModel._CACHE_TIME < SettingsModel._CACHE_TTL):
            return SettingsModel._CACHE

        col = get_settings_collection()
        doc = col.find_one({"key": SettingsModel.CONFIG_KEY})
        if not doc:
            # Fallback default
            doc = {
                "institution_name": Config.DEFAULT_INSTITUTION_NAME,
                "campus_location": Config.DEFAULT_CAMPUS_LOCATION,
                "similarity_threshold": Config.SIMILARITY_THRESHOLD,
                "cooldown_period_seconds": Config.COOLDOWN_PERIOD_SECONDS,
                "face_detection_confidence": Config.FACE_DETECTION_CONFIDENCE,
                "min_face_size": Config.MIN_FACE_SIZE,
                "min_blur_variance": Config.MIN_BLUR_VARIANCE,
                "anti_spoofing_enabled": Config.ANTI_SPOOFING_ENABLED,
                "cameras": [
                    {"id": "cam_gate_north", "name": "North Main Entrance - Lane 1", "location": "North Gate", "status": "active", "source": "0"},
                    {"id": "cam_gate_south", "name": "South Entrance - Lane 2", "location": "South Gate", "status": "active", "source": "1"},
                    {"id": "cam_library", "name": "Central Library Entry", "location": "Library Building", "status": "active", "source": "2"}
                ],
                "data_retention": {
                    "entry_logs_days": Config.RETENTION_DAYS_ENTRY_LOGS,
                    "unknown_logs_days": Config.RETENTION_DAYS_UNKNOWN_LOGS
                }
            }
        else:
            doc["_id"] = str(doc["_id"])

        SettingsModel._CACHE = doc
        SettingsModel._CACHE_TIME = now
        return doc

    @staticmethod
    def update_settings(updates: dict) -> bool:
        SettingsModel._CACHE = None  # Invalidate cache immediately on update
        col = get_settings_collection()
        updates["updated_at"] = time.time()
        res = col.update_one(
            {"key": SettingsModel.CONFIG_KEY},
            {"$set": updates},
            upsert=True
        )
        return res.acknowledged
