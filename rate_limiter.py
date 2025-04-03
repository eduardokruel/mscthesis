import time
import threading
from datetime import datetime, timedelta

class RateLimiter:
    """
    Manages rate limits for different API models
    """
    def __init__(self):
        # Model-specific rate limits (requests per minute, requests per day)
        self.rate_limits = {
            "gpt-4o-mini": {"rpm": 500, "rpd": 10000},
            "gpt-3.5-turbo": {"rpm": 3500, "rpd": 80000},
            "deepseek-chat": {"rpm": 100, "rpd": 10000}  # Adjust based on DeepSeek's limits
        }
        
        # Usage tracking
        self.usage = {}
        self.lock = threading.RLock()
        
        # Initialize usage counters for all models
        for model in self.rate_limits:
            self.usage[model] = {
                "minute": {"count": 0, "reset_time": datetime.now() + timedelta(minutes=1)},
                "day": {"count": 0, "reset_time": datetime.now() + timedelta(days=1)}
            }
    
    def check_and_update(self, model):
        """
        Check if a request can be made and update usage counters
        
        Args:
            model: The model name
            
        Returns:
            (can_request, wait_time): Boolean indicating if request can be made, and time to wait if not
        """
        with self.lock:
            # Default to some reasonable limits if model not in our list
            if model not in self.rate_limits:
                model_limits = {"rpm": 100, "rpd": 10000}
            else:
                model_limits = self.rate_limits[model]
            
            # Initialize model usage if not present
            if model not in self.usage:
                self.usage[model] = {
                    "minute": {"count": 0, "reset_time": datetime.now() + timedelta(minutes=1)},
                    "day": {"count": 0, "reset_time": datetime.now() + timedelta(days=1)}
                }
            
            # Check and reset minute counter if needed
            now = datetime.now()
            if now >= self.usage[model]["minute"]["reset_time"]:
                self.usage[model]["minute"]["count"] = 0
                self.usage[model]["minute"]["reset_time"] = now + timedelta(minutes=1)
            
            # Check and reset day counter if needed
            if now >= self.usage[model]["day"]["reset_time"]:
                self.usage[model]["day"]["count"] = 0
                self.usage[model]["day"]["reset_time"] = now + timedelta(days=1)
            
            # Check if we're at the rate limits
            minute_limited = self.usage[model]["minute"]["count"] >= model_limits["rpm"]
            day_limited = self.usage[model]["day"]["count"] >= model_limits["rpd"]
            
            if minute_limited or day_limited:
                # Calculate wait time
                if minute_limited:
                    wait_time = (self.usage[model]["minute"]["reset_time"] - now).total_seconds()
                else:
                    wait_time = (self.usage[model]["day"]["reset_time"] - now).total_seconds()
                
                return False, wait_time
            
            # Update counters
            self.usage[model]["minute"]["count"] += 1
            self.usage[model]["day"]["count"] += 1
            
            return True, 0
    
    def get_usage_stats(self):
        """Get current usage statistics for all models"""
        with self.lock:
            stats = {}
            for model, usage_data in self.usage.items():
                stats[model] = {
                    "minute": {
                        "count": usage_data["minute"]["count"],
                        "limit": self.rate_limits.get(model, {"rpm": "unknown"})["rpm"],
                        "reset_in": (usage_data["minute"]["reset_time"] - datetime.now()).total_seconds()
                    },
                    "day": {
                        "count": usage_data["day"]["count"],
                        "limit": self.rate_limits.get(model, {"rpd": "unknown"})["rpd"],
                        "reset_in": (usage_data["day"]["reset_time"] - datetime.now()).total_seconds()
                    }
                }
            return stats

# Global rate limiter instance
rate_limiter = RateLimiter()

def get_rate_limiter():
    """Get the global rate limiter instance"""
    return rate_limiter 