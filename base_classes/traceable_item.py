from abc import ABC, abstractmethod
from datetime import datetime

class TimeTraceableItem(ABC):
    """
    A class that represents an item that can be traced over time.
    """
    _created_time: datetime
    _last_accessed: datetime
    _last_write: datetime
    def __init__(self):
        """
        Initialize the TimeTraceableItem with the current time.
        """
        self._created_time = datetime.now()
        self._last_accessed = self._created_time
        self._last_write = self._created_time
        
    @property
    def created_time(self):
        return self._created_time
    @property
    def last_accessed(self):
        return self._last_accessed
    @property
    def last_write(self):
        return self._last_write
    
    @last_accessed.setter
    def last_accessed(self, value: datetime):
        """
        Set the last accessed time to the current time.
        """
        self._last_accessed = value
    @last_write.setter
    def last_write(self, value: datetime):
        """
        Set the last write time to the current time.
        """
        self._last_accessed = value
        self._last_write = value
