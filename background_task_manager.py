import asyncio
import logging
from typing import Set, Dict, Any, Coroutine

logger = logging.getLogger(__name__)

class BackgroundTaskManager:
    """Manages background tasks to ensure proper cleanup on exit"""
    
    def __init__(self):
        self.tasks: Set[asyncio.Task] = set()
        
    def create_task(self, coro: Coroutine) -> asyncio.Task:
        """Create a background task and track it"""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task
        
    async def cancel_all_tasks(self):
        """Cancel all tracked tasks"""
        tasks = list(self.tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
                
        # Wait for all tasks to complete with cancellation
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        self.tasks.clear()