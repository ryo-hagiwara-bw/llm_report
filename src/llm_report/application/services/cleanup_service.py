"""Service for cleaning up temporary files and generated images."""

import os
import shutil
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CleanupResult:
    """Result of cleanup operation."""
    success: bool
    deleted_files: List[str]
    deleted_directories: List[str]
    error: Optional[str] = None


class CleanupService:
    """Service for cleaning up temporary files and generated content."""
    
    def __init__(self, temp_dir: str = "tmp", output_dir: str = "output"):
        """Initialize the cleanup service.
        
        Args:
            temp_dir: Directory for temporary files
            output_dir: Directory for generated output files
        """
        self.temp_dir = temp_dir
        self.output_dir = output_dir
    
    def cleanup_temp_files(self) -> CleanupResult:
        """Clean up all temporary files.
        
        Returns:
            CleanupResult with cleanup details
        """
        try:
            deleted_files = []
            deleted_directories = []
            
            if os.path.exists(self.temp_dir):
                # List all files in temp directory
                for filename in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_files.append(file_path)
                
                # Remove temp directory if empty
                if not os.listdir(self.temp_dir):
                    os.rmdir(self.temp_dir)
                    deleted_directories.append(self.temp_dir)
            
            return CleanupResult(
                success=True,
                deleted_files=deleted_files,
                deleted_directories=deleted_directories
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                deleted_files=[],
                deleted_directories=[],
                error=str(e)
            )
    
    def cleanup_output_files(self) -> CleanupResult:
        """Clean up all generated output files.
        
        Returns:
            CleanupResult with cleanup details
        """
        try:
            deleted_files = []
            deleted_directories = []
            
            if os.path.exists(self.output_dir):
                # List all files in output directory
                for filename in os.listdir(self.output_dir):
                    file_path = os.path.join(self.output_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_files.append(file_path)
                
                # Remove output directory if empty
                if not os.listdir(self.output_dir):
                    os.rmdir(self.output_dir)
                    deleted_directories.append(self.output_dir)
            
            return CleanupResult(
                success=True,
                deleted_files=deleted_files,
                deleted_directories=deleted_directories
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                deleted_files=[],
                deleted_directories=[],
                error=str(e)
            )
    
    def cleanup_all(self) -> CleanupResult:
        """Clean up all temporary and output files.
        
        Returns:
            CleanupResult with cleanup details
        """
        try:
            # Clean up temp files
            temp_result = self.cleanup_temp_files()
            
            # Clean up output files
            output_result = self.cleanup_output_files()
            
            # Combine results
            all_deleted_files = temp_result.deleted_files + output_result.deleted_files
            all_deleted_directories = temp_result.deleted_directories + output_result.deleted_directories
            
            success = temp_result.success and output_result.success
            error = None
            if not success:
                errors = []
                if temp_result.error:
                    errors.append(f"Temp cleanup error: {temp_result.error}")
                if output_result.error:
                    errors.append(f"Output cleanup error: {output_result.error}")
                error = "; ".join(errors)
            
            return CleanupResult(
                success=success,
                deleted_files=all_deleted_files,
                deleted_directories=all_deleted_directories,
                error=error
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                deleted_files=[],
                deleted_directories=[],
                error=str(e)
            )
    
    def cleanup_old_files(self, days: int = 1) -> CleanupResult:
        """Clean up files older than specified days.
        
        Args:
            days: Number of days to keep files
            
        Returns:
            CleanupResult with cleanup details
        """
        try:
            deleted_files = []
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Clean up old temp files
            if os.path.exists(self.temp_dir):
                for filename in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, filename)
                    if os.path.isfile(file_path):
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time < cutoff_time:
                            os.remove(file_path)
                            deleted_files.append(file_path)
            
            # Clean up old output files
            if os.path.exists(self.output_dir):
                for filename in os.listdir(self.output_dir):
                    file_path = os.path.join(self.output_dir, filename)
                    if os.path.isfile(file_path):
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time < cutoff_time:
                            os.remove(file_path)
                            deleted_files.append(file_path)
            
            return CleanupResult(
                success=True,
                deleted_files=deleted_files,
                deleted_directories=[]
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                deleted_files=[],
                deleted_directories=[],
                error=str(e)
            )
    
    def get_file_counts(self) -> dict:
        """Get counts of files in temp and output directories.
        
        Returns:
            Dictionary with file counts
        """
        try:
            temp_count = 0
            output_count = 0
            
            if os.path.exists(self.temp_dir):
                temp_count = len([f for f in os.listdir(self.temp_dir) if os.path.isfile(os.path.join(self.temp_dir, f))])
            
            if os.path.exists(self.output_dir):
                output_count = len([f for f in os.listdir(self.output_dir) if os.path.isfile(os.path.join(self.output_dir, f))])
            
            return {
                "temp_files": temp_count,
                "output_files": output_count,
                "total_files": temp_count + output_count
            }
            
        except Exception as e:
            return {
                "temp_files": 0,
                "output_files": 0,
                "total_files": 0,
                "error": str(e)
            }
