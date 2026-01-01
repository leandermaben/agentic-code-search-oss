"""Tests for the localization finish tool."""

import os
import tempfile
import pytest
from pathlib import Path

from src.tools.localization_finish import (
    LocalizationFinishAction,
    LocalizationFinishExecutor,
    LocalizationFinishObservation,
)


class TestLocalizationFinishExecutor:
    """Tests for LocalizationFinishExecutor."""

    def setup_method(self):
        """Create executor and temp workspace for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = LocalizationFinishExecutor(workspace_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up temp workspace."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_empty_message(self):
        """Test with empty message."""
        action = LocalizationFinishAction(message="")
        result = self.executor(action)

        assert result.success is False
        assert result.num_locations == 0
        assert "No valid locations" in result.validation_message

    def test_malformed_input(self):
        """Test with malformed input."""
        action = LocalizationFinishAction(message="random text without proper format")
        result = self.executor(action)

        assert result.success is False
        # Malformed input results in no locations found
        assert "No valid locations" in result.validation_message

    def test_file_only_valid(self):
        """Test valid file-level localization (no class or function)."""
        # Create the file in workspace
        file_path = "test_file.py"
        Path(self.temp_dir, file_path).touch()

        action = LocalizationFinishAction(message=f"""```
{file_path}
```""")
        result = self.executor(action)

        assert result.success is True
        assert result.num_locations == 1
        assert "Successfully submitted" in result.validation_message

    def test_file_and_class_valid(self):
        """Test valid class-level localization (no function)."""
        file_path = "test_file.py"
        Path(self.temp_dir, file_path).touch()

        action = LocalizationFinishAction(message=f"""```
{file_path}
class: MyClass
```""")
        result = self.executor(action)

        assert result.success is True
        assert result.num_locations == 1

    def test_file_and_function_valid(self):
        """Test valid function-level localization (no class)."""
        file_path = "test_file.py"
        Path(self.temp_dir, file_path).touch()

        action = LocalizationFinishAction(message=f"""```
{file_path}
function: my_function
```""")
        result = self.executor(action)

        assert result.success is True
        assert result.num_locations == 1

    def test_complete_localization_valid(self):
        """Test complete localization with file, class, and function."""
        file_path = "test_file.py"
        Path(self.temp_dir, file_path).touch()

        action = LocalizationFinishAction(message=f"""```
{file_path}
class: MyClass
function: my_method
```""")
        result = self.executor(action)

        assert result.success is True
        assert result.num_locations == 1

    def test_multiple_locations_mixed_formats(self):
        """Test multiple locations with different formats."""
        # Create files
        for i in range(1, 5):
            Path(self.temp_dir, f"file{i}.py").touch()

        action = LocalizationFinishAction(message="""```
file1.py

file2.py
class: ClassA

file3.py
function: func_b

file4.py
class: ClassC
function: method_d
```""")
        result = self.executor(action)

        assert result.success is True
        assert result.num_locations == 4
        assert "Successfully submitted 4 location(s)" in result.validation_message

    def test_missing_file_warning(self):
        """Test that missing files trigger a warning."""
        action = LocalizationFinishAction(message="""```
nonexistent_file.py
```""")
        result = self.executor(action)

        # Should still parse but with warning
        assert result.success is False
        assert result.num_locations == 1
        assert "not found in workspace" in result.validation_message
        assert "nonexistent_file.py" in result.validation_message

    def test_some_files_missing(self):
        """Test when some files exist and some don't."""
        # Create only one file
        Path(self.temp_dir, "exists.py").touch()

        action = LocalizationFinishAction(message="""```
exists.py

missing.py
```""")
        result = self.executor(action)

        assert result.success is False
        assert result.num_locations == 2
        assert "missing.py" in result.validation_message

    def test_no_backticks(self):
        """Test that output without backticks fails parsing."""
        action = LocalizationFinishAction(message="""file.py
class: MyClass
""")
        result = self.executor(action)

        # Should fail because parse_simple_output expects backticks
        assert result.success is False

    def test_executor_without_workspace(self):
        """Test executor without workspace validation."""
        executor = LocalizationFinishExecutor(workspace_dir=None)

        action = LocalizationFinishAction(message="""```
any_file.py
```""")
        result = executor(action)

        # Should succeed without file existence check
        assert result.success is True
        assert result.num_locations == 1

    def test_nested_file_path(self):
        """Test with nested file paths."""
        # Create nested structure
        nested_dir = Path(self.temp_dir, "src", "utils")
        nested_dir.mkdir(parents=True)
        Path(nested_dir, "helper.py").touch()

        action = LocalizationFinishAction(message="""```
src/utils/helper.py
function: process
```""")
        result = self.executor(action)

        assert result.success is True
        assert result.num_locations == 1

    def test_dotted_function_name(self):
        """Test with ClassName.method_name format."""
        file_path = "test.py"
        Path(self.temp_dir, file_path).touch()

        action = LocalizationFinishAction(message=f"""```
{file_path}
function: MyClass.my_method
```""")
        result = self.executor(action)

        assert result.success is True
        assert result.num_locations == 1
        # Should parse as class=MyClass, function=my_method
        assert result.details["locations"][0]["class"] == "MyClass"
        assert result.details["locations"][0]["function"] == "my_method"

    def test_multiple_missing_files_truncated(self):
        """Test that many missing files are truncated in message."""
        action = LocalizationFinishAction(message="""```
file1.py
file2.py
file3.py
file4.py
file5.py
file6.py
file7.py
```""")
        result = self.executor(action)

        assert result.success is False
        assert result.num_locations == 7
        # Should show truncation message
        assert "and 2 more" in result.validation_message


class TestLocalizationFinishObservation:
    """Tests for LocalizationFinishObservation visualization."""

    def test_success_visualization(self):
        """Test visualization of successful observation."""
        obs = LocalizationFinishObservation(
            success=True,
            num_locations=3,
            validation_message="Successfully submitted 3 location(s)."
        )
        text = obs.visualize

        assert "✓" in str(text)
        assert "3 location(s)" in str(text)

    def test_failure_visualization(self):
        """Test visualization of failed observation."""
        obs = LocalizationFinishObservation(
            success=False,
            num_locations=0,
            validation_message="Missing file paths"
        )
        text = obs.visualize

        assert "✗" in str(text)
        assert "Missing file paths" in str(text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
