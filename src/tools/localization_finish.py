"""Custom finish tool for code localization tasks.

This tool allows the agent to submit localization results in a flexible format where:
- File path is required
- Class name is optional
- Function name is optional
"""

import os
from typing import TYPE_CHECKING
from collections.abc import Sequence

from pydantic import Field
from rich.text import Text

from openhands.sdk import (
    Action,
    Observation,
    ToolDefinition,
)
from openhands.sdk.tool import ToolExecutor

from src.tools import tool

if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation


class LocalizationFinishAction(Action):
    """Action for submitting final localization results."""

    message: str = Field(
        description="""Your final localization results in the specified format.

Format:
```
path/to/file1.py
class: ClassName
function: method_name

path/to/file2.py
function: standalone_function

path/to/file3.py
```

Requirements:
- Each location must have a file path
- Class name is optional (omit if change is file-level or function is standalone)
- Function name is optional (omit if change is file-level or class-level only)
- Wrap your answer in triple backticks

Example for different scenarios:
- File-level change (imports, globals): Just list the file
- Class-level change (new method, attributes): List file + class
- Method change: List file + class + function
- Standalone function: List file + function
"""
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this action."""
        content = Text()
        content.append("Submitting localization results:\n", style="bold blue")
        content.append(self.message)
        return content


class LocalizationFinishObservation(Observation):
    """Observation returned after submitting localization results."""

    success: bool = Field(default=True, description="Whether submission was successful")
    num_locations: int = Field(default=0, description="Number of locations submitted")
    validation_message: str = Field(default="", description="Validation feedback")
    details: dict = Field(default_factory=dict, description="Additional details")

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this observation."""
        content = Text()
        if self.success:
            content.append(f"✓ Successfully submitted {self.num_locations} location(s)\n", style="bold green")
        else:
            content.append(f"✗ Submission failed\n", style="bold red")
            content.append(f"{self.validation_message}\n", style="yellow")
        return content


def parse_localization_output(raw_output: str) -> list[dict]:
    """Parse localization output with optional class and function.

    This is an enhanced version of parse_simple_output that handles:
    - File-only entries (no class or function)
    - File + class entries (no function)
    - File + function entries (no class)
    - File + class + function entries

    Args:
        raw_output: Raw text output to parse

    Returns:
        List of dictionaries with 'file', 'class' (optional), 'function' (optional)
    """
    # Remove triple backticks and whitespace
    raw_output = raw_output.strip("` \n")

    locations = []
    current_file = None
    current_class = None
    current_function = None

    lines = raw_output.strip().split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            # Empty line - save current location if we have a file
            if current_file:
                locations.append({
                    "file": current_file,
                    "class": current_class,
                    "function": current_function,
                })
                current_file = None
                current_class = None
                current_function = None
            continue

        # Check if this is a file path (ends with .py)
        if line.endswith(".py"):
            # Save previous location if exists
            if current_file:
                locations.append({
                    "file": current_file,
                    "class": current_class,
                    "function": current_function,
                })
            # Start new location
            current_file = line
            current_class = None
            current_function = None
            continue

        # Parse class declaration
        if line.startswith("class:"):
            class_name = line[len("class:"):].strip()
            current_class = class_name
            continue

        # Parse function/method declaration
        if line.startswith("function:") or line.startswith("method:"):
            func_text = line.split(":", 1)[1].strip()
            func_name = func_text.split()[0].strip("() ")

            # Check if function includes class prefix (e.g., "MyClass.my_method")
            if "." in func_name:
                parts = func_name.split(".", 1)
                current_class = parts[0]
                current_function = parts[1]
            else:
                current_function = func_name
            continue

    # Don't forget the last location
    if current_file:
        locations.append({
            "file": current_file,
            "class": current_class,
            "function": current_function,
        })

    return locations


class LocalizationFinishExecutor(ToolExecutor):
    """Executor for localization finish tool with validation."""

    def __init__(self, workspace_dir: str | None = None):
        """Initialize the executor.

        Args:
            workspace_dir: Optional workspace directory to validate file existence.
        """
        self.workspace_dir = workspace_dir

    def __call__(
        self,
        action: LocalizationFinishAction,
        conversation: "BaseConversation | None" = None,
    ) -> LocalizationFinishObservation:
        """Execute the finish action with validation.

        Args:
            action: The localization finish action to execute
            conversation: Optional conversation context

        Returns:
            LocalizationFinishObservation with validation results
        """

        try:
            # Parse the output to validate format
            locations = parse_localization_output(action.message)
            num_locs = len(locations)

            # Validation 1: Check if any locations were found
            if num_locs == 0:
                return LocalizationFinishObservation(
                    success=False,
                    num_locations=0,
                    validation_message=(
                        "No valid locations found. Please provide at least one file path "
                        "in the correct format wrapped in triple backticks."
                    ),
                    details={"error": "empty_output"}
                )

            # Validation 2: Check each location has a file path
            errors = []
            for i, loc in enumerate(locations):
                if not loc.get('file'):
                    errors.append(f"Location {i+1} is missing a file path")

            if errors:
                return LocalizationFinishObservation(
                    success=False,
                    num_locations=0,
                    validation_message="\n".join(errors),
                    details={"error": "missing_file_paths", "locations": locations}
                )

            # Validation 3: Check file existence (if workspace provided)
            if self.workspace_dir:
                missing_files = []
                for loc in locations:
                    file_path = loc['file']
                    full_path = os.path.join(self.workspace_dir, file_path)
                    if not os.path.exists(full_path):
                        missing_files.append(file_path)

                if missing_files:
                    return LocalizationFinishObservation(
                        success=False,
                        num_locations=num_locs,
                        validation_message=(
                            f"Warning: {len(missing_files)} file(s) not found in workspace:\n" +
                            "\n".join(f"  - {f}" for f in missing_files[:5]) +
                            (f"\n  ... and {len(missing_files) - 5} more" if len(missing_files) > 5 else "")
                        ),
                        details={
                            "warning": "files_not_found",
                            "missing_files": missing_files,
                            "locations": locations
                        }
                    )

            # Success!
            return LocalizationFinishObservation(
                success=True,
                num_locations=num_locs,
                validation_message=f"Successfully submitted {num_locs} location(s).",
                details={"locations": locations}
            )

        except Exception as e:
            # Parsing failed
            return LocalizationFinishObservation(
                success=False,
                num_locations=0,
                validation_message=(
                    f"Error parsing output: {str(e)}\n\n"
                    "Please ensure your output follows the correct format:\n"
                    "```\n"
                    "path/to/file.py\n"
                    "class: ClassName\n"
                    "function: method_name\n"
                    "```"
                ),
                details={"error": "parse_error", "exception": str(e)}
            )


TOOL_DESCRIPTION = """Submit your final code localization results.

Use this tool when you have identified all relevant files, classes, and functions
that need to be modified to address the issue described in the problem statement.

Format your results as follows:
```
path/to/file1.py
class: ClassName
function: method_name

path/to/file2.py
function: standalone_function

path/to/file3.py
```

Requirements:
- Wrap your output in triple backticks (```)
- Each location must start with a file path
- Class name is OPTIONAL - include only if the change is within a specific class
- Function name is OPTIONAL - include only if the change is at function/method level

When to omit class/function:
- File-level only (imports, globals, new classes): List just the file
- Class-level only (new methods, attributes): List file + class (no function)
- Standalone function: List file + function (no class)
- Method in class: List file + class + function

The tool will validate your submission and provide feedback if the format is incorrect.
"""


class LocalizationFinishTool(ToolDefinition[LocalizationFinishAction, LocalizationFinishObservation]):
    """Tool for submitting final code localization results."""

    @classmethod
    def create(
        cls,
        conv_state,
        workspace_dir: str | None = None,
        **params
    ) -> Sequence["LocalizationFinishTool"]:
        """Create LocalizationFinishTool instance.

        Args:
            conv_state: Conversation state (provides workspace info)
            workspace_dir: Optional workspace directory override
            **params: Additional parameters

        Returns:
            A sequence containing a single LocalizationFinishTool instance.
        """
        # Get workspace from conv_state if not provided
        if workspace_dir is None and hasattr(conv_state, 'workspace'):
            workspace_dir = str(conv_state.workspace.working_dir)

        executor = LocalizationFinishExecutor(workspace_dir=workspace_dir)

        return [
            cls(
                action_type=LocalizationFinishAction,
                observation_type=LocalizationFinishObservation,
                description=TOOL_DESCRIPTION,
                executor=executor,
            )
        ]


@tool(name="finish")
def _make_localization_finish_tool(conv_state) -> list[ToolDefinition]:
    """Create localization finish tool.

    This replaces the default finish tool with a localization-specific version
    that validates the output format.
    """
    return LocalizationFinishTool.create(conv_state)
