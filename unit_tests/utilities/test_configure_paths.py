from CuISOX.utilities.configure_paths import (
	project_path,
	DataPaths)
from pathlib import Path

import pytest

def get_this_file_path():
	return Path(__file__).parent.resolve()

def test_project_path():
	resulting_project_path = project_path()
	assert str(resulting_project_path).find("cuBlackDream") != -1

def test_list_all_files_in_directory():
	this_file_directory_contents = DataPaths.list_all_files_in_directory(
		get_this_file_path())

	assert any(
		[
			str(path).find("test_configure_paths.py") != -1
				for path in this_file_directory_contents])

def test_get_path_with_substring_gets():
	this_file_directory_contents = DataPaths.list_all_files_in_directory(
		get_this_file_path())

	result = DataPaths.get_path_with_substring(
		this_file_directory_contents,
		input_substring="test_configure_paths")

	assert len(result) == 1
	assert result[0].name == "test_configure_paths.py"