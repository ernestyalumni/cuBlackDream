from CuISOX.utilities.configure_paths import (
	project_path,
	DataPaths)

import pytest

def test_project_path():
	resulting_project_path = project_path()
	assert str(resulting_project_path).find("cuBlackDream") != -1
