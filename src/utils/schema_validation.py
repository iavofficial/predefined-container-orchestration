# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Created by Lukas Stahlbock in 2026
#  Copyright (c) 2026 IAV GmbH Ingenieurgesellschaft Auto und Verkehr. All rights reserved.
#

import json
from pathlib import Path
from typing import Any

from jsonschema import validate
from jsonschema.exceptions import ValidationError, SchemaError


def load_json_schema(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_with_meta_schema(config: dict, meta_schema_path: Path) -> None:
    schema = load_json_schema(meta_schema_path)
    try:
        validate(instance=config, schema=schema)
    except SchemaError as e:
        raise RuntimeError(f"Invalid JSON Schema: {e.message}") from e
    except ValidationError as e:
        raise ValueError(
            f"Configuration does not conform to schema: {e.message}"
        ) from e
