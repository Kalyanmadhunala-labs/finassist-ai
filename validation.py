def validate_sql_result(result):
    if not isinstance(result, dict):
        return False

    if result.get("tool") != "sql_tool":
        return False

    tool_result = result.get("result")

    if not isinstance(tool_result, dict):
        return False

    status = tool_result.get("status")

    if status == "DATA_FOUND":
        return True

    if status in ["NO_DATA", "ERROR"]:
        return False

    return False


def validate_rag_result(result):
    if not isinstance(result, dict):
        return False

    if result.get("tool") != "rag_tool":
        return False

    rag_result = result.get("result")

    if not isinstance(rag_result, str):
        return False

    if not rag_result.strip():
        return False

    return True