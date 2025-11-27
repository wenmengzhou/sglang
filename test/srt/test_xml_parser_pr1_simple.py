"""
Simple test runner for XML Parser PR1 (without pytest dependency)
"""

import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

from sglang.srt.entrypoints.openai.protocol import Tool, ToolFunction, ToolFunctionParameters
from sglang.srt.function_call.qwen3_coder_new_detector import (
    StreamingXMLToolCallParser,
    Qwen3CoderDetector,
)


def test_initialization():
    """Test parser initializes with correct default state"""
    print("Testing initialization...")
    parser = StreamingXMLToolCallParser()

    assert parser.call_id_counter == 0
    assert parser.tool_call_index == 0
    assert parser.current_call_id is None
    assert parser.deltas == []
    print("✓ Initialization test passed")


def test_param_type_conversion():
    """Test parameter type conversion"""
    print("\nTesting parameter type conversion...")
    parser = StreamingXMLToolCallParser()

    # String
    assert parser._convert_param_value("hello", "string") == "hello"

    # Integer
    assert parser._convert_param_value("42", "integer") == 42

    # Float
    assert parser._convert_param_value("3.14", "float") == 3.14

    # Boolean
    assert parser._convert_param_value("true", "boolean") is True
    assert parser._convert_param_value("false", "boolean") is False

    # Null
    assert parser._convert_param_value("null", "string") is None

    print("✓ Parameter type conversion tests passed")


def test_xml_escaping():
    """Test XML special character handling"""
    print("\nTesting XML escaping...")
    parser = StreamingXMLToolCallParser()

    # Escaping
    assert parser._escape_xml_special_chars("&") == "&amp;"
    assert parser._escape_xml_special_chars("<") == "&lt;"
    assert parser._escape_xml_special_chars(">") == "&gt;"

    # Unescaping
    assert parser._unescape_xml_special_chars("&amp;") == "&"
    assert parser._unescape_xml_special_chars("&lt;") == "<"
    assert parser._unescape_xml_special_chars("&gt;") == ">"

    print("✓ XML escaping tests passed")


def test_xml_preprocessing():
    """Test XML chunk preprocessing"""
    print("\nTesting XML preprocessing...")
    parser = StreamingXMLToolCallParser()

    # Function format conversion
    result = parser._preprocess_xml_chunk("<function=test_func>")
    assert result == '<function name="test_func">'

    # Parameter format conversion
    result = parser._preprocess_xml_chunk("<parameter=arg1>")
    assert result == '<parameter name="arg1">'

    print("✓ XML preprocessing tests passed")


def test_detector_initialization():
    """Test detector initialization"""
    print("\nTesting detector initialization...")
    detector = Qwen3CoderDetector()

    assert detector.tool_call_start_token == "<tool_call>"
    assert detector.tool_call_end_token == "</tool_call>"
    assert isinstance(detector.parser, StreamingXMLToolCallParser)

    print("✓ Detector initialization test passed")


def test_has_tool_call():
    """Test tool call detection"""
    print("\nTesting tool call detection...")
    detector = Qwen3CoderDetector()

    assert detector.has_tool_call("text <tool_call> more") is True
    assert detector.has_tool_call("just regular text") is False

    print("✓ Tool call detection tests passed")


def main():
    """Run all tests"""
    print("=" * 60)
    print("XML Parser PR1 Test Suite")
    print("=" * 60)

    try:
        test_initialization()
        test_param_type_conversion()
        test_xml_escaping()
        test_xml_preprocessing()
        test_detector_initialization()
        test_has_tool_call()

        print("\n" + "=" * 60)
        print("✓ All PR1 tests passed!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
