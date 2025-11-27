"""
Test cases for XML Parser PR1: Base Framework and Utility Methods

This test file validates the functionality implemented in PR1:
- Class initialization and state management
- Parameter type detection and conversion
- XML special character handling
- Basic utility methods
"""

import pytest
from sglang.srt.entrypoints.openai.protocol import Tool, ToolFunction, ToolFunctionParameters
from sglang.srt.function_call.qwen3_coder_new_detector import (
    StreamingXMLToolCallParser,
    Qwen3CoderDetector,
)


class TestStreamingXMLToolCallParser:
    """Test StreamingXMLToolCallParser base framework"""

    def test_initialization(self):
        """Test parser initializes with correct default state"""
        parser = StreamingXMLToolCallParser()

        assert parser.call_id_counter == 0
        assert parser.tool_call_index == 0
        assert parser.current_call_id is None
        assert parser.current_function_name is None
        assert parser.parameters == {}
        assert parser.deltas == []
        assert parser.streaming_buffer == ""
        assert parser.text_content_buffer == ""
        assert parser.tool_call_start_token == "<tool_call>"
        assert parser.tool_call_end_token == "</tool_call>"

    def test_set_tools(self):
        """Test setting tools configuration"""
        parser = StreamingXMLToolCallParser()

        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="test_func",
                    description="Test function",
                    parameters=ToolFunctionParameters(
                        type="object",
                        properties={
                            "arg1": {"type": "string"},
                            "arg2": {"type": "integer"}
                        }
                    )
                )
            )
        ]

        parser.set_tools(tools)
        assert parser.tools == tools
        assert len(parser.tools) == 1

    def test_get_param_type_with_tools(self):
        """Test parameter type detection with tool configuration"""
        parser = StreamingXMLToolCallParser()
        parser.current_function_name = "test_func"

        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="test_func",
                    parameters=ToolFunctionParameters(
                        type="object",
                        properties={
                            "str_param": {"type": "string"},
                            "int_param": {"type": "integer"},
                            "bool_param": {"type": "boolean"},
                            "array_param": {"type": "array"}
                        }
                    )
                )
            )
        ]
        parser.set_tools(tools)

        assert parser._get_param_type("str_param") == "string"
        assert parser._get_param_type("int_param") == "integer"
        assert parser._get_param_type("bool_param") == "boolean"
        assert parser._get_param_type("array_param") == "array"
        assert parser._get_param_type("unknown_param") == "string"  # default

    def test_get_param_type_without_tools(self):
        """Test parameter type defaults to string without tools"""
        parser = StreamingXMLToolCallParser()
        assert parser._get_param_type("any_param") == "string"

    def test_convert_param_value_string(self):
        """Test string parameter conversion"""
        parser = StreamingXMLToolCallParser()

        assert parser._convert_param_value("hello", "string") == "hello"
        assert parser._convert_param_value("world", "str") == "world"
        assert parser._convert_param_value("test", "text") == "test"

    def test_convert_param_value_integer(self):
        """Test integer parameter conversion"""
        parser = StreamingXMLToolCallParser()

        assert parser._convert_param_value("42", "integer") == 42
        assert parser._convert_param_value("100", "int") == 100
        assert parser._convert_param_value("-5", "long") == -5

    def test_convert_param_value_float(self):
        """Test float parameter conversion"""
        parser = StreamingXMLToolCallParser()

        assert parser._convert_param_value("3.14", "float") == 3.14
        assert parser._convert_param_value("2.5", "number") == 2.5
        assert parser._convert_param_value("10.0", "float") == 10  # Should be int if no decimal

    def test_convert_param_value_boolean(self):
        """Test boolean parameter conversion"""
        parser = StreamingXMLToolCallParser()

        assert parser._convert_param_value("true", "boolean") is True
        assert parser._convert_param_value("True", "bool") is True
        assert parser._convert_param_value("false", "boolean") is False
        assert parser._convert_param_value("False", "bool") is False

    def test_convert_param_value_array(self):
        """Test array parameter conversion"""
        parser = StreamingXMLToolCallParser()

        result = parser._convert_param_value("[1, 2, 3]", "array")
        assert result == [1, 2, 3]

        result = parser._convert_param_value('["a", "b"]', "array")
        assert result == ["a", "b"]

    def test_convert_param_value_null(self):
        """Test null value conversion"""
        parser = StreamingXMLToolCallParser()

        assert parser._convert_param_value("null", "string") is None
        assert parser._convert_param_value("NULL", "integer") is None
        assert parser._convert_param_value("Null", "boolean") is None

    def test_convert_for_json_streaming_string(self):
        """Test JSON streaming conversion for strings"""
        parser = StreamingXMLToolCallParser()

        # String type should remove quotes
        result = parser._convert_for_json_streaming("hello", "string")
        assert result == "hello"

        # Empty string
        result = parser._convert_for_json_streaming("", "string")
        assert result == ""

    def test_convert_for_json_streaming_number(self):
        """Test JSON streaming conversion for numbers"""
        parser = StreamingXMLToolCallParser()

        result = parser._convert_for_json_streaming(42, "integer")
        assert result == "42"

        result = parser._convert_for_json_streaming(3.14, "float")
        assert result == "3.14"

    def test_convert_for_json_streaming_boolean(self):
        """Test JSON streaming conversion for booleans"""
        parser = StreamingXMLToolCallParser()

        result = parser._convert_for_json_streaming(True, "boolean")
        assert result == "true"

        result = parser._convert_for_json_streaming(False, "boolean")
        assert result == "false"

    def test_convert_for_json_streaming_array(self):
        """Test JSON streaming conversion for arrays"""
        parser = StreamingXMLToolCallParser()

        result = parser._convert_for_json_streaming([1, 2, 3], "array")
        assert result == "[1, 2, 3]"

    def test_escape_xml_special_chars(self):
        """Test XML special character escaping"""
        parser = StreamingXMLToolCallParser()

        assert parser._escape_xml_special_chars("&") == "&amp;"
        assert parser._escape_xml_special_chars("<") == "&lt;"
        assert parser._escape_xml_special_chars(">") == "&gt;"
        assert parser._escape_xml_special_chars('"') == "&quot;"
        assert parser._escape_xml_special_chars("'") == "&apos;"

        # Test combined
        text = 'Hello & "world" <test>'
        expected = 'Hello &amp; &quot;world&quot; &lt;test&gt;'
        assert parser._escape_xml_special_chars(text) == expected

    def test_unescape_xml_special_chars(self):
        """Test XML special character unescaping"""
        parser = StreamingXMLToolCallParser()

        assert parser._unescape_xml_special_chars("&amp;") == "&"
        assert parser._unescape_xml_special_chars("&lt;") == "<"
        assert parser._unescape_xml_special_chars("&gt;") == ">"
        assert parser._unescape_xml_special_chars("&quot;") == '"'
        assert parser._unescape_xml_special_chars("&apos;") == "'"

        # Test combined
        text = 'Hello &amp; &quot;world&quot; &lt;test&gt;'
        expected = 'Hello & "world" <test>'
        assert parser._unescape_xml_special_chars(text) == expected

    def test_preprocess_xml_chunk_function_format(self):
        """Test preprocessing of function XML format"""
        parser = StreamingXMLToolCallParser()

        # Test <function=name> format
        chunk = "<function=test_func>"
        result = parser._preprocess_xml_chunk(chunk)
        assert result == '<function name="test_func">'

        # Test <parameter=name> format
        chunk = "<parameter=arg1>"
        result = parser._preprocess_xml_chunk(chunk)
        assert result == '<parameter name="arg1">'

    def test_preprocess_xml_chunk_text_escaping(self):
        """Test that non-tool-call text gets escaped"""
        parser = StreamingXMLToolCallParser()

        # Regular text should be escaped
        chunk = "Hello & <world>"
        result = parser._preprocess_xml_chunk(chunk)
        assert result == "Hello &amp; &lt;world&gt;"

        # Tool call tags should not be escaped
        chunk = "<tool_call>"
        result = parser._preprocess_xml_chunk(chunk)
        assert result == "<tool_call>"

    def test_should_skip_element_tool_call(self):
        """Test that tool_call elements are not skipped"""
        parser = StreamingXMLToolCallParser()

        assert parser._should_skip_element("<tool_call>") is False

    def test_should_skip_element_text_before_tool_call(self):
        """Test that text before tool_call is collected"""
        parser = StreamingXMLToolCallParser()
        parser.current_call_id = None  # Not in a tool call

        # Text should be collected and element should be skipped
        result = parser._should_skip_element("some text")
        assert result is True
        assert parser.text_content_buffer == "some text"

    def test_should_skip_element_during_tool_call(self):
        """Test that content during tool_call is not skipped"""
        parser = StreamingXMLToolCallParser()
        parser.current_call_id = "call_123"  # In a tool call

        # Should not skip during tool call
        assert parser._should_skip_element("param value") is False

    def test_get_next_call_id(self):
        """Test call ID generation"""
        parser = StreamingXMLToolCallParser()

        call_id1 = parser._get_next_call_id()
        call_id2 = parser._get_next_call_id()

        # Should generate unique IDs
        assert call_id1 != call_id2
        assert call_id1.startswith("call_")
        assert call_id2.startswith("call_")
        assert len(call_id1) == 29  # "call_" + 24 hex chars

    def test_extract_function_name_from_attrs(self):
        """Test function name extraction from attributes"""
        parser = StreamingXMLToolCallParser()

        result = parser._extract_function_name("function", {"name": "test_func"})
        assert result == "test_func"

    def test_extract_function_name_from_tag(self):
        """Test function name extraction from tag format"""
        parser = StreamingXMLToolCallParser()

        result = parser._extract_function_name("function=test_func", {})
        assert result == "test_func"

    def test_extract_parameter_name_from_attrs(self):
        """Test parameter name extraction from attributes"""
        parser = StreamingXMLToolCallParser()

        result = parser._extract_parameter_name("parameter", {"name": "arg1"})
        assert result == "arg1"

    def test_extract_parameter_name_from_tag(self):
        """Test parameter name extraction from tag format"""
        parser = StreamingXMLToolCallParser()

        result = parser._extract_parameter_name("parameter=arg1", {})
        assert result == "arg1"

    def test_reset_streaming_state(self):
        """Test that reset clears all state"""
        parser = StreamingXMLToolCallParser()

        # Set some state
        parser.tool_call_index = 5
        parser.current_call_id = "call_123"
        parser.current_function_name = "test"
        parser.parameters = {"key": "value"}
        parser.streaming_buffer = "buffer"
        parser.deltas = ["delta1", "delta2"]

        # Reset
        parser.reset_streaming_state()

        # Verify all state is cleared
        assert parser.tool_call_index == 0
        assert parser.current_call_id is None
        assert parser.current_function_name is None
        assert parser.parameters == {}
        assert parser.streaming_buffer == ""
        assert parser.deltas == []


class TestQwen3CoderDetector:
    """Test Qwen3CoderDetector basic functionality"""

    def test_initialization(self):
        """Test detector initializes correctly"""
        detector = Qwen3CoderDetector()

        assert detector.tool_call_start_token == "<tool_call>"
        assert detector.tool_call_end_token == "</tool_call>"
        assert detector._buf == ""
        assert detector.parser is not None
        assert isinstance(detector.parser, StreamingXMLToolCallParser)

    def test_has_tool_call_positive(self):
        """Test detection of tool call in text"""
        detector = Qwen3CoderDetector()

        text = "Some text <tool_call> more text"
        assert detector.has_tool_call(text) is True

    def test_has_tool_call_negative(self):
        """Test no tool call detected"""
        detector = Qwen3CoderDetector()

        text = "Just regular text without tool calls"
        assert detector.has_tool_call(text) is False

    def test_reset_streaming_state(self):
        """Test detector state reset"""
        detector = Qwen3CoderDetector()

        detector._buf = "some buffer"
        detector.parser.tool_call_index = 5

        detector._reset_streaming_state()

        assert detector._buf == ""
        assert detector.parser.tool_call_index == 0

    def test_supports_structural_tag(self):
        """Test structural tag support"""
        detector = Qwen3CoderDetector()
        assert detector.supports_structural_tag() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
