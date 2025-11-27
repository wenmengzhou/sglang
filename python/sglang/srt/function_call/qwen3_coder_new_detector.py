import ast
import html
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from xml.parsers.expat import ParserCreate

from sglang.srt.entrypoints.openai.protocol import (
    DeltaMessage,
    FunctionResponse,
    Tool,
    ToolCall,
)
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer

logger = logging.getLogger(__name__)


class StreamingXMLToolCallParser:
    """
    Core streaming parser responsible for handling XML format tool calls


    ## Main state variables:
    - current_call_id: unique identifier for the current tool call
    - current_function_name: current function name
    - parameters: stores parsed parameters
    - current_param_name: current parameter name
    - current_param_value: current parameter value
    - tool_call_index: tool call index counter
    - streaming_buffer: streaming processing buffer
    - text_content_buffer: text content buffer

    ## Processing flow:
    -. State initialization: set initial state variables and XML parser
    -. Streaming input processing: receive data chunks through parse_single_streaming_chunks
    -. XML element identification: use _find_next_complete_element to identify complete XML elements
    -. XML parsing: use expat parser to process XML elements, triggering _start_element, _char_data, _end_element callbacks
    -. State transition: update state variables based on XML element types
    -. Delta generation: generate DeltaMessage at appropriate times and send

    ## State transition process:
    -. Start parsing: <tool_call> tag resets tool call state
    -. Function identification: <function> tag extracts function name and generates initial tool call Delta
    -. Parameter processing: <parameter> tag starts parameter parsing, _char_data processes parameter values
    -. Parameter end: decide whether to add quotes based on parameter type, store converted value
    -. Function end: close JSON object, output complete function call
    -. Tool call end: </tool_call> tag ends current tool call, reset parser state

     Special handling:
    - XML special character escaping and unescaping
    - Parameter type conversion (string, number, boolean, etc.)
    - Streaming output delay processing to ensure correct JSON format
    - Multiple tool_call handling and state isolation

    """

    def __init__(self):
        self.call_id_counter = 0
        self.tool_call_index = 0
        self.current_call_id = None
        self.last_completed_call_id = None  # Save the most recently completed call_id
        self.current_function_name = None
        self.current_function_open = False
        self.parameters = {}
        self.current_param_name = None
        self.current_param_value = ""
        self.current_param_value_converted = (
            ""  # Record parameter value after type conversion
        )
        self.current_param_is_first = (
            False  # Record whether this is the first parameter
        )
        # Need to delay output here because parameter end will contain an extra newline that needs to be removed, but intermediate newlines need to be preserved
        self.should_emit_end_newline = False  # Record whether to output ending newline
        self.start_quote_emitted = False  # Record whether the starting quote for string parameter has been output

        self.deltas = []

        # Single chunk streaming processing state
        self.streaming_buffer = ""
        self.last_processed_pos = 0

        # Used to collect text content before tool_call
        self.text_content_buffer = ""

        # XML parser related
        self.parser = ParserCreate()
        self.setup_parser()

        # Tool configuration information
        self.tools = []

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function"
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter"
        self.parameter_end_token: str = "</parameter>"

    def set_tools(self, tools: List[Tool]):
        """Set tool configuration information"""
        self.tools = tools

    def _get_param_type(self, param_name: str) -> str:
        """Get parameter type based on tool configuration, default to string"""
        if not self.tools or not self.current_function_name:
            return "string"

        for tool in self.tools:
            if not hasattr(tool, "type") or not (
                hasattr(tool, "function") and hasattr(tool.function, "name")
            ):
                continue
            if (
                tool.type == "function"
                and tool.function.name == self.current_function_name
            ):
                if not hasattr(tool.function, "parameters"):
                    return "string"
                params = tool.function.parameters
                if isinstance(params, dict) and "properties" in params:
                    properties = params["properties"]
                    if param_name in properties and isinstance(
                        properties[param_name], dict
                    ):
                        return str(properties[param_name].get("type", "string"))
                elif isinstance(params, dict) and param_name in params:
                    param_config = params[param_name]
                    if isinstance(param_config, dict):
                        return str(param_config.get("type", "string"))
                break
        return "string"

    def _convert_param_value(self, param_value: str, param_type: str) -> Any:
        """Convert value based on parameter type"""
        # Special case: for example, model outputs True/False for bool value, need json.dumps to convert to true/false
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        STRING_TYPE_GROUP = ["string", "str", "text", "varchar", "char", "enum"]
        BOOL_TYPE_GROUP = ["boolean", "bool", "binary"]
        ARRAY_TYPE_GROUP = ["array"]
        param_type = param_type.strip().lower()
        if param_type in STRING_TYPE_GROUP:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                param_value = int(param_value)
            except Exception as e:
                logger.warning(f"Error during fallback completion: {e}")
            return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                param_value = (
                    float_param_value
                    if float_param_value - int(float_param_value) != 0
                    else int(float_param_value)
                )
            except Exception as e:
                logger.warning(f"Error during fallback completion: {e}")

            return param_value
        elif param_type in BOOL_TYPE_GROUP:
            param_value = param_value.lower()
            return param_value == "true"
        elif param_type in ARRAY_TYPE_GROUP:
            try:
                # First try ast.literal_eval for safe evaluation of Python literals
                param_value = ast.literal_eval(param_value)
                if not isinstance(param_value, list):
                    param_value = list(param_value)
            except (ValueError, SyntaxError):
                # If literal_eval fails, try json.loads for JSON array format
                try:
                    param_value = json.loads(param_value)
                    if not isinstance(param_value, list):
                        param_value = list(param_value)
                except (json.JSONDecodeError, TypeError) as e:
                    # If both parsing methods fail, keep as string
                    logger.warning(f"Error during fallback completion: {e}")
            return param_value
        else:
            return param_value

    def _convert_for_json_streaming(self, converted_value: Any, param_type: str) -> str:
        """Convert convert_value based on whether converted_value is empty and whether type is string

        Args:
            converted_value: converted value
            param_type: parameter type

        Returns:
            converted string for streaming output
        """
        # Check if it's an empty value, but exclude the number 0
        if converted_value is None or converted_value == "":
            return ""

        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            # String type, remove double quotes
            return json.dumps(converted_value, ensure_ascii=False)[1:-1]
        else:
            # Non-string type, return complete JSON string
            if not isinstance(converted_value, str):
                return json.dumps(converted_value, ensure_ascii=False)
            else:
                return converted_value

    def reset_streaming_state(self):
        """Reset streaming parsing state"""
        self.call_id_counter = 0
        self.tool_call_index = 0
        self.current_call_id = None
        self.last_completed_call_id = None
        self.current_function_name = None
        self.current_function_open = False
        self.parameters = {}
        self.current_param_name = None
        self.current_param_value = ""
        self.current_param_value_converted = ""
        self.current_param_is_first = False
        self.should_emit_end_newline = False
        self.start_quote_emitted = False

        # Reset single chunk streaming processing state
        self.streaming_buffer = ""
        self.last_processed_pos = 0

        # Reset text content buffer
        self.text_content_buffer = ""

        self.deltas = []

        # Recreate parser
        self.parser = ParserCreate()
        self.setup_parser()

    def _escape_xml_special_chars(self, text: str) -> str:
        """
        Escape XML special characters

        Args:
            text: original text

        Returns:
            escaped text
        """
        # XML special character escape mapping
        XML_ESCAPES_MAP = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&apos;",
        }

        for char, escape in XML_ESCAPES_MAP.items():
            text = text.replace(char, escape)

        return text

    def _unescape_xml_special_chars(self, text: str) -> str:
        """
        Unescape XML special characters

        Args:
            text: escaped text

        Returns:
            original text
        """
        # XML special character unescape mapping
        xml_unescapes = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&apos;": "'",
        }

        for escape, char in xml_unescapes.items():
            text = text.replace(escape, char)

        return text

    def _preprocess_xml_chunk(self, chunk: str) -> str:
        """
        Preprocess XML chunk, handle non-standard format

        Args:
            chunk: original XML chunk

        Returns:
            processed XML chunk
        """
        is_tool_call = False
        if chunk.startswith("<tool_call>") or chunk.startswith("</tool_call>"):
            is_tool_call = True
        if chunk.startswith("<function=") or chunk.startswith("</function>"):
            is_tool_call = True
        if chunk.startswith("<parameter=") or chunk.startswith("</parameter>"):
            is_tool_call = True
        # Handle <function=name> format -> <function name="name">
        processed = re.sub(r"<function=([^>]+)>", r'<function name="\1">', chunk)
        # Handle <parameter=name> format -> <parameter name="name">
        processed = re.sub(r"<parameter=([^>]+)>", r'<parameter name="\1">', processed)
        # If processed doesn't contain special_token, escape processed
        # This is because XML parsing will error on special characters, so escaping is needed
        if not is_tool_call:
            processed = self._escape_xml_special_chars(processed)
        return processed

    def _should_skip_element(self, element: str) -> bool:
        """
        Determine whether to skip a certain element

        Args:
            element: element to judge

        Returns:
            bool: True means should skip, False means should process
        """
        # element = element.strip()

        # If it's tool_call XML tag, don't skip
        if element.startswith("<tool_call>"):
            return False

        # If currently not parsing tool call and not empty, collect this text instead of skipping
        # Only process other XML elements when tool_call appears, otherwise treat as plain text
        if self.current_call_id is None and element:
            # Collect text content to buffer
            self.text_content_buffer += element
            return True  # Still skip, but content has been collected

        # If currently parsing tool call, this might be parameter value, don't skip
        if self.current_call_id is not None:
            return False

        # Skip empty content
        if not element:
            return True

        return False

    def setup_parser(self):
        """Set up XML parser event handlers"""
        self.parser.buffer_text = True
        self.parser.StartElementHandler = self._start_element
        self.parser.EndElementHandler = self._end_element
        self.parser.CharacterDataHandler = self._char_data

    def _get_next_call_id(self):
        """Generate unique call ID"""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _extract_function_name(self, name: str, attrs: Dict[str, str]) -> Optional[str]:
        """Extract function name from various formats"""
        if attrs and "name" in attrs:
            return attrs["name"]

        # Handle <function=name> format
        if "=" in name:
            parts = name.split("=", 1)
            if len(parts) == 2 and parts[0] == "function":
                return parts[1]

        return None

    def _extract_parameter_name(
        self, name: str, attrs: Dict[str, str]
    ) -> Optional[str]:
        """Extract parameter name from various formats"""
        if attrs and "name" in attrs:
            return attrs["name"]

        # Handle <parameter=name> format
        if "=" in name:
            parts = name.split("=", 1)
            if len(parts) == 2 and parts[0] == "parameter":
                return parts[1]

        return None

    def _emit_delta(self, delta: DeltaMessage):
        """Emit Delta response (streaming output)"""
        self.deltas.append(delta)

    # ========== Placeholder methods for PR2 and PR3 ==========
    # These will be implemented in subsequent PRs

    def parse_single_streaming_chunks(self, xml_chunk: str) -> DeltaMessage:
        """
        Parse single streaming XML chunk and return Delta response

        NOTE: Full implementation will be added in PR3
        """
        raise NotImplementedError("Will be implemented in PR3")

    def _process_complete_xml_elements(self) -> bool:
        """
        Process complete XML elements in the buffer

        NOTE: Full implementation will be added in PR2
        """
        raise NotImplementedError("Will be implemented in PR2")

    def _find_next_complete_element(self, start_pos: int) -> Tuple[Optional[str], int]:
        """
        Find the next complete XML element from specified position

        NOTE: Full implementation will be added in PR2
        """
        raise NotImplementedError("Will be implemented in PR2")

    def _merge_new_deltas_to_single_response(self, initial_count: int) -> DeltaMessage:
        """
        Merge newly generated deltas in this processing into single DeltaMessage

        NOTE: Full implementation will be added in PR2
        """
        raise NotImplementedError("Will be implemented in PR2")

    def _start_element(self, name: str, attrs: Dict[str, str]):
        """
        Handle XML start element event

        NOTE: Full implementation will be added in PR3
        """
        raise NotImplementedError("Will be implemented in PR3")

    def _char_data(self, data: str):
        """
        Handle XML character data event

        NOTE: Full implementation will be added in PR3
        """
        raise NotImplementedError("Will be implemented in PR3")

    def _end_element(self, name: str):
        """
        Handle XML end element event

        NOTE: Full implementation will be added in PR3
        """
        raise NotImplementedError("Will be implemented in PR3")


class Qwen3CoderDetector(BaseFormatDetector):
    """
    Detector for Qwen 3 models.
    Assumes function call format:
        <tool_call>
        <function=execute_bash>
        <parameter=command>
        pwd && ls
        </parameter>
        </function>
        </tool_call>

    NOTE: Full implementation will be added in PR3
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self._buf: str = ""

        # for non-stream extract
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )

        self.parser = StreamingXMLToolCallParser()

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """NOTE: Full implementation will be added in PR3"""
        raise NotImplementedError("Will be implemented in PR3")

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """NOTE: Full implementation will be added in PR3"""
        raise NotImplementedError("Will be implemented in PR3")

    def _reset_streaming_state(self):
        """Reset streaming state for the next tool call"""
        self._buf = ""
        self.parser.reset_streaming_state()

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            individual_call_start_token=self.tool_call_start_token.replace("\n", "\\n"),
            individual_call_end_token=self.tool_call_end_token.replace("\n", "\\n"),
            tool_call_separator="\\n",
            function_format="xml",
            call_rule_fmt='"<function={name}>\\n" {arguments_rule} "\\n</function>"',
            key_value_rule_fmt='"<parameter={key}>\\n" {valrule} "\\n</parameter>"',
            key_value_separator="\\n",
        )
