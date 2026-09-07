from champion_runtime.agent_loop import _extract_tool_call, _extract_between

def test_extract_tool_call_formats():
    assert _extract_tool_call('<tool_call>{"name":"search","arguments":{"query":["a"]}}</tool_call>').startswith('{"name"')
    assert '"name": "search"' in _extract_tool_call('<function=search><parameter=query>["a"]</parameter></function>')
    assert _extract_tool_call('x {"name":"search","arguments":{"query":["a"]}} y').startswith('{"name"')

def test_invalid_and_answer_semantics():
    assert _extract_tool_call('<tool_call name="search">x</tool_call>') is None
    assert _extract_between('<answer>first</answer><answer>second</answer>', '<answer>', '</answer>') == 'first'
