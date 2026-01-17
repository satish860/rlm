import os
from dotenv import load_dotenv
from openai import OpenAI
from segmenter import segment_document, split_into_pages

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

ROOT_MODEL = "anthropic/claude-opus-4"
SUB_MODEL = "openai/gpt-4o-mini"

def llm_query(prompt: str, model: str = SUB_MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

with open("recursive_language_models.converted.md", "r", encoding="utf-8") as f:
    context = f.read()

pages = split_into_pages(context, lines_per_page=50)

def get_section(start_page: int, end_page: int, padding: int = 1) -> str:
    """Get content for pages start_page to end_page (1-indexed) with optional padding."""
    start = max(0, start_page - 1 - padding)
    end = min(len(pages), end_page + padding)
    return '\n\n'.join(pages[start:end])

def ask_about_section(question: str, start_page: int, end_page: int) -> str:
    """Ask the sub-LLM a question about a specific section. Automatically includes the content."""
    content = get_section(start_page, end_page)
    prompt = f"""Answer this question about the following document section:

QUESTION: {question}

DOCUMENT SECTION (pages {start_page}-{end_page}):
{content}"""
    return llm_query(prompt)

print("Segmenting document...")
segments = segment_document(context, lines_per_page=50, chunk_size=10, max_workers=3)

toc_text = "\n".join([
    f"[{s['page_range']['start']}-{s['page_range']['end']}] {s['heading']}"
    for s in segments
])

SYSTEM_PROMPT = """You are tasked with answering a query about a document. You have access to a pre-indexed table of contents.

DOCUMENT STRUCTURE:
{toc}

The REPL environment has:
1. get_section(start_page, end_page, padding=1) - Get content for specific pages. Returns the text.
2. ask_about_section(question, start_page, end_page) - Ask sub-LLM about a section. AUTOMATICALLY includes the content.
3. llm_query(prompt) - Raw sub-LLM call. YOU must include any content in the prompt yourself.
4. print() - View output.

You have three tools:
1. execute_code(code) - Run Python code in the REPL.
2. final_answer(answer="your answer here") - Return answer directly as string.
3. final_answer_var(variable="result") - Return a variable you stored in REPL.

WORKFLOW:
1. Look at the DOCUMENT STRUCTURE above to find relevant sections.
2. Use get_section(start, end) to read content, or ask_about_section() for analysis.
3. Call final_answer or final_answer_var when done.

IMPORTANT: ask_about_section() is preferred over llm_query() because it automatically passes the content."""

def build_system_prompt():
    return SYSTEM_PROMPT.format(toc=toc_text)

def build_user_message(query: str):
    return f"Query: {query}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in the REPL environment. Variables persist across calls. Use print() to see output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Return your final answer as a string. Use this when you want to write the answer directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete final answer text"
                    }
                },
                "required": ["answer"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer_var",
            "description": "Return a variable from the REPL as your answer. Use this when you stored the answer in a variable like 'result' or 'answer'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variable": {
                        "type": "string",
                        "description": "Variable name (e.g. 'result', 'answer', 'references')"
                    }
                },
                "required": ["variable"]
            }
        }
    }
]

import json
import io
import sys

def execute_code_in_namespace(code: str, namespace: dict) -> str:
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(code, namespace)
        output = sys.stdout.getvalue()
    except Exception as e:
        output = f"Error: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout
    return output

def run_rlm(query: str, max_iterations: int = 20) -> str:
    namespace = {
        "context": context,
        "llm_query": llm_query,
        "get_section": get_section,
        "ask_about_section": ask_about_section,
    }

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_message(query)}
    ]

    for i in range(max_iterations):
        print(f"\n=== Iteration {i+1} ===")

        response = client.chat.completions.create(
            model=ROOT_MODEL,
            messages=messages,
            tools=TOOLS
        )

        msg = response.choices[0].message

        if msg.content:
            print(f"LLM: {msg.content}")

        if not msg.tool_calls:
            print("No tool calls, stopping.")
            return msg.content

        messages.append(msg)

        for tool_call in msg.tool_calls:
            name = tool_call.function.name

            try:
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            except json.JSONDecodeError as e:
                print(f"Error parsing tool arguments: {e}")
                print(f"Raw arguments: {tool_call.function.arguments}")
                print(f"Full message content: {msg.content}")
                result = f"Error: Invalid arguments - {e}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                continue

            print(f"Tool: {name}")
            print(f"Args: {args}")

            if name == "execute_code" and "code" not in args:
                print(f"WARNING: Empty code args. LLM said: {msg.content}")

            if name == "execute_code":
                if "code" not in args:
                    result = "Error: Missing 'code' argument"
                else:
                    result = execute_code_in_namespace(args["code"], namespace)
                print(f"Output: {result}")
            elif name == "final_answer":
                answer = args.get("answer", "No answer provided")
                print(f"FINAL ANSWER: {answer}")
                return answer
            elif name == "final_answer_var":
                var_name = args.get("variable", "")
                if not var_name:
                    result = "Error: Missing 'variable' argument"
                    print(result)
                else:
                    result = str(namespace.get(var_name, f"Variable '{var_name}' not found"))
                    print(f"FINAL ANSWER (from {var_name}): {result}")
                    return result

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result if name == "execute_code" else "OK"
            })

    return "Max iterations reached"

if __name__ == "__main__":
    answer = run_rlm("List all the references cited in this paper.")
    print(f"\n=== RESULT ===\n{answer}")
