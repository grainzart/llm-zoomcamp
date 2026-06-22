from rag_helper import RAGBase
from openai import OpenAI
from dotenv import load_dotenv


openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

load_dotenv()


messages = [{"role": "user", "content": "I just discovered the course. Can I join it?"}]


response.output_text
from sqlitesearch import TextSearchIndex

sqlite_index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
    db_path="faq.db",
)


def search(query):
    boost_dict = {"question": 3.0, "section": 0.5}
    filter_dict = {"course": "llm-zoomcamp"}

    return sqlite_index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
    )


search("I just discovered the course. Can I join it?")

search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query text to look up in the course FAQ."}},
        "required": ["query"],
        "additionalProperties": False,
    },
}

response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=[search_tool],
)

response.output
import json, pprint

call = response.output[0]
args = json.loads(call.arguments)

results = search(**args)
result_json = json.dumps(results, indent=2)
pprint.pprint(result_json)

messages.extend(response.output)

messages.append({
    "type": "function_call_output",
    "call_id": call.call_id,
    "output": result_json,
})

response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=[search_tool],
)

response.output_text

usage = response.usage
usage.input_tokens, usage.output_tokens


def calculate_gpt54mini_price(input_tokens, output_tokens):
    INPUT_PRICE_PER_MILLION = 0.15
    OUTPUT_PRICE_PER_MILLION = 0.60

    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


result = calculate_gpt54mini_price(652, 33)
print("Total cost: $", round(result["total_cost"], 8))
