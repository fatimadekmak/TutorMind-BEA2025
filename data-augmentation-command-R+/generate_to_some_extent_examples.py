import json
import cohere
from datasets import load_dataset

co = cohere.ClientV2("YOUR_COHERE_API_KEY")  # Replace with your actual Cohere API key

# load json file using Datasets
# load a random sample of 100 conversations from the dev set 
dataset = load_dataset("json", data_files="~\mrbench_v3_devset_sample.json", split="train")
# Output list to store the generated examples
results = []

# Loop through the dataset
for i, obj in enumerate(dataset):
    if i % 10 == 0:
        print(f"Processing item {i} of {len(dataset)}")

    convo_id = obj["conversation_id"]
    convo = obj["conversation_history"]

    try:
        # Generate response using your LLM client (e.g., Cohere)
        response = co.chat(
            model="command-r-plus-04-2024",
            temperature=0.9,
            messages=[{"role": "system", "content": "You are a math tutor giving feedback to a student. Based on the conversation, write a single-sentence response that gently suggests the student may have made a mistake, but without clearly identifying what the mistake is. Your tone should sound uncertain, cautious, or exploratory. Do not explicitly say what is wrong. Do not state that something is definitely incorrect. Keep your response to ONE short sentence."},
                    {"role": "user", "content": convo},],)
        # Extract assistant text
        message_content = response.message.content[0].text.strip()

        # Store result
        results.append({
            "conversation_id": convo_id,
            "generated_response": message_content
        })

    except Exception as e:
        print(f"Error at index {i}, ID {convo_id}: {e}")
        continue
    # break
# Save to JSON file
output_path = "generated_tse_responses.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Done. Saved {len(results)} responses to {output_path}.")