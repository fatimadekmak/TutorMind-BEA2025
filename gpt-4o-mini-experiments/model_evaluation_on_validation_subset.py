import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import classification_report
import time
from openai import OpenAI

api_key = "your_openai_api_key_here"  # Replace with your OpenAI API key
client = OpenAI(api_key=api_key)

# Load dataset
dataset = load_dataset("json", data_files="data\Mistake_identification_validation_split_chat_format.jsonl")

# Convert to Pandas DataFrame
df = dataset["train"].to_pandas()

# Extract output from messages (assuming it's the assistant's content in the last message)
df['output'] = df['messages'].apply(lambda x: [msg['content'] for msg in x if msg['role'] == 'assistant'][0])

test_df = df

test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# Print class distributions to verify stratification
print("\nTest Class Distribution:\n", test_dataset.to_pandas()["output"].value_counts())

start = time.time()
# Initialize counters and storage
results = []
total_responses, failed_responses = 0, 0
ground_truths = []
predictions = []

# Run inference
for convo in test_dataset:
    total_responses += 1
    if total_responses % 10 == 0:
        print(f"Processing request number {total_responses}")

    try:
        # Get ground truth (assistant's content in the last message)
        ground_truth = [msg['content'] for msg in convo['messages'] if msg['role'] == 'assistant'][0]
        ground_truths.append(ground_truth.lower())
        
        # Get conversation history (all messages except assistant's last message)
        conversation_history = [msg for msg in convo['messages'] if not (msg['role'] == 'assistant' and msg['content'] == ground_truth)]
        
        # Call GPT-4o
        response = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:tutormind:augmented-training-data-tutor-eval",
            messages=conversation_history,
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Extract model output
        model_output = response.choices[0].message.content.strip()

        # Extract prediction (everything after "Evaluation:")
        extracted_prediction = model_output.replace("Evaluation:", "").strip().lower()
        predictions.append(extracted_prediction)

        # Compare extracted prediction with ground truth (case insensitive)
        match = extracted_prediction == ground_truth.lower()

    except Exception as e:
        failed_responses += 1
        print(f"Failed request {total_responses} - Error: {e}")
        model_output, extracted_prediction, match = "ERROR", "ERROR", False
        predictions.append("ERROR")
    
    # Store results
    results.append({
        "Input": conversation_history,
        "Model Output": model_output,
        "Extracted Prediction": extracted_prediction,
        "Ground Truth": ground_truth,
        "Match": match
    })

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Save DataFrame to CSV
df_results.to_csv(r"openai\finetuned\augmented_gpt-4o_results.csv", index=False)

# Compute accuracy
accuracy = df_results["Match"].mean()
print(f"\nAccuracy: {accuracy:.2%}")

# Compute F1 score and other metrics
print("\nClassification Report:")
print(classification_report(ground_truths, predictions, target_names=['No', 'To some extent', 'Yes'], zero_division=0, digits=4))

print(f"\nTotal Requests: {total_responses}, Failed Requests: {failed_responses}, done in {time.time() - start:.2f} seconds")
