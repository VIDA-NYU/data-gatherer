# this file is for checking whether the Prompt id from LLMs_responses_cache.json also appears as file name in the prompt_eval folder. If not, response will be deleted from LLMs_responses.json
import json
import os

def check_prompt_id_in_prompt_eval_folder():
    print("Checking if Prompt id in LLMs_responses_cache.json also appears as file name in the prompt_eval folder...")
    # Load the LLMs_responses.json file
    new_llms_responses = {}
    with open('LLMs_responses_cache.json', 'r') as f:
        llms_responses = json.load(f)

    # Get the list of prompt IDs from the prompt_eval folder
    prompt_eval_folder = 'prompt_evals'
    prompt_ids = [f.split('.json')[0] for f in os.listdir(prompt_eval_folder)]
    print(f"Prompt IDs found in prompt_eval folder: {prompt_ids}")

    # Check if each prompt ID in LLMs_responses.json exists in the prompt_eval folder
    conter = 0
    for response in llms_responses:
        if response in prompt_ids:
            new_llms_responses[response] = llms_responses[response]
        else:
            conter += 1
            print(f"Prompt ID {response} not found in prompt_eval folder. Deleting response.")
    print(f"Total responses deleted: {conter}")

    # Save the cleaned LLMs_responses.json file
    with open('LLMs_responses_cache.json', 'w') as f:
        json.dump(new_llms_responses, f, indent=4)

check_prompt_id_in_prompt_eval_folder()
