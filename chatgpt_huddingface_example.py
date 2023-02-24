from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# check https://huggingface.co/models?sort=downloads
# for more pre-trained models
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")


def ask_chatgpt(prompt):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    # generated a response while limiting the total chat history to 1000 tokens,
    # https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id,
                                      do_sample=True,
                                      top_k=300,
                                      top_p=0.7, temperature=0.7)

    # pretty print last ouput tokens from bot
    return tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)


if __name__ == "__main__":
    prompt = "Lets play a dungeon game in a medieval theme, keep it short 5-6 lines tops, you start "
    response = ask_chatgpt(prompt)
    print(response)
