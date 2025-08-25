from ollama import chat
import ollama
import subprocess


class OllamaAgent:
    def __init__(self, model='llama2:7b', system_prompt='You are a helpful assistant.'):
        self.model = model
        self.system_prompt = system_prompt
        self.history = [{"role": "system", "content": self.system_prompt}]

    def generate_one_answer(self, input: str) -> str:
        resp = ollama.generate(model=self.model, prompt=input)
        return resp['response']

    def get_llm_response(self, user_prompt) -> str:
        if isinstance(user_prompt, list):
            for prompt in user_prompt:
                self.history.append({"role": "user", "content": prompt})
        else:
            self.history.append({"role": "user", "content": user_prompt})
            
        print(self.history)
        chat_response = chat(model=self.model, messages=self.history)
        response_content = chat_response['message']['content']
        self.history.append({"role": "assistant", "content": response_content})
        return response_content

    def pull_model(self, model_name):
        try:
            result = subprocess.run(
                ["docker", "exec", "ollama", "ollama", "pull", model_name],
                check=True,
                capture_output=True,
                text=True
            )
            print("Output:\n", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error:\n", e.stderr)

    def is_model_available(self, model_name):
        try:
            response = requests.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(m["name"] == model_name for m in models)
        except requests.RequestException as e:
            print("Error querying Ollama:", e)
            return False

        
        


def generate_one_answer(input: str, model='llama2:7b') -> str:
    resp = ollama.generate(model=model, prompt=input)
    return resp['response']

def system_and_user(user_prompt, model='llama2:7b', system_prompt='') -> str:
    if system_prompt == '':
        system_prompt = "You are a helpful assistant. Return the answers in JSON format."        
        
    messages = [{"role": "system", "content": system_prompt}]

    if isinstance(user_prompt, list):
        for prompt in user_prompt:
            messages.append({"role": "user", "content": prompt})
            
    else:
        messages.append({"role": "user", "content": user_prompt})
    
    chat_response = chat(model=model, 
                        messages=messages)

    return chat_response['message']['content'] 
    
def pull_ollama_model(model_name):
    try:
        result = subprocess.run(
            ["docker", "exec", "ollama", "ollama", "pull", model_name],
            check=True,
            capture_output=True,
            text=True
        )
        print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:\n", e.stderr)


import requests

def is_model_available(model_name):
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return any(m["name"] == model_name for m in models)
    except requests.RequestException as e:
        print("Error querying Ollama:", e)
        return False

 
if __name__ == "__main__":
    # input = "What is the capital of France?"
    # result = system_and_user(input)
    # print(result)
    
        
    chat_response = chat(model='llama3.3:70b', 
                         messages=[{"role": "system", "content": 'You are a helpful assistant.'},
                                   {"role": "user", "content": ' I nead some help with a question.'}])
    
    print(chat_response)