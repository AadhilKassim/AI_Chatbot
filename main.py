import json
import requests
import logging
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")
SYLLABUS_FILE = 'syllabus.json'

def clear_terminal():
    """Clears the terminal screen."""

    if os.name == 'nt':  # For Windows
        os.system('cls')

def load_syllabus_data():
    """Load syllabus data from JSON file"""
    try:
        with open(SYLLABUS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File {SYLLABUS_FILE} not found.")
        return {}
    except json.JSONDecodeError as json_err:
        logging.error(f"JSON decode error for file {SYLLABUS_FILE}: {json_err}")
        return {}

def create_request_data(prompt, syllabus_data):
    """Create the data for the request"""
    data = {
        "model": MODEL_NAME,  # Specify the model you are using
        "prompt": prompt      # Send the user's input as a prompt
    }
    # Add syllabus data to the request if available
    if syllabus_data:
        data["syllabus"] = syllabus_data
    return data

def make_request(url, data, session):
    """Make a POST request to the Ollama server with retries and handle streaming responses."""
    try:
        response = session.post(url, json=data, timeout=10, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Initialize an empty string to accumulate the response
        full_response = ""
        
        # Iterate over each line in the response
        for line in response.iter_lines():
            if line:
                try:
                    # Decode the line (bytes to string)
                    decoded_line = line.decode('utf-8')
                    logging.debug(f"Decoded line: {decoded_line}")
                    
                    # Parse the JSON object
                    json_obj = json.loads(decoded_line)
                    
                    # Extract the 'response' field and append it
                    fragment = json_obj.get("response", "")
                    full_response += fragment
                    
                    # Check if the response is done
                    if json_obj.get("done", False):
                        break  # Exit the loop when done
                except json.JSONDecodeError as json_err:
                    logging.error(f"JSON decode error for line: {line} | Error: {json_err}")
                    continue  # Skip invalid JSON lines
        return {"text": full_response.strip()}
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return {"error": f"HTTP error: {http_err}"}
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Connection error occurred: {conn_err}")
        return {"error": "Connection error. Please check if the Ollama server is running."}
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"Timeout error occurred: {timeout_err}")
        return {"error": "The request timed out. Please try again later."}
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request exception occurred: {req_err}")
        return {"error": "An unexpected error occurred."}

def chat_with_gpt(prompt, syllabus_data, session):
    """Chat with GPT using the local Ollama instance"""
    data = create_request_data(prompt, syllabus_data)
    response_data = make_request(OLLAMA_URL, data, session)
    if "error" in response_data:
        return f"An error occurred: {response_data['error']}"
    return response_data.get("text", "No response text found.").strip()

def setup_session():
    """Set up a requests session with retry strategy"""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def main():
    clear_terminal()
    print("Welcome to the EduBot! Type 'quit', 'exit', or 'bye' to end the conversation.")
    syllabus_data = load_syllabus_data()
    session = setup_session()
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break
            if not user_input:
                print("Please enter a valid input.")
                continue
            response = chat_with_gpt(user_input, syllabus_data, session)
            print("\nEduBot:", response, '\n')
    except KeyboardInterrupt:
        print("\nConversation ended by user.")
    finally:
        session.close()

if __name__ == "__main__":
    logging.basicConfig(filename='output.log', 
                    level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    main()