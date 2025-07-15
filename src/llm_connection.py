import openai
import json
import os
from utils import writeChunksToFile

# # Replace this with your actual OpenAI API key
# api_key = "INSERT_KEY_HERE"
# openai.api_key = api_key
# # Replace this with your actual OpenAI API key
# api_key = "INSERT API KEY HERE"
# openai.api_key = api_key


def createQuestions(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You are an assistant. I am creating sample questions for a RAG context for a tutoring LLM. Please create a sample question where the answer is contained verbatim in the following text."
                "Be sure to respond with the question only. Only return with what the student would ask. Do not add any additional information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,  # Limit on tokens in the response
            temperature=0.7,  # Controls the creativity of the responses
        )

        return response['choices'][0]['message']['content'].strip()
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

def promptCreations(file_path, output_path):
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            syllabus_data = json.load(file)
    except FileNotFoundError:
        print(f"The file {file_path} was not found. Creating an empty file.")

    result = []
    for index, chunk in enumerate(syllabus_data):
        data = createQuestions(chunk) 
        result.append(data)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"An error occurred while writing to the output file: {e}")
    
    return result

# def main():
#     syllabus_name = 'syl_1'
#     chunk_size = 32
#     output_path = "/Users/willsaccount/Desktop/Tailored Tutor/src/Auto_Gen_Test_Questions/" + str(syllabus_name) + "_autoprompts_" + str(chunk_size) + ".json"
#     input_path = "Chunked_Syllabuses/" + str(syllabus_name) + "_chunk_" + str(chunk_size) + ".json"
#     promptCreations(input_path, output_path)

# if __name__ == "__main__":
#     main()
