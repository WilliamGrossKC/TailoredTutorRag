import json
from rag_implementation import ragAlgo
from syllabus_chunking import chunkSyllabus
from llm_connection import promptCreations

SYLLABUS_PATH = "/Users/willsaccount/Desktop/Tailored Tutor/src/Syllabuses/"
CHUNK_PATH_CONST = "/Users/willsaccount/Desktop/Tailored Tutor/src/Chunked_Syllabuses/"
PROMPT_PATH_CONST = "/Users/willsaccount/Desktop/Tailored Tutor/src/Auto_Gen_Test_Questions/"
SYLLABI = ["ACC 300 Syllabus FS24 2024 08 23.docx", "Syllabus PHYS2213 Sp24.pdf", "Syllabus_FINA5210_Spring 2024.pdf"]

def main():
    # chunkFilesOnly()
    # promptFilesOnly()
    loopedPipeline()

def promptFilesOnly():
    chunk_sizes = [16, 32, 64, 128, 256]

    for syllabus in SYLLABI:
        for chunk_size in chunk_sizes:
            chunk_file = CHUNK_PATH_CONST + "chunked_" + str(syllabus) + "_" + str(chunk_size) + ".json"
            prompt_file = PROMPT_PATH_CONST + "autoprompts_" + str(syllabus) + "_" + str(chunk_size) + ".json"
            promptCreations(chunk_file, prompt_file)

def chunkFilesOnly():
    chunk_sizes = [16, 32, 64, 128, 256]
    
    for syllabus in SYLLABI:
        for chunk_size in chunk_sizes:
            syllabus_file = SYLLABUS_PATH + syllabus
            chunk_file = CHUNK_PATH_CONST + "chunked_" + str(syllabus) + "_" + str(chunk_size) + ".json"
            chunkSyllabus(syllabus_file, chunk_file, chunk_size = chunk_size)

def fullPipeline(syllabus_name, chunk_size = 32, overlap_size = 0, k_value = 8, algo_type = 0, prompts_needed = True, chunks_needed = True):

    syllabus_file = SYLLABUS_PATH + syllabus_name
    chunk_file = CHUNK_PATH_CONST + "chunked_" + str(syllabus_name) + "_" + str(chunk_size) + ".json"
    prompt_file = PROMPT_PATH_CONST + "autoprompts_" + str(syllabus_name) + "_" + str(chunk_size) + ".json"
    # chunked_txt = chunk_syllabus(["ACC 300 Syllabus FS24 2024 08 23.docx"], chunk_size = 32)

    if chunks_needed:
        chunked_txt = chunkSyllabus(chunk_file, prompt_file)
    else:
        try:
            with open(chunk_file, 'r') as f:
                chunked_txt = json.load(f)
        except FileNotFoundError:
            print("Prompt file not found, creating prompts.")
            chunked_txt = chunkSyllabus(chunk_file, prompt_file)

    if prompts_needed:
        prompts = promptCreations(chunk_file, prompt_file)
    else:
        try:
            with open(prompt_file, 'r') as f:
                prompts = json.load(f)
        except FileNotFoundError:
            print("Prompt file not found, creating prompts.")
            prompts = promptCreations(chunk_file, prompt_file)
    
    for i in range(5):
        print(str(i))
        if i != 0:
            ragAlgo(chunked_txt, prompts, chunk_size = chunk_size, k_value = k_value, algo_type = i)

def loopedPipeline():
    chunk_sizes = [16, 32, 64, 128, 256]
    k_values = [1, 3, 6]

    for syllabus in SYLLABI:
        print(str(syllabus) + "\n")
        for chunk_size in chunk_sizes:
            for k_value in k_values:
                fullPipeline(syllabus_name=syllabus, chunk_size=chunk_size, overlap_size=0, k_value=k_value, algo_type=0, prompts_needed=False, chunks_needed=False)


if __name__ == "__main__":
    main()


