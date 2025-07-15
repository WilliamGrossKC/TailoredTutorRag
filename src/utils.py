import os
import json
import docx 
import pdfplumber 
import PyPDF2

def readJsonFile(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from the file {file_path}: {e}")
    return None

def extract_text_from_txt(txt_path):
    print("Opening File")
    start_time = time.time()
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    end_time = time.time() 
    return text


def readDocx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def readPdf(file_path):
    text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text())
    except Exception as e:
        print(f"Error reading PDF with pdfplumber: {e}")
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text.append(page.extract_text())
        except Exception as e:
            print(f"Error reading PDF with PyPDF2: {e}")
            return ''
    return '\n'.join(text)

    
def writeChunksToFile(chunks, file_path):
    foldername = os.path.dirname(file_path)
    
    if foldername and not os.path.exists(foldername):
        os.makedirs(foldername)
        
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(chunks, file, ensure_ascii=False, indent=4)
    print(f"Chunks written to {file_path}")