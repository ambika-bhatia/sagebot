from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

if __name__ == "__main__":
    pdf_path = "data/Dynamic_Memory_Allocation_C.pdf"  
    text = extract_text_from_pdf(pdf_path)
    print(f"Total characters extracted: {len(text)}")
    print(text[:500])