


![Logo](img/logo_resized.png)

# 🐾 ARQA - Animal Remedies Question Answering 🚀

**ARQA** is an ongoing project that leverages **artificial intelligence (AI)** and **Large Language Models (LLMs)** to answer questions about **veterinary medicines**. It is an adaptation of the [MeQA](https://github.com/jesusSant/GAMe/tree/main) system, which focuses on human-use medicines.

🔍 The name refers to **Noah’s Ark**, symbolizing a **repository of knowledge** about animal health.


## 🏗️ Approach

### 📄 PDF to Text Conversion

Evaluation of tools for extracting information from technical documents:

🔹 **[Mistral-OCR](https://mistral.ai/news/mistral-ocr)** → Uses LLM. Advantages: *To be added*. Disadvantages: *To be added*.  
🔹 **[Unstructured](https://unstructured.io/)** → Outputs in JSON. Advantages: *To be added*. Disadvantages: *To be added*.  
🔹 **[Marker](https://github.com/VikParuchuri/marker)** → Outputs in Markdown/JSON. Advantages: *To be added*. Disadvantages: *To be added*.  
🔹 **[PyMuPDF4LLM](https://github.com/pymupdf/RAG)** → Outputs in Markdown. Advantages: *To be added*. Disadvantages: *To be added*.  
🔹 **[olmOCR](https://olmocr.allenai.org/)** → Uses LLM. Advantages: *To be added*. Disadvantages: *To be added*.

The extracted information should be structured and parsable in JSON format. This includes identifying and organizing elements such as titles, sections, and subsections.

### 🧠 Vector Conversion and Storage

**FAISS** is planned as the vector database for efficient information storage. **ARQA** will require storing all vectors in advance. This is because semantic similarity will be essential for information retrieval, as questions do not necessarily have to mention medicine names explicitly.

---

## 📚 Resources

🔎 **Veterinary medicine Q&A corpus:**
   - 🌍 [JustAnswer](https://www.justanswer.com/veterinary/) → Questions in English.
   - 💬 [Cosmos Veterinary Medicine Forum](https://www.cosmos.com.mx/foros/medicamentos-veterinarios-d0w8.html)

📄 **AEMPS Information Sources:**
   - 🌐 **CIMA-Vet** → SmPCs downloads.
   - 🌐 **Regulatory Information** → Structured data (supply issues, safety concerns, etc.).




