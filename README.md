# Voice-Based Concept Understanding Analyser (VBCUA)

This repository is formatted according to the official **AI-ML-and-GEN-AI-Track-Project-Template** structure.

---

## 📂 Repository Structure

The project deliverables are organized across the following phase directories:

1. **📁 1. Brainstorming & Ideation**
   - High-level overview of the conceptual idea, problem statement, and brainstorming results.
2. **📁 2. Requirement Analysis**
   - Detailed functional and non-functional requirements, and software prerequisites.
3. **📁 3. Project Design Phase**
   - Database schemas, Entity-Relationship diagrams, and module interactions.
4. **📁 4. Project Planning Phase**
   - Implementation plan, project milestones, and task checklists.
5. **📁 5. Project Development Phase**
   - The entire production codebase including the Streamlit frontend (`app.py`), FastAPI backend (`server.py`), and module libraries.
6. **📁 6. Project Testing**
   - Automated unit and integration test scripts (`test_pipeline.py`).
7. **📁 7. Project Documentation**
   - End-to-end user manuals, reports, and walkthrough files.

---

## 🛠️ Getting Started & Launching VBCUA

Follow these commands to run and test the project locally.

### 1. Inital Virtual Environment Setup (Run from root)
```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 2. Install Project Dependencies & Init DB
First, navigate into the Development phase directory:
```bash
cd "5. Project Development Phase"

# Install all required modules
pip install -r requirements.txt

# Initialize the 10-table SQLite database schema
python init_db.py
```

### 3. Run Automated Tests
To run all 38 unit and integration tests from the Testing phase directory:
```bash
cd ../"6. Project Testing"
python test_pipeline.py
```

### 4. Run the Servers (FastAPI Backend + Streamlit Frontend)
Navigate back into the Development folder:
```bash
cd ../"5. Project Development Phase"
```

* **Start the FastAPI Backend Service** (runs on port 8000):
  ```bash
  uvicorn server:app --reload --port 8000
  ```
* **Start the Streamlit Frontend Dashboard** (runs on port 8501):
  ```bash
  streamlit run app.py --server.port 8501
  ```
