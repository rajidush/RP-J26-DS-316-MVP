# Offline Socratic Child Safety Engine

This is the offline Python backend (The Cognitive Engine) designed to run completely locally, protecting children's privacy by maintaining **zero data transit** (100% offline). It acts as a deterministic pedagogical state machine that guides minors through a 3-step cognitive scaffold when a safety threat is intercepted on their screen.

## Prerequisites

1. **Python 3.8 - 3.11** installed.
2. **LM Studio** installed on your local computer.
3. Download a Small Language Model (SLM) such as **google/gemma-3-1b** (or **Llama-3.2-1B-Instruct**) via LM Studio.
4. In LM Studio, click on the **Local Server** icon (left panel) and click **Start Server**. It runs on `http://localhost:1234` by default. Make sure your model (e.g. `google/gemma-3-1b`) is loaded into memory.

## Python Backend Setup

1. Open your terminal in this `offline_backend` folder.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows (Command Prompt):
   venv\Scripts\activate
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   python main.py
   ```
   The backend will start running on `http://127.0.0.1:8000`.

## Integrating with the React Frontend

The React frontend (built in Next.js) can connect directly to this local backend.

1. Open a new terminal in the root folder of the Next.js / React application.
2. Install the frontend dependencies:
   ```bash
   npm install
   ```
3. Run the React development server:
   ```bash
   npm run dev
   ```
4. Open the web interface in your browser (usually `http://localhost:3000`).
5. Toggle the **Connection Mode** to **Local Python Backend** in the top control panel of the dashboard. This directs the frontend's interception logic and chat turns to your local `http://127.0.0.1:8000` FastAPI server instead of the simulated cloud API!

## Architecture Design

* **The Socratic State Machine**: Controls dialogue sequence deterministically (`Acknowledge` -> `Reason` -> `Contract`). The LLM is restricted to a very low temperature (`temperature=0.2`) to eliminate hallucination or drift.
* **Dynamic Age-Based Routing**: Checks `child_age` dynamically:
  * **Age <= 10 (Protective Prompt)**: Simple, reassuring words, strict scaffolding, and exactly one elementary, safe question.
  * **Age >= 11 (Autonomy & Negotiation)**: High-level critical thinking, deep respectful inquiries, negotiating boundaries as a peer.
* **Sliding Window Memory**: Keeps only the most recent user-assistant turns, ensuring the 1B/3B model does not suffer from context length overflow, and remains highly focused on safety directives.
