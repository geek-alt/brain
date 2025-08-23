import React, { useState } from "react";

export default function App() {
  const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("user_id", "default");

    try {
      const res = await fetch(`${API}/upload/`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      alert(data.message || "File uploaded successfully");
    } catch (err) {
      alert("Upload failed: " + err.message);
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) {
      alert("Please enter a question.");
      return;
    }

    try {
      const res = await fetch(`${API}/query/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: question,
          model: "gpt-oss-20b", // your local model
          explanation_level: "intermediate",
          language: "English",
          user_id: "default",
        }),
      });
      const data = await res.json();
      setAnswer(data.answer);
      setSources(data.source_documents || []);
    } catch (err) {
      alert("Query failed: " + err.message);
    }
  };

  return (
    <div style={{ maxWidth: "600px", margin: "auto", padding: "20px" }}>
      <h1>📚 Document Q&A</h1>

      <div style={{ marginBottom: "15px" }}>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleUpload} style={{ marginLeft: "10px" }}>
          Upload
        </button>
      </div>

      <div style={{ marginBottom: "15px" }}>
        <textarea
          placeholder="Ask a question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          style={{ width: "100%", minHeight: "80px" }}
        />
        <br />
        <button onClick={handleAsk} style={{ marginTop: "10px" }}>
          Ask
        </button>
      </div>

      {answer && (
        <div
          style={{
            marginTop: "20px",
            padding: "10px",
            border: "1px solid #ddd",
            borderRadius: "8px",
          }}
        >
          <h2>Answer</h2>
          <p>{answer}</p>
          {sources.length > 0 && (
            <>
              <h3>Sources:</h3>
              <ul>
                {sources.map((s, i) => (
                  <li key={i}>
                    {s.metadata?.source} (p. {s.metadata?.page})
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}
