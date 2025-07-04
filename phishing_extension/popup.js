document.getElementById('checkBtn').addEventListener('click', async () => {
  const resultDiv = document.getElementById('result');
  resultDiv.textContent = "Checking...";

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab.url;

    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });

    if (!res.ok) throw new Error(`API error: ${res.status}`);

    const data = await res.json();

    resultDiv.innerHTML = `
      <p><strong>URL:</strong> ${data.url}</p>
      <p><strong>Prediction:</strong> ${data.prediction}</p>
      <p><strong>Confidence:</strong> Legitimate ${(data.confidence.legitimate * 100).toFixed(2)}%, Phishing ${(data.confidence.phishing * 100).toFixed(2)}%</p>
    `;
  } catch (err) {
    resultDiv.textContent = `Error: ${err.message}`;
  }
});
