import React, { useState } from "react";
import { predictAQI } from "../services/api";

export default function PredictionForm() {
  const [inputs, setInputs] = useState({
    PM2_5: "",
    NO2: "",
    SO2: "",
    OZONE: "",
  });

  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  function handleChange(e) {
    const { name, value } = e.target;
    if (value === "" || /^[0-9]*\.?[0-9]*$/.test(value)) {
      setInputs((prev) => ({ ...prev, [name]: value }));
    }
  }

  function getAQIColor(aqi) {
    if (aqi <= 50) return "#22c55e";
    if (aqi <= 100) return "#eab308";
    if (aqi <= 150) return "#f97316";
    if (aqi <= 200) return "#ef4444";
    if (aqi <= 300) return "#8b5cf6";
    return "#7f1d1d";
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setPrediction(null);

    const { PM2_5, NO2, SO2, OZONE } = inputs;
    if (![PM2_5, NO2, SO2, OZONE].every((v) => v !== "")) {
      setError("Please fill all fields.");
      return;
    }

    const payload = {
      PM2_5: parseFloat(PM2_5),
      NO2: parseFloat(NO2),
      SO2: parseFloat(SO2),
      OZONE: parseFloat(OZONE),
    };

    try {
      setLoading(true);
      const resp = await predictAQI(payload);
      setPrediction(resp.AQI_prediction);
    } catch (err) {
      setError("Prediction failed. Check backend.");
    } finally {
      setLoading(false);
    }
  }

  function clearForm() {
    setInputs({ PM2_5: "", NO2: "", SO2: "", OZONE: "" });
    setError(null);
    setPrediction(null);
  }

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        display: "grid",
        gap: 18,
        background: "rgb(19, 25, 29)",
        padding: 22,
        borderRadius: 12,
        boxShadow: "2px 2px 10px black, 2px 2px 20px black",
        transition: "0.3s",
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}
      >
        {["PM2_5", "NO2", "SO2", "OZONE"].map((field) => (
          <label
            key={field}
            style={{ display: "flex", flexDirection: "column", color: "rgb(105, 208, 215)" }}
          >
            {field}
            <input
              name={field}
              value={inputs[field]}
              onChange={handleChange}
              placeholder={`Enter ${field}`}
              style={{
                marginTop: 6,
                padding: "10px",
                borderRadius: 8,
                border: "2px solid rgb(105, 208, 215)",
                background: "transparent",
                color: "rgb(105, 208, 215)",
              }}
              required
            />
          </label>
        ))}
      </div>

      <div style={{ display: "flex", gap: 10 }}>
        <button type="submit" disabled={loading} style={buttonStyle}>
          {loading ? "Predicting..." : "Predict AQI"}
        </button>
        <button type="button" onClick={clearForm} style={clearButtonStyle}>
          Clear
        </button>
      </div>

      {error && <div style={{ color: "red" }}>{error}</div>}

      {prediction !== null && (
        <div
          style={{
            padding: 16,
            borderRadius: 10,
            background: getAQIColor(prediction),
            color: "#fff",
            textAlign: "center",
            fontWeight: 700,
          }}
        >
          AQI: {prediction.toFixed(2)}
        </div>
      )}
    </form>
  );
}

const buttonStyle = {
  flex: 1,
  padding: "12px",
  borderRadius: 8,
  background: "rgb(105, 208, 215)",
  color: "black",
  border: "2px solid rgb(105, 208, 215)",
  cursor: "pointer",
  fontWeight: 600,
};

const clearButtonStyle = {
  padding: "12px",
  borderRadius: 8,
  border: "2px solid rgb(105, 208, 215)",
  background: "transparent",
  color: "rgb(105, 208, 215)",
  cursor: "pointer",
  fontWeight: 600,
};
