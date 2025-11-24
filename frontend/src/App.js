import React from "react";
import PredictionForm from "./components/PredictionForm";

export default function App() {
  return (
    <div
      style={{
        width: "100%",
        maxWidth: 760,
        marginTop: 80,
        background: "rgb(19, 25, 29)",
        padding: "2rem",
        borderRadius: 16,
        boxShadow: "2px 2px 10px black, 2px 2px 20px black",
        color: "rgb(105, 208, 215)",
        transition: "0.3s",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow =
          "2px 2px 10px rgb(105, 208, 215), 2px 2px 20px rgb(105, 208, 215)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow =
          "2px 2px 10px black, 2px 2px 20px black";
      }}
    >
      <header style={{ textAlign: "center", marginBottom: 20 }}>
        <h1 style={{ margin: 0, fontSize: 32, fontWeight: 700 }}>
          🌫️ AQI Predictor
        </h1>
        <p style={{ marginTop: 6, color: "rgb(105, 208, 215)" }}>
          Enter pollutant values to get AQI prediction.
        </p>
      </header>

      <main>
        <PredictionForm />
      </main>

      <footer
        style={{
          marginTop: 30,
          textAlign: "center",
          fontSize: 14,
          color: "rgb(105, 208, 215)",
          borderTop: "1px solid rgb(105, 208, 215)",
          paddingTop: 10,
        }}
      >
        Built with ❤️ — Saurabh Kumar
      </footer>
    </div>
  );
}
