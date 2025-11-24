import axios from "axios";

export const API_URL = "https://aqi-predictor-0bdu.onrender.com";


export const predictAQI = async (data) => {
  const response = await axios.post(`${API_URL}/predict`, {
    PM2_5: parseFloat(data.PM2_5),
    NO2: parseFloat(data.NO2),
    SO2: parseFloat(data.SO2),
    OZONE: parseFloat(data.OZONE)
  });

  return response.data;
};
