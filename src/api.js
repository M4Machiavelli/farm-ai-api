import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000'; // Change this to your API server IP if needed

export const getFarmRecommendation = async (data) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/predict/`, null, {
            params: data
        });
        return response.data;
    } catch (error) {
        console.error("API Error:", error);
        return { error: "Failed to get recommendation" };
    }
};

